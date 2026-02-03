"""
Question Progress API Routes

Endpoints for tracking student question progress and triggering
the Type 5 Learning Agent for automatic question generation.
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
import logging

from app import db
from app.routes.users import require_auth
from app.utils.jwt_handler import verify_token
from app.models.user import User

logger = logging.getLogger(__name__)

question_progress_bp = Blueprint("question_progress", __name__, url_prefix="/api/questions")


def get_current_user():
    """Get current user from JWT token"""
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None
    
    try:
        token = auth_header.split()[1]
        payload = verify_token(token)
        return User.query.get(payload["user_id"])
    except:
        return None


@question_progress_bp.route("/progress/<topic_id>", methods=["GET"])
@require_auth
def get_topic_progress(topic_id):
    """
    Get question progress for a specific topic.
    
    Returns:
        - total_questions: Total questions available for topic
        - questions_attempted: Questions the student has answered
        - attempt_percentage: Percentage of questions attempted
        - should_generate: True if 80% threshold reached
        - learning_agent_status: Status of the Learning Agent for this topic
    """
    try:
        from app.models.curriculum import TopicQuestion, StudentQuestionResponse, LearningAgentMemory
        
        user = get_current_user()
        user_id = user.id if user else None
        
        # Get total questions for topic
        total_questions = TopicQuestion.query.filter_by(
            classroom_topic_id=topic_id,
            is_active=True
        ).count()
        
        # Get questions this student has attempted
        question_ids = db.session.query(TopicQuestion.id).filter_by(
            classroom_topic_id=topic_id,
            is_active=True
        ).subquery()
        
        questions_attempted = db.session.query(StudentQuestionResponse).filter(
            StudentQuestionResponse.user_id == user_id,
            StudentQuestionResponse.question_id.in_(question_ids)
        ).distinct(StudentQuestionResponse.question_id).count()
        
        # Calculate percentage
        attempt_percentage = (questions_attempted / total_questions * 100) if total_questions > 0 else 0
        
        # Check learning agent memory
        memory = LearningAgentMemory.query.filter_by(topic_id=topic_id).first()
        
        learning_status = {
            "active": True,
            "learning_iterations": memory.learning_iterations if memory else 0,
            "calibrated_difficulty": memory.calibrated_difficulty if memory else 0.5,
            "last_learning_at": memory.last_learning_at.isoformat() if memory and memory.last_learning_at else None
        }
        
        return jsonify({
            "topic_id": topic_id,
            "total_questions": total_questions,
            "questions_attempted": questions_attempted,
            "attempt_percentage": round(attempt_percentage, 1),
            "should_generate": attempt_percentage >= 80,
            "learning_agent_status": learning_status
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting topic progress: {e}")
        return jsonify({"error": str(e)}), 500


@question_progress_bp.route("/progress/classroom/<classroom_id>", methods=["GET"])
@require_auth
def get_classroom_progress(classroom_id):
    """
    Get question progress for all topics in a classroom.
    
    Returns progress data for each topic with Learning Agent status.
    """
    try:
        from app.models.curriculum import ClassroomTopic, TopicQuestion, StudentQuestionResponse, LearningAgentMemory
        
        user = get_current_user()
        user_id = user.id if user else None
        
        # Get all topics for classroom
        topics = ClassroomTopic.query.filter_by(
            classroom_id=classroom_id,
            is_active=True
        ).all()
        
        progress_data = []
        
        for topic in topics:
            # Get question counts
            total = TopicQuestion.query.filter_by(
                classroom_topic_id=topic.id,
                is_active=True
            ).count()
            
            # Get attempted count
            question_ids = db.session.query(TopicQuestion.id).filter_by(
                classroom_topic_id=topic.id,
                is_active=True
            ).subquery()
            
            attempted = db.session.query(StudentQuestionResponse).filter(
                StudentQuestionResponse.user_id == user_id,
                StudentQuestionResponse.question_id.in_(question_ids)
            ).distinct(StudentQuestionResponse.question_id).count()
            
            percentage = (attempted / total * 100) if total > 0 else 0
            
            # Get learning memory
            memory = LearningAgentMemory.query.filter_by(topic_id=topic.id).first()
            
            progress_data.append({
                "topic_id": topic.id,
                "topic_name": topic.name,
                "total_questions": total,
                "questions_attempted": attempted,
                "attempt_percentage": round(percentage, 1),
                "should_generate": percentage >= 80,
                "learning_agent": {
                    "iterations": memory.learning_iterations if memory else 0,
                    "difficulty": memory.calibrated_difficulty if memory else 0.5
                }
            })
        
        return jsonify({
            "classroom_id": classroom_id,
            "topics": progress_data,
            "total_topics": len(progress_data),
            "topics_at_threshold": sum(1 for p in progress_data if p["should_generate"])
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting classroom progress: {e}")
        return jsonify({"error": str(e)}), 500


@question_progress_bp.route("/trigger-generation/<topic_id>", methods=["POST"])
@require_auth
def trigger_generation(topic_id):
    """
    Manually trigger the Learning Agent to generate questions for a topic.
    
    Can be used when:
    - 80% threshold is reached
    - Admin wants to add more questions
    - Initial population of question bank
    """
    try:
        from app.models.curriculum import TopicQuestion, StudentQuestionResponse, ClassroomTopic
        
        user = get_current_user()
        
        # Check if topic exists
        topic = ClassroomTopic.query.get(topic_id)
        if not topic:
            return jsonify({"error": "Topic not found"}), 404
        
        # Get existing questions for duplicate checking
        existing_questions = TopicQuestion.query.filter_by(
            classroom_topic_id=topic_id,
            is_active=True
        ).all()
        
        existing_list = [q.to_dict(include_answer=True) for q in existing_questions]
        
        # Trigger Learning Agent asynchronously
        # In production, this would emit a Kafka event
        import asyncio
        from app.agents.learning_agent import get_learning_agent
        
        agent = get_learning_agent()
        
        result = asyncio.run(agent.execute({
            'task_type': 'generate',
            'topic_id': topic_id,
            'classroom_id': topic.classroom_id,
            'existing_questions': existing_list,
            'questions_attempted': len(existing_list),  # Force generation
            'total_questions': len(existing_list)
        }))
        
        # Store generated questions
        new_questions = result.get('questions', [])
        stored_count = 0
        
        for q_data in new_questions:
            new_question = TopicQuestion(
                classroom_topic_id=topic_id,
                question_type=q_data.get('question_type', 'mcq'),
                question_text=q_data['question_text'],
                options=q_data.get('options', []),
                correct_answer=q_data.get('correct_answer'),
                explanation=q_data.get('explanation'),
                difficulty=q_data.get('difficulty', 'medium'),
                question_hash=q_data.get('question_hash'),
                auto_generated=True,
                created_by=user.id if user else None
            )
            db.session.add(new_question)
            stored_count += 1
        
        db.session.commit()
        
        return jsonify({
            "success": True,
            "topic_id": topic_id,
            "questions_generated": len(new_questions),
            "questions_stored": stored_count,
            "learning_agent_output": result.get('data', {})
        }), 200
        
    except Exception as e:
        logger.error(f"Error triggering generation: {e}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@question_progress_bp.route("/effectiveness/<topic_id>", methods=["GET"])
@require_auth
def get_topic_effectiveness(topic_id):
    """
    Get effectiveness scores for questions in a topic.
    
    Returns top performing and underperforming questions.
    """
    try:
        from app.models.curriculum import TopicQuestion, QuestionEffectiveness
        
        # Get all questions with effectiveness data
        questions = db.session.query(TopicQuestion, QuestionEffectiveness).join(
            QuestionEffectiveness,
            TopicQuestion.id == QuestionEffectiveness.question_id,
            isouter=True
        ).filter(
            TopicQuestion.classroom_topic_id == topic_id,
            TopicQuestion.is_active == True
        ).all()
        
        effectiveness_data = []
        for question, eff in questions:
            data = {
                "question_id": question.id,
                "question_text": question.question_text[:100] + "..." if len(question.question_text) > 100 else question.question_text,
                "effectiveness_score": eff.effectiveness_score if eff else None,
                "discrimination_index": eff.discrimination_index if eff else None,
                "difficulty_index": eff.difficulty_index if eff else None,
                "sample_size": eff.sample_size if eff else 0
            }
            effectiveness_data.append(data)
        
        # Sort by effectiveness score
        effectiveness_data.sort(key=lambda x: x.get('effectiveness_score') or 0, reverse=True)
        
        return jsonify({
            "topic_id": topic_id,
            "total_questions": len(effectiveness_data),
            "questions_with_data": sum(1 for q in effectiveness_data if q["effectiveness_score"] is not None),
            "top_questions": effectiveness_data[:5],
            "underperforming_questions": [q for q in effectiveness_data if (q.get("effectiveness_score") or 0.5) < 0.3][:5]
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting effectiveness data: {e}")
        return jsonify({"error": str(e)}), 500
