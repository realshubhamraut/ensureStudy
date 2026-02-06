"""
Interview Questions API Routes

CRUD operations for descriptive interview questions.
Questions are AI-generated and stored per topic.
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
from app import db
from app.models.interview_questions import DescriptiveQuestion, StudentQuestionAttempt
from app.models.curriculum import ClassroomTopic
from app.routes.users import require_auth

interview_questions_bp = Blueprint('interview_questions', __name__, url_prefix='/api/interview-questions')


@interview_questions_bp.route('/topic/<topic_id>', methods=['GET'])
@require_auth
def get_questions_for_topic(topic_id):
    """
    Get all active descriptive questions for a topic.
    
    Query params:
        - limit: Max questions to return (default 10)
        - exclude_answered: Exclude questions user already answered in a session
        - session_id: Session ID to check for answered questions
    """
    user_id = request.user_id
    limit = min(int(request.args.get('limit', 10)), 50)
    exclude_answered = request.args.get('exclude_answered', 'false').lower() == 'true'
    session_id = request.args.get('session_id')
    
    # Verify topic exists
    topic = ClassroomTopic.query.get(topic_id)
    if not topic:
        return jsonify({'success': False, 'error': 'Topic not found'}), 404
    
    # Get questions
    query = DescriptiveQuestion.query.filter_by(
        topic_id=topic_id,
        is_active=True
    )
    
    # Optionally exclude already answered questions in this session
    if exclude_answered and session_id:
        answered_ids = db.session.query(StudentQuestionAttempt.question_id).filter_by(
            user_id=user_id,
            session_id=session_id
        ).subquery()
        query = query.filter(~DescriptiveQuestion.id.in_(answered_ids))
    
    # Order by least asked first (distribute usage)
    questions = query.order_by(DescriptiveQuestion.times_asked.asc()).limit(limit).all()
    
    return jsonify({
        'success': True,
        'topic': {
            'id': topic.id,
            'name': topic.name,
            'description': topic.description
        },
        'questions': [q.to_interview_dict() for q in questions],
        'count': len(questions),
        'total_available': DescriptiveQuestion.query.filter_by(topic_id=topic_id, is_active=True).count()
    })


@interview_questions_bp.route('/topics/batch', methods=['POST'])
@require_auth
def get_questions_for_topics():
    """
    Get questions for multiple topics at once.
    Used when starting a mock interview with multiple selected topics.
    
    Body:
        {
            "topic_ids": ["id1", "id2"],
            "questions_per_topic": 3,
            "session_id": "optional"
        }
    
    Note: Returns expected_answer and key_concepts for AI-service evaluation.
    Frontend should NOT display expected_answer to students.
    """
    user_id = request.user_id
    data = request.get_json() or {}
    
    topic_ids = data.get('topic_ids', [])
    questions_per_topic = min(int(data.get('questions_per_topic', 3)), 10)
    session_id = data.get('session_id')
    
    if not topic_ids:
        return jsonify({'success': False, 'error': 'topic_ids required'}), 400
    
    all_questions = []
    topics_info = []
    
    for topic_id in topic_ids:
        topic = ClassroomTopic.query.get(topic_id)
        if not topic:
            continue
        
        topics_info.append({
            'id': topic.id,
            'name': topic.name,
            'description': topic.description
        })
        
        # Get questions for this topic
        query = DescriptiveQuestion.query.filter_by(
            topic_id=topic_id,
            is_active=True
        )
        
        # Exclude answered in this session
        if session_id:
            answered_ids = db.session.query(StudentQuestionAttempt.question_id).filter_by(
                user_id=user_id,
                session_id=session_id
            ).subquery()
            query = query.filter(~DescriptiveQuestion.id.in_(answered_ids))
        
        questions = query.order_by(DescriptiveQuestion.times_asked.asc()).limit(questions_per_topic).all()
        
        for q in questions:
            # Merge interview dict (for display) with evaluation dict (for AI)
            q_dict = {
                **q.to_interview_dict(),
                **q.to_evaluation_dict(),  # Adds reference_answer and key_concepts
                'topic_name': topic.name
            }
            all_questions.append(q_dict)
    
    return jsonify({
        'success': True,
        'topics': topics_info,
        'questions': all_questions,
        'count': len(all_questions)
    })


@interview_questions_bp.route('/generate', methods=['POST'])
@require_auth
def generate_questions():
    """
    Generate new questions for topics using AI.
    Called when question pool is low or on-demand.
    
    Body:
        {
            "topics": [{"id": "...", "name": "...", "description": "..."}],
            "questions_per_topic": 5,
            "difficulty": "medium"
        }
    """
    data = request.get_json() or {}
    
    topics = data.get('topics', [])
    questions_per_topic = min(int(data.get('questions_per_topic', 5)), 10)
    difficulty = data.get('difficulty', 'medium')
    
    if not topics:
        return jsonify({'success': False, 'error': 'topics required'}), 400
    
    try:
        import asyncio
        from app.agents.interview_question_agent import get_interview_question_agent
        
        agent = get_interview_question_agent()
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(agent.generate({
            'topics': topics,
            'questions_per_topic': questions_per_topic,
            'difficulty': difficulty
        }))
        loop.close()
        
        if not result.get('success'):
            return jsonify({'success': False, 'error': result.get('error', 'Generation failed')}), 500
        
        # Store generated questions in database
        stored_count = 0
        for q in result.get('questions', []):
            question = DescriptiveQuestion(
                topic_id=q.get('topic_id'),
                question_text=q.get('question'),
                expected_answer=q.get('expected_answer'),
                key_concepts=q.get('key_concepts', []),
                difficulty=q.get('difficulty', difficulty),
                source='ai_generated'
            )
            db.session.add(question)
            stored_count += 1
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'generated': result.get('count', 0),
            'stored': stored_count,
            'message': f'Generated and stored {stored_count} questions'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@interview_questions_bp.route('/<question_id>/stats', methods=['PUT'])
@require_auth
def update_question_stats(question_id):
    """
    Update question statistics after an answer is evaluated.
    
    Body:
        {
            "score": 85.5,
            "session_id": "...",
            "student_answer": "...",
            "feedback": "...",
            "concept_scores": {...},
            "response_time_seconds": 120
        }
    """
    user_id = request.user_id
    data = request.get_json() or {}
    
    question = DescriptiveQuestion.query.get(question_id)
    if not question:
        return jsonify({'success': False, 'error': 'Question not found'}), 404
    
    score = data.get('score', 0)
    
    # Update question stats
    question.update_stats(score)
    
    # Record attempt
    attempt = StudentQuestionAttempt(
        user_id=user_id,
        question_id=question_id,
        session_id=data.get('session_id'),
        student_answer=data.get('student_answer'),
        score=score,
        concept_scores=data.get('concept_scores'),
        feedback=data.get('feedback'),
        response_time_seconds=data.get('response_time_seconds'),
        audio_duration_seconds=data.get('audio_duration_seconds')
    )
    db.session.add(attempt)
    
    # Update StudentTopicScore to aggregate with MCQ scores
    from app.models.curriculum import StudentTopicScore
    
    topic_score = StudentTopicScore.query.filter_by(
        user_id=user_id,
        classroom_topic_id=question.topic_id
    ).first()
    
    if not topic_score:
        # Create new score record if doesn't exist
        topic_score = StudentTopicScore(
            user_id=user_id,
            classroom_topic_id=question.topic_id
        )
        db.session.add(topic_score)
    
    # Update descriptive score (score is 0-100, max is 100)
    topic_score.update_descriptive_score(score_awarded=score, max_score=100)
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'question_stats': {
            'times_asked': question.times_asked,
            'avg_score': round(question.avg_score, 1)
        },
        'topic_score': {
            'mastery_percentage': round(topic_score.mastery_percentage, 1),
            'status': topic_score.status,
            'descriptive_avg_score': round(topic_score.descriptive_avg_score, 1)
        },
        'attempt_id': attempt.id
    })


@interview_questions_bp.route('/topic/<topic_id>/count', methods=['GET'])
@require_auth
def get_question_count(topic_id):
    """Get the number of questions available for a topic."""
    active_count = DescriptiveQuestion.query.filter_by(
        topic_id=topic_id,
        is_active=True
    ).count()
    
    total_count = DescriptiveQuestion.query.filter_by(topic_id=topic_id).count()
    
    return jsonify({
        'success': True,
        'topic_id': topic_id,
        'active_count': active_count,
        'total_count': total_count,
        'needs_generation': active_count < 5  # Threshold
    })


@interview_questions_bp.route('/user/history', methods=['GET'])
@require_auth
def get_user_question_history():
    """Get user's question attempt history."""
    user_id = request.user_id
    limit = min(int(request.args.get('limit', 20)), 100)
    
    attempts = StudentQuestionAttempt.query.filter_by(
        user_id=user_id
    ).order_by(StudentQuestionAttempt.attempted_at.desc()).limit(limit).all()
    
    return jsonify({
        'success': True,
        'attempts': [a.to_dict() for a in attempts],
        'count': len(attempts)
    })


@interview_questions_bp.route('/topic/<topic_id>/update-score', methods=['PUT'])
@require_auth
def update_topic_score(topic_id):
    """
    Update topic mastery score after mock interview answer.
    Works with both DB questions and dynamically generated ones.
    
    Body:
        {
            "score": 85.5,  # 0-100
            "session_id": "...",  # Optional
            "question_text": "...",  # Optional, for logging
            "student_answer": "..."  # Optional, for logging
        }
    """
    from app.models.curriculum import StudentTopicScore, ClassroomTopic
    
    user_id = request.user_id
    data = request.get_json() or {}
    
    score = float(data.get('score', 0))  # 0-100
    
    # Verify topic exists
    topic = ClassroomTopic.query.get(topic_id)
    if not topic:
        return jsonify({'success': False, 'error': 'Topic not found'}), 404
    
    # Get or create StudentTopicScore
    topic_score = StudentTopicScore.query.filter_by(
        user_id=user_id,
        classroom_topic_id=topic_id
    ).first()
    
    if not topic_score:
        topic_score = StudentTopicScore(
            user_id=user_id,
            classroom_topic_id=topic_id
        )
        db.session.add(topic_score)
    
    # Update descriptive score (score is 0-100, max is 100)
    topic_score.update_descriptive_score(score_awarded=score, max_score=100)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'topic_id': topic_id,
        'topic_name': topic.name,
        'topic_score': {
            'mastery_percentage': round(topic_score.mastery_percentage, 1),
            'status': topic_score.status,
            'descriptive_attempts': topic_score.descriptive_attempts,
            'descriptive_avg_score': round(topic_score.descriptive_avg_score, 1)
        }
    })
