"""
Curriculum Routes - Subject, Topic, Subtopic API
"""
from flask import Blueprint, request, jsonify
from app import db
from app.models.curriculum import Subject, Topic, Subtopic, SubtopicAssessment, StudentSubtopicProgress
from app.models.user import User
from app.utils.jwt_handler import verify_token

curriculum_bp = Blueprint("curriculum", __name__, url_prefix="/api/curriculum")


def auth_required(f):
    """Decorator to require authentication"""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "Missing authorization header"}), 401
        
        try:
            token = auth_header.split()[1]
            payload = verify_token(token)
            user = User.query.get(payload["user_id"])
            
            if not user:
                return jsonify({"error": "User not found"}), 404
            
            request.current_user = user
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({"error": f"Authentication failed: {str(e)}"}), 401
    
    return decorated


# ==================== Subjects ====================

@curriculum_bp.route("/subjects", methods=["GET"])
@auth_required
def list_subjects():
    """Get all subjects"""
    grade = request.args.get("grade")
    board = request.args.get("board")
    
    query = Subject.query.filter_by(is_active=True)
    
    subjects = query.order_by(Subject.order).all()
    
    # Filter by grade/board if user is student with profile
    user = request.current_user
    if user.role == "student" and user.student_profile:
        profile = user.student_profile
        if profile.subjects:
            subjects = [s for s in subjects if s.name in profile.subjects or s.code in profile.subjects]
    
    return jsonify({
        "subjects": [s.to_dict() for s in subjects],
        "count": len(subjects)
    }), 200


@curriculum_bp.route("/subjects/<subject_id>", methods=["GET"])
@auth_required
def get_subject(subject_id):
    """Get subject with topics"""
    subject = Subject.query.get(subject_id)
    
    if not subject:
        return jsonify({"error": "Subject not found"}), 404
    
    return jsonify({
        "subject": subject.to_dict(include_topics=True)
    }), 200


# ==================== Topics ====================

@curriculum_bp.route("/subjects/<subject_id>/topics", methods=["GET"])
@auth_required
def list_topics(subject_id):
    """Get topics for a subject"""
    subject = Subject.query.get(subject_id)
    
    if not subject:
        return jsonify({"error": "Subject not found"}), 404
    
    topics = subject.topics.filter_by(is_active=True).order_by(Topic.order).all()
    
    # Include progress for students
    user = request.current_user
    result = []
    
    for topic in topics:
        topic_data = topic.to_dict()
        
        if user.role == "student":
            # Calculate topic progress
            subtopic_count = topic.subtopics.count()
            completed_count = StudentSubtopicProgress.query.join(Subtopic)\
                .filter(
                    Subtopic.topic_id == topic.id,
                    StudentSubtopicProgress.user_id == user.id,
                    StudentSubtopicProgress.status.in_(["completed", "mastered"])
                ).count()
            
            topic_data["progress"] = {
                "total_subtopics": subtopic_count,
                "completed": completed_count,
                "percentage": (completed_count / subtopic_count * 100) if subtopic_count > 0 else 0
            }
        
        result.append(topic_data)
    
    return jsonify({
        "topics": result,
        "count": len(result)
    }), 200


@curriculum_bp.route("/topics/<topic_id>", methods=["GET"])
@auth_required
def get_topic(topic_id):
    """Get topic with subtopics"""
    topic = Topic.query.get(topic_id)
    
    if not topic:
        return jsonify({"error": "Topic not found"}), 404
    
    return jsonify({
        "topic": topic.to_dict(include_subtopics=True)
    }), 200


# ==================== Subtopics ====================

@curriculum_bp.route("/topics/<topic_id>/subtopics", methods=["GET"])
@auth_required
def list_subtopics(topic_id):
    """Get subtopics for a topic"""
    topic = Topic.query.get(topic_id)
    
    if not topic:
        return jsonify({"error": "Topic not found"}), 404
    
    subtopics = topic.subtopics.filter_by(is_active=True).order_by(Subtopic.order).all()
    
    user = request.current_user
    result = []
    
    for subtopic in subtopics:
        subtopic_data = subtopic.to_dict()
        
        if user.role == "student":
            progress = StudentSubtopicProgress.query.filter_by(
                user_id=user.id,
                subtopic_id=subtopic.id
            ).first()
            
            subtopic_data["progress"] = progress.to_dict() if progress else {
                "status": "not_started",
                "attempts": 0,
                "best_score": 0
            }
        
        result.append(subtopic_data)
    
    return jsonify({
        "subtopics": result,
        "count": len(result)
    }), 200


@curriculum_bp.route("/subtopics/<subtopic_id>", methods=["GET"])
@auth_required
def get_subtopic(subtopic_id):
    """Get subtopic details"""
    subtopic = Subtopic.query.get(subtopic_id)
    
    if not subtopic:
        return jsonify({"error": "Subtopic not found"}), 404
    
    data = subtopic.to_dict()
    
    # Include user progress
    user = request.current_user
    if user.role == "student":
        progress = StudentSubtopicProgress.query.filter_by(
            user_id=user.id,
            subtopic_id=subtopic.id
        ).first()
        
        data["progress"] = progress.to_dict() if progress else None
    
    return jsonify({"subtopic": data}), 200


# ==================== Assessments ====================

@curriculum_bp.route("/subtopics/<subtopic_id>/assessment", methods=["GET"])
@auth_required
def get_assessment(subtopic_id):
    """Get MCQ assessment for a subtopic"""
    subtopic = Subtopic.query.get(subtopic_id)
    
    if not subtopic:
        return jsonify({"error": "Subtopic not found"}), 404
    
    assessment = SubtopicAssessment.query.filter_by(
        subtopic_id=subtopic_id,
        is_active=True
    ).first()
    
    if not assessment:
        return jsonify({"error": "No assessment available"}), 404
    
    # Return assessment with questions (without correct answers for students)
    data = assessment.to_dict(include_questions=True)
    
    user = request.current_user
    if user.role == "student":
        # Hide correct answers
        for q in data.get("questions", []):
            q.pop("correct_answer", None)
            q.pop("explanation", None)
    
    return jsonify({"assessment": data}), 200


@curriculum_bp.route("/subtopics/<subtopic_id>/assessment/submit", methods=["POST"])
@auth_required
def submit_assessment(subtopic_id):
    """Submit assessment answers and get results"""
    user = request.current_user
    
    if user.role != "student":
        return jsonify({"error": "Only students can submit assessments"}), 403
    
    subtopic = Subtopic.query.get(subtopic_id)
    if not subtopic:
        return jsonify({"error": "Subtopic not found"}), 404
    
    assessment = SubtopicAssessment.query.filter_by(
        subtopic_id=subtopic_id,
        is_active=True
    ).first()
    
    if not assessment:
        return jsonify({"error": "No assessment available"}), 404
    
    data = request.get_json()
    answers = data.get("answers", {})  # {question_id: answer_index}
    time_taken = data.get("time_taken", 0)  # seconds
    
    # Grade assessment
    correct = 0
    total = len(assessment.questions)
    results = []
    
    for question in assessment.questions:
        q_id = question.get("id")
        user_answer = answers.get(q_id)
        correct_answer = question.get("correct_answer")
        is_correct = user_answer == correct_answer
        
        if is_correct:
            correct += 1
        
        results.append({
            "question_id": q_id,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "explanation": question.get("explanation")
        })
    
    score = (correct / total * 100) if total > 0 else 0
    passed = score >= assessment.passing_score
    
    # Update progress
    from datetime import datetime
    progress = StudentSubtopicProgress.query.filter_by(
        user_id=user.id,
        subtopic_id=subtopic_id
    ).first()
    
    if not progress:
        progress = StudentSubtopicProgress(
            user_id=user.id,
            subtopic_id=subtopic_id,
            first_attempt_at=datetime.utcnow()
        )
        db.session.add(progress)
    
    progress.attempts += 1
    progress.last_score = score
    progress.last_attempt_at = datetime.utcnow()
    progress.total_time_minutes += time_taken // 60
    
    if score > progress.best_score:
        progress.best_score = score
    
    progress.average_score = (
        (progress.average_score * (progress.attempts - 1) + score) / progress.attempts
    )
    
    # Update status
    if passed:
        if score >= 90:
            progress.status = "mastered"
            progress.mastery_level = min(100, progress.mastery_level + 20)
        else:
            progress.status = "completed"
            progress.mastery_level = max(progress.mastery_level, score)
        progress.completed_at = datetime.utcnow()
    else:
        progress.status = "in_progress"
    
    db.session.commit()
    
    return jsonify({
        "score": score,
        "correct": correct,
        "total": total,
        "passed": passed,
        "passing_score": assessment.passing_score,
        "results": results,
        "progress": progress.to_dict()
    }), 200


# ==================== Student Progress ====================

@curriculum_bp.route("/progress", methods=["GET"])
@auth_required
def get_progress():
    """Get student's overall curriculum progress"""
    user = request.current_user
    
    if user.role != "student":
        return jsonify({"error": "Only students can view progress"}), 403
    
    # Get all progress records
    progress_records = StudentSubtopicProgress.query.filter_by(user_id=user.id).all()
    
    # Calculate by subject
    subject_progress = {}
    
    for record in progress_records:
        subtopic = Subtopic.query.get(record.subtopic_id)
        if not subtopic:
            continue
        
        topic = Topic.query.get(subtopic.topic_id)
        if not topic:
            continue
        
        subject = Subject.query.get(topic.subject_id)
        if not subject:
            continue
        
        if subject.name not in subject_progress:
            subject_progress[subject.name] = {
                "subject_id": subject.id,
                "total_subtopics": 0,
                "completed": 0,
                "in_progress": 0,
                "mastered": 0,
                "average_score": 0,
                "scores": []
            }
        
        subject_progress[subject.name]["scores"].append(record.best_score)
        
        if record.status in ["completed", "mastered"]:
            subject_progress[subject.name]["completed"] += 1
        if record.status == "mastered":
            subject_progress[subject.name]["mastered"] += 1
        if record.status == "in_progress":
            subject_progress[subject.name]["in_progress"] += 1
    
    # Calculate averages
    for name, data in subject_progress.items():
        if data["scores"]:
            data["average_score"] = sum(data["scores"]) / len(data["scores"])
        del data["scores"]
    
    return jsonify({
        "by_subject": subject_progress,
        "total_completed": sum(s["completed"] for s in subject_progress.values()),
        "total_mastered": sum(s["mastered"] for s in subject_progress.values())
    }), 200
