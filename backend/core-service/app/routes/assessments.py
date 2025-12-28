"""
Assessment Routes
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
from uuid import uuid4
from app import db
from app.models.user import Assessment, AssessmentResult, Progress
from app.routes.users import require_auth, require_teacher

assessments_bp = Blueprint("assessments", __name__, url_prefix="/api/assessments")


@assessments_bp.route("/", methods=["GET"])
@require_auth
def list_assessments():
    """List available assessments"""
    subject = request.args.get("subject")
    difficulty = request.args.get("difficulty")
    
    query = Assessment.query
    
    if subject:
        query = query.filter_by(subject=subject)
    if difficulty:
        query = query.filter_by(difficulty=difficulty)
    
    assessments = query.order_by(Assessment.created_at.desc()).limit(50).all()
    
    return jsonify({
        "assessments": [a.to_dict() for a in assessments],
        "count": len(assessments)
    }), 200


@assessments_bp.route("/<assessment_id>", methods=["GET"])
@require_auth
def get_assessment(assessment_id):
    """Get assessment details"""
    assessment = Assessment.query.get(assessment_id)
    
    if not assessment:
        return jsonify({"error": "Assessment not found"}), 404
    
    return jsonify({"assessment": assessment.to_dict()}), 200


@assessments_bp.route("/", methods=["POST"])
@require_teacher
def create_assessment():
    """Create new assessment (teacher only)"""
    data = request.get_json()
    
    required_fields = ["topic", "subject", "questions"]
    for field in required_fields:
        if not data.get(field):
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    assessment = Assessment(
        id=uuid4(),
        topic=data["topic"],
        subject=data["subject"],
        title=data.get("title", f"{data['topic']} Assessment"),
        description=data.get("description"),
        questions=data["questions"],
        difficulty=data.get("difficulty", "medium"),
        time_limit_minutes=data.get("time_limit_minutes", 30),
        is_adaptive=data.get("is_adaptive", False),
        scheduled_date=datetime.fromisoformat(data["scheduled_date"]) if data.get("scheduled_date") else None,
        created_by=request.user_id
    )
    
    db.session.add(assessment)
    db.session.commit()
    
    return jsonify({"assessment": assessment.to_dict()}), 201


@assessments_bp.route("/<assessment_id>/submit", methods=["POST"])
@require_auth
def submit_assessment(assessment_id):
    """Submit assessment answers"""
    assessment = Assessment.query.get(assessment_id)
    
    if not assessment:
        return jsonify({"error": "Assessment not found"}), 404
    
    data = request.get_json()
    answers = data.get("answers")
    time_taken = data.get("time_taken_seconds")
    confidence_score = data.get("confidence_score")
    
    if not answers:
        return jsonify({"error": "Answers required"}), 400
    
    # Calculate score
    correct = 0
    total = len(assessment.questions)
    feedback = []
    
    for i, question in enumerate(assessment.questions):
        user_answer = answers.get(str(i)) or answers.get(i)
        correct_answer = question.get("correct_answer")
        is_correct = user_answer == correct_answer
        
        if is_correct:
            correct += 1
        
        feedback.append({
            "question_index": i,
            "is_correct": is_correct,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "explanation": question.get("explanation", "")
        })
    
    score = (correct / total) * 100 if total > 0 else 0
    
    # Save result
    result = AssessmentResult(
        id=uuid4(),
        user_id=request.user_id,
        assessment_id=assessment_id,
        answers=answers,
        score=score,
        max_score=100.0,
        time_taken_seconds=time_taken,
        confidence_score=confidence_score,
        feedback=feedback
    )
    
    db.session.add(result)
    
    # Update progress
    progress = Progress.query.filter_by(
        user_id=request.user_id,
        topic=assessment.topic,
        subject=assessment.subject
    ).first()
    
    if progress:
        scores = progress.assessment_scores or []
        scores.append({
            "assessment_id": str(assessment_id),
            "score": score,
            "date": datetime.utcnow().isoformat()
        })
        progress.assessment_scores = scores
        
        # Recalculate confidence based on recent scores
        recent_scores = [s["score"] for s in scores[-5:]]
        progress.confidence_score = sum(recent_scores) / len(recent_scores)
        progress.is_weak = progress.confidence_score < 50
    
    db.session.commit()
    
    return jsonify({
        "result": result.to_dict(),
        "score": score,
        "correct": correct,
        "total": total,
        "feedback": feedback
    }), 200


@assessments_bp.route("/results", methods=["GET"])
@require_auth
def get_user_results():
    """Get all assessment results for current user"""
    user_id = request.user_id
    
    results = AssessmentResult.query.filter_by(user_id=user_id).order_by(
        AssessmentResult.completed_at.desc()
    ).limit(50).all()
    
    return jsonify({
        "results": [r.to_dict() for r in results],
        "count": len(results)
    }), 200


@assessments_bp.route("/results/<assessment_id>", methods=["GET"])
@require_auth
def get_assessment_results(assessment_id):
    """Get user's result for a specific assessment"""
    result = AssessmentResult.query.filter_by(
        user_id=request.user_id,
        assessment_id=assessment_id
    ).order_by(AssessmentResult.completed_at.desc()).first()
    
    if not result:
        return jsonify({"error": "Result not found"}), 404
    
    return jsonify({"result": result.to_dict()}), 200
