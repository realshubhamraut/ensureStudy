"""
Evaluation Routes - Exam session and student evaluation management
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
from app import db
from app.models.user import User
from app.models.exam_evaluation import ExamSession, StudentEvaluation
from app.utils.jwt_handler import verify_token

evaluation_bp = Blueprint("evaluation", __name__, url_prefix="/api/evaluation")


def teacher_required(f):
    """Decorator to require teacher role"""
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
            
            if not user or user.role != "teacher":
                return jsonify({"error": "Teacher access required"}), 403
            
            request.current_user = user
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({"error": f"Authentication failed: {str(e)}"}), 401
    
    return decorated


# ====================  Exam Sessions ====================

@evaluation_bp.route("/exam-session", methods=["POST"])
@teacher_required
def create_exam_session():
    """Create a new exam session"""
    user = request.current_user
    data = request.get_json()
    
    if not user.organization_id:
        return jsonify({"error": "Not part of an organization"}), 400
    
    required = ["name", "exam_type", "subject", "class_name", "date"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    try:
        exam_date = datetime.strptime(data["date"], "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    
    session = ExamSession(
        name=data["name"],
        exam_type=data["exam_type"],
        subject=data["subject"],
        class_name=data["class_name"],
        date=exam_date,
        teacher_id=user.id,
        organization_id=user.organization_id,
        total_marks=data.get("total_marks", 0),
        question_paper=data.get("question_paper", {}),
        status="in_progress"
    )
    
    db.session.add(session)
    db.session.commit()
    
    return jsonify({
        "message": "Exam session created successfully",
        "exam_session": session.to_dict()
    }), 201


@evaluation_bp.route("/exam-sessions", methods=["GET"])
@teacher_required
def list_exam_sessions():
    """List all exam sessions for the teacher"""
    user = request.current_user
    
    sessions = ExamSession.query.filter_by(
        teacher_id=user.id
    ).order_by(ExamSession.created_at.desc()).all()
    
    return jsonify({
        "exam_sessions": [s.to_dict() for s in sessions],
        "count": len(sessions)
    }), 200


@evaluation_bp.route("/exam-session/<session_id>", methods=["GET"])
@teacher_required
def get_exam_session(session_id):
    """Get exam session details with evaluations"""
    user = request.current_user
    
    session = ExamSession.query.filter_by(
        id=session_id,
        teacher_id=user.id
    ).first()
    
    if not session:
        return jsonify({"error": "Exam session not found"}), 404
    
    # Get all evaluations for this session
    evaluations = StudentEvaluation.query.filter_by(
        exam_session_id=session_id
    ).all()
    
    return jsonify({
        "exam_session": session.to_dict(),
        "evaluations": [e.to_dict(include_student=True) for e in evaluations]
    }), 200


@evaluation_bp.route("/exam-session/<session_id>", methods=["PUT"])
@teacher_required
def update_exam_session(session_id):
    """Update exam session"""
    user = request.current_user
    data = request.get_json()
    
    session = ExamSession.query.filter_by(
        id=session_id,
        teacher_id=user.id
    ).first()
    
    if not session:
        return jsonify({"error": "Exam session not found"}), 404
    
    # Update allowed fields
    for field in ["name", "total_marks", "question_paper", "status"]:
        if field in data:
            setattr(session, field, data[field])
    
    db.session.commit()
    
    return jsonify({
        "message": "Exam session updated",
        "exam_session": session.to_dict()
    }), 200


@evaluation_bp.route("/exam-session/<session_id>", methods=["DELETE"])
@teacher_required
def delete_exam_session(session_id):
    """Delete exam session and all its evaluations"""
    user = request.current_user
    
    session = ExamSession.query.filter_by(
        id=session_id,
        teacher_id=user.id
    ).first()
    
    if not session:
        return jsonify({"error": "Exam session not found"}), 404
    
    db.session.delete(session)
    db.session.commit()
    
    return jsonify({"message": "Exam session deleted"}), 200


# ====================  Student Evaluations ====================

@evaluation_bp.route("/student-evaluation", methods=["POST"])
@teacher_required
def create_or_update_student_evaluation():
    """Create or update student evaluation for an exam"""
    user = request.current_user
    data = request.get_json()
    
    required = ["exam_session_id", "student_id"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Verify exam session belongs to teacher
    session = ExamSession.query.filter_by(
        id=data["exam_session_id"],
        teacher_id=user.id
    ).first()
    
    if not session:
        return jsonify({"error": "Exam session not found"}), 404
    
    # Check if evaluation already exists
    evaluation = StudentEvaluation.query.filter_by(
        exam_session_id=data["exam_session_id"],
        student_id=data["student_id"]
    ).first()
    
    if evaluation:
        # Update existing
        if "question_evaluations" in data:
            evaluation.question_evaluations = data["question_evaluations"]
        if "total_score" in data:
            evaluation.total_score = data["total_score"]
        if "max_score" in data:
            evaluation.max_score = data["max_score"]
            if evaluation.max_score > 0:
                evaluation.percentage = (evaluation.total_score / evaluation.max_score) * 100
                evaluation.calculate_grade()
        if "teacher_comments" in data:
            evaluation.teacher_comments = data["teacher_comments"]
        if "status" in data:
            evaluation.status = data["status"]
    else:
        # Create new
        total_score = data.get("total_score", 0)
        max_score = data.get("max_score", 0)
        percentage = (total_score / max_score * 100) if max_score > 0 else 0
        
        evaluation = StudentEvaluation(
            exam_session_id=data["exam_session_id"],
            student_id=data["student_id"],
            question_evaluations=data.get("question_evaluations", []),
            total_score=total_score,
            max_score=max_score,
            percentage=percentage,
            status=data.get("status", "pending"),
            teacher_comments=data.get("teacher_comments")
        )
        evaluation.calculate_grade()
        db.session.add(evaluation)
    
    # Update exam session status
    if session.status == "in_progress":
        session.status = "evaluating"
    
    db.session.commit()
    
    return jsonify({
        "message": "Evaluation saved",
        "evaluation": evaluation.to_dict(include_student=True)
    }), 200


@evaluation_bp.route("/exam/<session_id>/evaluations", methods=["GET"])
@teacher_required
def get_exam_evaluations(session_id):
    """Get all evaluations for an exam session"""
    user = request.current_user
    
    session = ExamSession.query.filter_by(
        id=session_id,
        teacher_id=user.id
    ).first()
    
    if not session:
        return jsonify({"error": "Exam session not found"}), 404
    
    evaluations = StudentEvaluation.query.filter_by(
        exam_session_id=session_id
    ).all()
    
    return jsonify({
        "exam_session": session.to_dict(),
        "evaluations": [e.to_dict(include_student=True) for e in evaluations],
        "count": len(evaluations)
    }), 200


@evaluation_bp.route("/student-evaluation/<evaluation_id>", methods=["GET"])
@teacher_required
def get_student_evaluation(evaluation_id):
    """Get specific student evaluation"""
    evaluation = StudentEvaluation.query.get(evaluation_id)
    
    if not evaluation:
        return jsonify({"error": "Evaluation not found"}), 404
    
    # Verify teacher owns the exam session
    session = ExamSession.query.filter_by(
        id=evaluation.exam_session_id,
        teacher_id=request.current_user.id
    ).first()
    
    if not session:
        return jsonify({"error": "Access denied"}), 403
    
    return jsonify({
        "evaluation": evaluation.to_dict(include_student=True),
        "exam_session": session.to_dict()
    }), 200


# ====================  Results Declaration ====================

@evaluation_bp.route("/exam/<session_id>/declare-results", methods=["POST"])
@teacher_required
def declare_results(session_id):
    """Declare results for an exam session"""
    user = request.current_user
    
    session = ExamSession.query.filter_by(
        id=session_id,
        teacher_id=user.id
    ).first()
    
    if not session:
        return jsonify({"error": "Exam session not found"}), 404
    
    if session.status == "results_declared":
        return jsonify({"error": "Results already declared"}), 400
    
    # Check if there are any evaluations
    evaluation_count = StudentEvaluation.query.filter_by(
        exam_session_id=session_id,
        status="evaluated"
    ).count()
    
    if evaluation_count == 0:
        return jsonify({"error": "No evaluated students. Complete evaluations first."}), 400
    
    session.status = "results_declared"
    session.results_declared_at = datetime.utcnow()
    
    db.session.commit()
    
    return jsonify({
        "message": "Results declared successfully",
        "exam_session": session.to_dict()
    }), 200


@evaluation_bp.route("/exam/<session_id>/rollback-results", methods=["POST"])
@teacher_required
def rollback_results(session_id):
    """Rollback declared results to allow editing marks"""
    user = request.current_user
    
    session = ExamSession.query.filter_by(
        id=session_id,
        teacher_id=user.id
    ).first()
    
    if not session:
        return jsonify({"error": "Exam session not found"}), 404
    
    if session.status != "results_declared":
        return jsonify({"error": "Results are not declared yet"}), 400
    
    # Rollback to evaluating status
    session.status = "evaluating"
    session.results_declared_at = None
    
    db.session.commit()
    
    return jsonify({
        "message": "Results rolled back successfully. You can now edit marks.",
        "exam_session": session.to_dict()
    }), 200


@evaluation_bp.route("/results", methods=["GET"])
@teacher_required
def get_declared_results():
    """Get all declared results for the teacher"""
    user = request.current_user
    
    sessions = ExamSession.query.filter_by(
        teacher_id=user.id,
        status="results_declared"
    ).order_by(ExamSession.results_declared_at.desc()).all()
    
    results = []
    for session in sessions:
        evaluations = StudentEvaluation.query.filter_by(
            exam_session_id=session.id
        ).all()
        
        results.append({
            "exam_session": session.to_dict(),
            "evaluations": [e.to_dict(include_student=True) for e in evaluations]
        })
    
    return jsonify({
        "results": results,
        "count": len(results)
    }), 200


# ====================  Students for Evaluation ====================

@evaluation_bp.route("/students", methods=["GET"])
@teacher_required
def get_students_for_evaluation():
    """Get list of students available for evaluation"""
    user = request.current_user
    
    if not user.organization_id:
        return jsonify({"error": "Not part of an organization"}), 400
    
    students = User.query.filter_by(
        organization_id=user.organization_id,
        role="student",
        is_active=True
    ).all()
    
    return jsonify({
        "students": [{
            "id": s.id,
            "name": f"{s.first_name or ''} {s.last_name or ''}".strip() or s.username,
            "email": s.email,
            "username": s.username,
            "avatar_url": s.avatar_url
        } for s in students],
        "count": len(students)
    }), 200


# ====================  Student Results (For Students) ====================

def student_required(f):
    """Decorator to require student role"""
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


@evaluation_bp.route("/my-results", methods=["GET"])
@student_required
def get_my_results():
    """Get declared exam results for the logged-in student"""
    user = request.current_user
    
    # Get all evaluations for this student where results are declared
    evaluations = db.session.query(StudentEvaluation).join(
        ExamSession, StudentEvaluation.exam_session_id == ExamSession.id
    ).filter(
        StudentEvaluation.student_id == user.id,
        ExamSession.status == "results_declared"
    ).order_by(ExamSession.results_declared_at.desc()).all()
    
    results = []
    for evaluation in evaluations:
        exam_session = ExamSession.query.get(evaluation.exam_session_id)
        results.append({
            "id": evaluation.id,
            "exam_session": {
                "id": exam_session.id,
                "name": exam_session.name,
                "exam_type": exam_session.exam_type,
                "subject": exam_session.subject,
                "class_name": exam_session.class_name,
                "date": exam_session.date.isoformat() if exam_session.date else None,
                "results_declared_at": exam_session.results_declared_at.isoformat() if exam_session.results_declared_at else None
            },
            "total_score": evaluation.total_score,
            "max_score": evaluation.max_score,
            "percentage": evaluation.percentage,
            "grade": evaluation.grade,
            "question_evaluations": evaluation.question_evaluations or []
        })
    
    return jsonify({
        "results": results,
        "count": len(results)
    }), 200
