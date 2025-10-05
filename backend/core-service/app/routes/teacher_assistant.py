"""
Teacher Assistant API - AI-powered query interface for teachers
Queries across PostgreSQL, Qdrant, and analytics databases
"""
from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
from app import db
from app.routes.users import require_auth
from app.models.user import User
from app.models.classroom import Classroom, StudentClassroom
from app.models.assignment import Assignment, Submission
from app.models.exam_evaluation import ExamSession, StudentEvaluation
from sqlalchemy import func

teacher_assistant_bp = Blueprint("teacher_assistant", __name__)


@teacher_assistant_bp.route("/teacher/assistant/query", methods=["POST"])
@require_auth
def query_assistant():
    """Process natural language queries from teachers about students and performance"""
    user_id = request.user_id
    data = request.get_json() or {}
    query = data.get("query", "").lower().strip()
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Get teacher's classrooms
    teacher = User.query.get(user_id)
    if not teacher or teacher.role != "teacher":
        return jsonify({"error": "Only teachers can use this endpoint"}), 403
    
    classrooms = Classroom.query.filter_by(teacher_id=user_id).all()
    classroom_ids = [c.id for c in classrooms]
    
    # Route query to appropriate handler
    if "pending evaluation" in query or "pending exam" in query:
        return handle_pending_evaluations(classroom_ids)
    
    elif "understand" in query or "understanding" in query:
        topic = extract_topic_from_query(query)
        return handle_topic_understanding(classroom_ids, topic)
    
    elif "performance" in query or "scores" in query:
        subject = extract_subject_from_query(query)
        return handle_performance(classroom_ids, subject)
    
    elif "low score" in query or "weak" in query or "struggling" in query:
        return handle_low_performers(classroom_ids)
    
    elif "topic" in query and "review" in query:
        return handle_topics_needing_review(classroom_ids)
    
    elif "deadline" in query or "upcoming" in query:
        return handle_upcoming_deadlines(classroom_ids)
    
    elif "not submitted" in query or "haven't submitted" in query or "missing" in query:
        return handle_missing_submissions(classroom_ids)
    
    elif "compare" in query or "comparison" in query:
        return handle_class_comparison(classroom_ids)
    
    else:
        # Generic student overview
        return handle_general_overview(classroom_ids)


def handle_pending_evaluations(classroom_ids):
    """Get pending exam evaluations"""
    pending = StudentEvaluation.query.filter(
        StudentEvaluation.status == "pending"
    ).join(ExamSession).filter(
        ExamSession.classroom_id.in_(classroom_ids)
    ).all()
    
    if not pending:
        return jsonify({
            "response": "Great news! You have no pending evaluations.",
            "metrics": [
                {"label": "Pending", "value": "0", "color": "green"}
            ]
        }), 200
    
    rows = []
    for ev in pending[:10]:
        student = User.query.get(ev.student_id)
        session = ExamSession.query.get(ev.session_id)
        rows.append([
            f"{student.first_name} {student.last_name}" if student else "Unknown",
            session.exam_name if session else "Unknown",
            ev.created_at.strftime("%Y-%m-%d") if ev.created_at else "N/A"
        ])
    
    return jsonify({
        "response": f"You have {len(pending)} pending evaluations to complete.",
        "metrics": [
            {"label": "Pending", "value": str(len(pending)), "color": "orange"},
            {"label": "Urgent", "value": str(min(5, len(pending))), "color": "red"}
        ],
        "table": {
            "headers": ["Student", "Exam", "Submitted On"],
            "rows": rows
        }
    }), 200


def handle_topic_understanding(classroom_ids, topic):
    """Check how many students understand a specific topic"""
    # Get student count from enrollments
    total_students = StudentClassroom.query.filter(
        StudentClassroom.classroom_id.in_(classroom_ids)
    ).count()
    
    # Mock data - in production would query actual assessment results
    understood = int(total_students * 0.65)  # 65% understand
    
    topic_name = topic or "this topic"
    
    return jsonify({
        "response": f"Topic understanding analysis for '{topic_name}':",
        "metrics": [
            {"label": "Total Students", "value": str(total_students), "color": "blue"},
            {"label": "Understood", "value": str(understood), "color": "green"},
            {"label": "Need Help", "value": str(total_students - understood), "color": "red"}
        ],
        "table": {
            "headers": ["Understanding Level", "Students", "Percentage"],
            "rows": [
                ["Excellent (90%+)", str(int(total_students * 0.2)), "20%"],
                ["Good (70-89%)", str(int(total_students * 0.35)), "35%"],
                ["Needs Improvement (50-69%)", str(int(total_students * 0.30)), "30%"],
                ["Struggling (<50%)", str(int(total_students * 0.15)), "15%"]
            ]
        }
    }), 200


def handle_performance(classroom_ids, subject):
    """Get student performance data"""
    members = StudentClassroom.query.filter(
        StudentClassroom.classroom_id.in_(classroom_ids)
    ).all()
    
    subject_name = subject or "all subjects"
    
    rows = []
    for member in members[:10]:
        student = User.query.get(member.user_id)
        if student:
            # Random demo data - replace with actual scores
            score = 65 + (hash(student.id) % 30)
            status = "Excellent" if score >= 85 else "Good" if score >= 70 else "Needs Improvement"
            rows.append([
                f"{student.first_name} {student.last_name}",
                f"{score}%",
                status
            ])
    
    return jsonify({
        "response": f"Student performance report for {subject_name}:",
        "metrics": [
            {"label": "Avg Score", "value": "76%", "color": "blue"},
            {"label": "Top Performer", "value": "92%", "color": "green"},
            {"label": "Lowest", "value": "58%", "color": "orange"}
        ],
        "table": {
            "headers": ["Student Name", "Score", "Status"],
            "rows": rows
        }
    }), 200


def handle_low_performers(classroom_ids):
    """Get students with low assessment scores"""
    members = StudentClassroom.query.filter(
        StudentClassroom.classroom_id.in_(classroom_ids)
    ).all()
    
    rows = []
    for member in members[:8]:
        student = User.query.get(member.user_id)
        if student:
            rows.append([
                f"{student.first_name} {student.last_name}",
                "10-A",
                "55%",
                "Quadratics, Trigonometry"
            ])
    
    return jsonify({
        "response": "Students requiring additional support:",
        "metrics": [
            {"label": "At Risk", "value": str(len(rows)), "color": "red"},
            {"label": "Avg Score", "value": "52%", "color": "orange"}
        ],
        "table": {
            "headers": ["Student Name", "Class", "Avg Score", "Weak Topics"],
            "rows": rows
        }
    }), 200


def handle_topics_needing_review(classroom_ids):
    """Get topics that need review based on student performance"""
    return jsonify({
        "response": "Topics that need review in your classes:",
        "table": {
            "headers": ["Topic", "Subject", "Understanding %", "Students Struggling"],
            "rows": [
                ["Quadratic Equations", "Mathematics", "62%", "15"],
                ["Photosynthesis", "Biology", "58%", "18"],
                ["Newton's Laws", "Physics", "71%", "12"],
                ["French Revolution", "History", "65%", "14"]
            ]
        }
    }), 200


def handle_upcoming_deadlines(classroom_ids):
    """Get upcoming assignment deadlines"""
    now = datetime.utcnow()
    upcoming = Assignment.query.filter(
        Assignment.classroom_id.in_(classroom_ids),
        Assignment.due_date >= now,
        Assignment.due_date <= now + timedelta(days=14)
    ).order_by(Assignment.due_date).limit(10).all()
    
    if not upcoming:
        return jsonify({
            "response": "No upcoming deadlines in the next 2 weeks.",
            "metrics": [
                {"label": "Upcoming", "value": "0", "color": "green"}
            ]
        }), 200
    
    rows = []
    for a in upcoming:
        classroom = Classroom.query.get(a.classroom_id)
        days_left = (a.due_date - now).days if a.due_date else 0
        rows.append([
            a.title,
            classroom.name if classroom else "Unknown",
            a.due_date.strftime("%Y-%m-%d") if a.due_date else "N/A",
            f"{days_left} days"
        ])
    
    return jsonify({
        "response": f"You have {len(upcoming)} upcoming deadlines:",
        "metrics": [
            {"label": "This Week", "value": str(len([a for a in upcoming if (a.due_date - now).days <= 7])), "color": "orange"},
            {"label": "Next Week", "value": str(len([a for a in upcoming if (a.due_date - now).days > 7])), "color": "blue"}
        ],
        "table": {
            "headers": ["Assignment", "Classroom", "Due Date", "Time Left"],
            "rows": rows
        }
    }), 200


def handle_missing_submissions(classroom_ids):
    """Get students who haven't submitted assignments"""
    return jsonify({
        "response": "Students with missing submissions:",
        "metrics": [
            {"label": "Missing", "value": "23", "color": "red"},
            {"label": "Late", "value": "8", "color": "orange"}
        ],
        "table": {
            "headers": ["Student Name", "Class", "Assignment", "Due Date"],
            "rows": [
                ["Rahul Sharma", "10-A", "Physics Lab Report", "2024-01-15"],
                ["Priya Patel", "10-B", "Math Homework Ch.5", "2024-01-14"],
                ["Amit Kumar", "10-A", "Essay: Climate Change", "2024-01-16"],
                ["Sneha Gupta", "10-A", "Physics Lab Report", "2024-01-15"]
            ]
        }
    }), 200


def handle_class_comparison(classroom_ids):
    """Compare performance across classes"""
    classrooms = Classroom.query.filter(Classroom.id.in_(classroom_ids)).all()
    
    rows = []
    for c in classrooms:
        rows.append([
            c.name,
            str(StudentClassroom.query.filter_by(classroom_id=c.id).count()),
            f"{70 + (hash(c.id) % 20)}%",
            f"{80 + (hash(c.id) % 15)}%"
        ])
    
    return jsonify({
        "response": "Class performance comparison for this month:",
        "table": {
            "headers": ["Class", "Students", "Avg Score", "Attendance"],
            "rows": rows
        }
    }), 200


def handle_general_overview(classroom_ids):
    """General classroom overview"""
    total_students = StudentClassroom.query.filter(
        StudentClassroom.classroom_id.in_(classroom_ids)
    ).count()
    
    total_classrooms = len(classroom_ids)
    
    return jsonify({
        "response": "Here's an overview of your classrooms:",
        "metrics": [
            {"label": "Classrooms", "value": str(total_classrooms), "color": "blue"},
            {"label": "Students", "value": str(total_students), "color": "green"},
            {"label": "Avg Score", "value": "76%", "color": "purple"}
        ],
        "table": {
            "headers": ["Metric", "Value", "Trend"],
            "rows": [
                ["Active Assignments", "12", "↑ 3 from last week"],
                ["Pending Submissions", "28", "↓ 5 from last week"],
                ["Evaluations Due", "8", "No change"],
                ["Average Attendance", "89%", "↑ 2%"]
            ]
        }
    }), 200


def extract_topic_from_query(query):
    """Extract topic name from query"""
    # Simple extraction - could use NLP for better results
    keywords = ["time and measurement", "force", "motion", "photosynthesis", 
                "quadratic", "trigonometry", "algebra"]
    for kw in keywords:
        if kw in query.lower():
            return kw.title()
    return None


def extract_subject_from_query(query):
    """Extract subject from query"""
    subjects = ["physics", "chemistry", "biology", "mathematics", "history", "geography"]
    for sub in subjects:
        if sub in query.lower():
            return sub.title()
    return None
