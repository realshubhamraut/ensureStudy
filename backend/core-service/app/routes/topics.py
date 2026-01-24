"""
Topics API Routes

Endpoints for managing curriculum topic hierarchy:
- Subjects (top-level organizational unit)
- Topics (within subjects)
- Subtopics (within topics)
- Syllabi (linked to classrooms)
- Topic Progress
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
from uuid import uuid4
from app import db
from app.models.curriculum import Subject, Topic, Subtopic, SubtopicAssessment, StudentSubtopicProgress, Syllabus
from app.routes.users import require_auth, require_teacher

topics_bp = Blueprint("topics", __name__, url_prefix="/api/topics")


# ============================================================================
# Subject Endpoints
# ============================================================================

@topics_bp.route("/subjects", methods=["GET"])
@require_auth
def list_subjects():
    """List all subjects optionally filtered by grade/board"""
    grade = request.args.get("grade")
    board = request.args.get("board")
    include_topics = request.args.get("include_topics", "false").lower() == "true"
    
    query = Subject.query.filter_by(is_active=True)
    
    subjects = query.order_by(Subject.order).all()
    
    # Filter by grade/board if provided (stored as JSON arrays)
    result = []
    for s in subjects:
        if grade and s.grade_levels and grade not in s.grade_levels:
            continue
        if board and s.boards and board not in s.boards:
            continue
        result.append(s.to_dict(include_topics=include_topics))
    
    return jsonify({
        "subjects": result,
        "count": len(result)
    }), 200


@topics_bp.route("/subjects/<subject_id>", methods=["GET"])
@require_auth
def get_subject(subject_id):
    """Get a subject with its topics"""
    subject = Subject.query.get(subject_id)
    
    if not subject:
        return jsonify({"error": "Subject not found"}), 404
    
    return jsonify({"subject": subject.to_dict(include_topics=True)}), 200


@topics_bp.route("/subjects", methods=["POST"])
@require_teacher
def create_subject():
    """Create a new subject"""
    data = request.get_json()
    
    if not data.get("name"):
        return jsonify({"error": "Name is required"}), 400
    
    # Generate code if not provided
    code = data.get("code") or data["name"][:3].upper()
    
    # Check for duplicate code
    existing = Subject.query.filter_by(code=code).first()
    if existing:
        return jsonify({"error": f"Subject with code {code} already exists"}), 400
    
    subject = Subject(
        id=str(uuid4()),
        name=data["name"],
        code=code,
        description=data.get("description"),
        icon=data.get("icon"),
        color=data.get("color"),
        grade_levels=data.get("grade_levels", []),
        boards=data.get("boards", []),
        order=data.get("order", 0)
    )
    
    db.session.add(subject)
    db.session.commit()
    
    return jsonify({"subject": subject.to_dict()}), 201


# ============================================================================
# Topic Endpoints
# ============================================================================

@topics_bp.route("/", methods=["GET"])
@require_auth
def list_topics():
    """List topics optionally filtered by subject"""
    subject_id = request.args.get("subject_id")
    include_subtopics = request.args.get("include_subtopics", "false").lower() == "true"
    
    query = Topic.query.filter_by(is_active=True)
    
    if subject_id:
        query = query.filter_by(subject_id=subject_id)
    
    topics = query.order_by(Topic.order).all()
    
    return jsonify({
        "topics": [t.to_dict(include_subtopics=include_subtopics) for t in topics],
        "count": len(topics)
    }), 200


@topics_bp.route("/<topic_id>", methods=["GET"])
@require_auth
def get_topic(topic_id):
    """Get a topic with subtopics"""
    topic = Topic.query.get(topic_id)
    
    if not topic:
        return jsonify({"error": "Topic not found"}), 404
    
    return jsonify({"topic": topic.to_dict(include_subtopics=True)}), 200


@topics_bp.route("/", methods=["POST"])
@require_teacher
def create_topic():
    """Create a new topic under a subject"""
    data = request.get_json()
    
    required = ["subject_id", "name"]
    for field in required:
        if not data.get(field):
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Verify subject exists
    subject = Subject.query.get(data["subject_id"])
    if not subject:
        return jsonify({"error": "Subject not found"}), 404
    
    # Get max order
    max_order = db.session.query(db.func.max(Topic.order)).filter_by(
        subject_id=data["subject_id"]
    ).scalar() or 0
    
    topic = Topic(
        id=str(uuid4()),
        subject_id=data["subject_id"],
        name=data["name"],
        description=data.get("description"),
        estimated_hours=data.get("estimated_hours", 2.0),
        difficulty=data.get("difficulty", "medium"),
        order=data.get("order", max_order + 1)
    )
    
    db.session.add(topic)
    db.session.commit()
    
    return jsonify({"topic": topic.to_dict()}), 201


@topics_bp.route("/<topic_id>", methods=["PUT"])
@require_teacher
def update_topic(topic_id):
    """Update a topic"""
    topic = Topic.query.get(topic_id)
    
    if not topic:
        return jsonify({"error": "Topic not found"}), 404
    
    data = request.get_json()
    
    if "name" in data:
        topic.name = data["name"]
    if "description" in data:
        topic.description = data["description"]
    if "difficulty" in data:
        topic.difficulty = data["difficulty"]
    if "estimated_hours" in data:
        topic.estimated_hours = data["estimated_hours"]
    if "order" in data:
        topic.order = data["order"]
    
    db.session.commit()
    
    return jsonify({"topic": topic.to_dict()}), 200


@topics_bp.route("/<topic_id>", methods=["DELETE"])
@require_teacher
def delete_topic(topic_id):
    """Delete a topic (soft delete)"""
    topic = Topic.query.get(topic_id)
    
    if not topic:
        return jsonify({"error": "Topic not found"}), 404
    
    topic.is_active = False
    db.session.commit()
    
    return jsonify({"message": "Topic deleted"}), 200


# ============================================================================
# Subtopic Endpoints
# ============================================================================

@topics_bp.route("/<topic_id>/subtopics", methods=["GET"])
@require_auth
def list_subtopics(topic_id):
    """List subtopics for a topic"""
    topic = Topic.query.get(topic_id)
    
    if not topic:
        return jsonify({"error": "Topic not found"}), 404
    
    subtopics = Subtopic.query.filter_by(topic_id=topic_id, is_active=True).order_by(Subtopic.order).all()
    
    return jsonify({
        "subtopics": [s.to_dict() for s in subtopics],
        "count": len(subtopics)
    }), 200


@topics_bp.route("/<topic_id>/subtopics", methods=["POST"])
@require_teacher
def create_subtopic(topic_id):
    """Create a subtopic under a topic"""
    topic = Topic.query.get(topic_id)
    
    if not topic:
        return jsonify({"error": "Topic not found"}), 404
    
    data = request.get_json()
    
    if not data.get("name"):
        return jsonify({"error": "Name is required"}), 400
    
    # Get max order
    max_order = db.session.query(db.func.max(Subtopic.order)).filter_by(
        topic_id=topic_id
    ).scalar() or 0
    
    subtopic = Subtopic(
        id=str(uuid4()),
        topic_id=topic_id,
        name=data["name"],
        description=data.get("description"),
        key_concepts=data.get("key_concepts", []),
        learning_objectives=data.get("learning_objectives", []),
        estimated_minutes=data.get("estimated_minutes", 30),
        difficulty=data.get("difficulty", "medium"),
        order=data.get("order", max_order + 1)
    )
    
    db.session.add(subtopic)
    db.session.commit()
    
    return jsonify({"subtopic": subtopic.to_dict()}), 201


# ============================================================================
# Syllabus Endpoints
# ============================================================================

@topics_bp.route("/syllabus", methods=["GET"])
@require_auth
def list_syllabi():
    """List syllabi for a classroom"""
    classroom_id = request.args.get("classroom_id")
    
    query = Syllabus.query
    if classroom_id:
        query = query.filter_by(classroom_id=classroom_id)
    
    syllabi = query.order_by(Syllabus.created_at.desc()).all()
    
    return jsonify({
        "syllabi": [s.to_dict() for s in syllabi],
        "count": len(syllabi)
    }), 200


@topics_bp.route("/syllabus", methods=["POST"])
@require_teacher
def create_syllabus():
    """Create a new syllabus (triggers topic extraction)"""
    data = request.get_json()
    
    required = ["classroom_id", "title"]
    for field in required:
        if not data.get(field):
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    syllabus = Syllabus(
        id=str(uuid4()),
        classroom_id=data["classroom_id"],
        document_id=data.get("document_id"),
        title=data["title"],
        subject_id=data.get("subject_id"),
        academic_year=data.get("academic_year"),
        description=data.get("description"),
        extraction_status="pending",
        created_by=request.user_id
    )
    
    db.session.add(syllabus)
    db.session.commit()
    
    # TODO: Trigger async topic extraction job (Phase 2)
    # extract_topics_from_syllabus.delay(syllabus.id)
    
    return jsonify({"syllabus": syllabus.to_dict()}), 201


@topics_bp.route("/syllabus/<syllabus_id>", methods=["GET"])
@require_auth
def get_syllabus(syllabus_id):
    """Get syllabus details"""
    syllabus = Syllabus.query.get(syllabus_id)
    
    if not syllabus:
        return jsonify({"error": "Syllabus not found"}), 404
    
    return jsonify({"syllabus": syllabus.to_dict()}), 200


@topics_bp.route("/syllabus/<syllabus_id>", methods=["PUT"])
def update_syllabus(syllabus_id):
    """Update syllabus (link to subject, update status, etc.)"""
    syllabus = Syllabus.query.get(syllabus_id)
    
    if not syllabus:
        return jsonify({"error": "Syllabus not found"}), 404
    
    data = request.get_json() or {}
    
    # Update allowed fields
    if "subject_id" in data:
        syllabus.subject_id = data["subject_id"]
    if "extraction_status" in data:
        syllabus.extraction_status = data["extraction_status"]
    if "topics_count" in data:
        syllabus.topics_count = data["topics_count"]
    if "description" in data:
        syllabus.description = data["description"]
    
    db.session.commit()
    
    return jsonify({"syllabus": syllabus.to_dict()}), 200


# ============================================================================
# Topics by Classroom Endpoints (for Mock Interview UI)
# ============================================================================

@topics_bp.route("/by-classrooms", methods=["POST"])
@require_auth
def get_topics_by_classrooms():
    """
    Get topics for selected classrooms with student's confidence scores.
    
    Request body:
    {
        "classroom_ids": ["class-1", "class-2"] or ["all"]
    }
    
    Returns topics with confidence scores calculated from StudentSubtopicProgress.
    """
    user_id = request.user_id
    data = request.get_json() or {}
    classroom_ids = data.get("classroom_ids", [])
    sort_by_confidence = data.get("sort_by_confidence", "asc")  # asc = weak topics first
    
    from app.models.classroom import Classroom, StudentClassroom
    
    # Get user's enrolled classrooms
    if not classroom_ids or "all" in classroom_ids:
        # Get all classrooms the student is enrolled in
        enrollments = StudentClassroom.query.filter_by(student_id=user_id).all()
        classroom_ids = [e.classroom_id for e in enrollments]
    
    if not classroom_ids:
        return jsonify({
            "topics": [],
            "count": 0,
            "message": "No classrooms found"
        }), 200
    
    # Get syllabi for these classrooms
    syllabi = Syllabus.query.filter(Syllabus.classroom_id.in_(classroom_ids)).all()
    subject_ids = [s.subject_id for s in syllabi if s.subject_id]
    
    # Get topics for these subjects
    if subject_ids:
        topics_query = Topic.query.filter(
            Topic.subject_id.in_(subject_ids),
            Topic.is_active == True
        )
    else:
        # Fallback: get all active topics if no syllabus-subject link
        topics_query = Topic.query.filter_by(is_active=True)
    
    topics = topics_query.order_by(Topic.order).all()
    
    # Calculate confidence scores for each topic
    result = []
    for topic in topics:
        # Get all subtopics for this topic
        subtopics = Subtopic.query.filter_by(topic_id=topic.id, is_active=True).all()
        subtopic_ids = [s.id for s in subtopics]
        
        # Get user's progress for these subtopics
        progress_records = StudentSubtopicProgress.query.filter(
            StudentSubtopicProgress.user_id == user_id,
            StudentSubtopicProgress.subtopic_id.in_(subtopic_ids)
        ).all() if subtopic_ids else []
        
        # Calculate confidence score (average of subtopic mastery levels)
        if progress_records:
            total_mastery = sum(p.mastery_level or 0 for p in progress_records)
            confidence_score = round(total_mastery / len(subtopics), 1) if subtopics else 0
            subtopics_mastered = sum(1 for p in progress_records if (p.mastery_level or 0) >= 80)
            subtopics_attempted = len(progress_records)
        else:
            confidence_score = 0
            subtopics_mastered = 0
            subtopics_attempted = 0
        
        # Get subject info
        subject = Subject.query.get(topic.subject_id) if topic.subject_id else None
        
        topic_data = {
            "id": str(topic.id),
            "name": topic.name,
            "description": topic.description,
            "difficulty": topic.difficulty,
            "estimated_hours": topic.estimated_hours,
            "subject_id": str(topic.subject_id) if topic.subject_id else None,
            "subject_name": subject.name if subject else None,
            "subject_icon": subject.icon if subject else None,
            "confidence_score": confidence_score,
            "subtopics_count": len(subtopics),
            "subtopics_mastered": subtopics_mastered,
            "subtopics_attempted": subtopics_attempted,
            "is_weak": confidence_score < 50
        }
        result.append(topic_data)
    
    # Sort by confidence (weak topics first by default)
    if sort_by_confidence == "asc":
        result.sort(key=lambda x: x["confidence_score"])
    elif sort_by_confidence == "desc":
        result.sort(key=lambda x: x["confidence_score"], reverse=True)
    
    return jsonify({
        "topics": result,
        "count": len(result),
        "classroom_ids": classroom_ids
    }), 200


@topics_bp.route("/for-classroom/<classroom_id>", methods=["GET"])
@require_auth
def get_topics_for_classroom(classroom_id):
    """
    Get topics for a single classroom with student's confidence scores.
    Convenience endpoint that wraps by-classrooms.
    """
    user_id = request.user_id
    
    # Get syllabus for this classroom
    syllabus = Syllabus.query.filter_by(classroom_id=classroom_id).first()
    
    if not syllabus or not syllabus.subject_id:
        # Fallback: return all topics
        topics = Topic.query.filter_by(is_active=True).order_by(Topic.order).all()
    else:
        topics = Topic.query.filter_by(
            subject_id=syllabus.subject_id,
            is_active=True
        ).order_by(Topic.order).all()
    
    result = []
    for topic in topics:
        subtopics = Subtopic.query.filter_by(topic_id=topic.id, is_active=True).all()
        subtopic_ids = [s.id for s in subtopics]
        
        progress_records = StudentSubtopicProgress.query.filter(
            StudentSubtopicProgress.user_id == user_id,
            StudentSubtopicProgress.subtopic_id.in_(subtopic_ids)
        ).all() if subtopic_ids else []
        
        if progress_records:
            total_mastery = sum(p.mastery_level or 0 for p in progress_records)
            confidence_score = round(total_mastery / len(subtopics), 1) if subtopics else 0
        else:
            confidence_score = 0
        
        subject = Subject.query.get(topic.subject_id) if topic.subject_id else None
        
        result.append({
            "id": str(topic.id),
            "name": topic.name,
            "description": topic.description,
            "subject_name": subject.name if subject else None,
            "confidence_score": confidence_score,
            "subtopics_count": len(subtopics),
            "is_weak": confidence_score < 50
        })
    
    # Sort weak topics first
    result.sort(key=lambda x: x["confidence_score"])
    
    return jsonify({
        "topics": result,
        "count": len(result),
        "classroom_id": classroom_id
    }), 200


# ============================================================================
# Progress Endpoints
# ============================================================================

@topics_bp.route("/progress", methods=["GET"])
@require_auth
def get_user_progress():
    """Get current user's progress across all subtopics"""
    user_id = request.user_id
    subject_id = request.args.get("subject_id")
    topic_id = request.args.get("topic_id")
    
    query = StudentSubtopicProgress.query.filter_by(user_id=user_id)
    
    progress_records = query.all()
    
    # Filter by topic/subject if needed
    result = []
    for p in progress_records:
        subtopic = Subtopic.query.get(p.subtopic_id)
        if not subtopic:
            continue
        
        topic = Topic.query.get(subtopic.topic_id)
        if not topic:
            continue
        
        # Apply filters
        if topic_id and str(topic.id) != topic_id:
            continue
        if subject_id and str(topic.subject_id) != subject_id:
            continue
        
        progress_data = p.to_dict()
        progress_data["subtopic_name"] = subtopic.name
        progress_data["topic_name"] = topic.name
        progress_data["topic_id"] = str(topic.id)
        result.append(progress_data)
    
    return jsonify({
        "progress": result,
        "count": len(result)
    }), 200


@topics_bp.route("/progress/<subtopic_id>", methods=["GET"])
@require_auth
def get_subtopic_progress(subtopic_id):
    """Get current user's progress for a specific subtopic"""
    user_id = request.user_id
    
    progress = StudentSubtopicProgress.query.filter_by(
        user_id=user_id,
        subtopic_id=subtopic_id
    ).first()
    
    if not progress:
        return jsonify({
            "progress": {
                "subtopic_id": subtopic_id,
                "status": "not_started",
                "attempts": 0,
                "best_score": 0,
                "mastery_level": 0
            }
        }), 200
    
    return jsonify({"progress": progress.to_dict()}), 200


@topics_bp.route("/progress/<subtopic_id>/update", methods=["POST"])
@require_auth
def update_subtopic_progress(subtopic_id):
    """Update progress for a subtopic after assessment"""
    user_id = request.user_id
    data = request.get_json()
    
    score = data.get("score")
    time_spent = data.get("time_spent_minutes", 0)
    
    if score is None:
        return jsonify({"error": "Score is required"}), 400
    
    # Get or create progress record
    progress = StudentSubtopicProgress.query.filter_by(
        user_id=user_id,
        subtopic_id=subtopic_id
    ).first()
    
    if not progress:
        progress = StudentSubtopicProgress(
            id=str(uuid4()),
            user_id=user_id,
            subtopic_id=subtopic_id,
            first_attempt_at=datetime.utcnow()
        )
        db.session.add(progress)
    
    # Update stats
    progress.attempts += 1
    progress.last_score = score
    progress.last_attempt_at = datetime.utcnow()
    progress.total_time_minutes += time_spent
    
    if score > progress.best_score:
        progress.best_score = score
    
    # Calculate average
    if progress.attempts == 1:
        progress.average_score = score
    else:
        progress.average_score = ((progress.average_score * (progress.attempts - 1)) + score) / progress.attempts
    
    # Update status based on score
    if progress.best_score >= 80:
        progress.status = "mastered"
        progress.completed_at = progress.completed_at or datetime.utcnow()
    elif progress.best_score >= 60:
        progress.status = "completed"
        progress.completed_at = progress.completed_at or datetime.utcnow()
    elif progress.attempts > 0:
        progress.status = "in_progress"
    
    # Mastery level (0-100) based on average score
    progress.mastery_level = min(100, progress.average_score)
    
    db.session.commit()
    
    return jsonify({"progress": progress.to_dict()}), 200


@topics_bp.route("/weak-subtopics", methods=["GET"])
@require_auth
def get_weak_subtopics():
    """Get subtopics where the user is struggling (score < 60)"""
    user_id = request.user_id
    
    weak_progress = StudentSubtopicProgress.query.filter(
        StudentSubtopicProgress.user_id == user_id,
        StudentSubtopicProgress.best_score < 60,
        StudentSubtopicProgress.attempts > 0
    ).all()
    
    result = []
    for p in weak_progress:
        subtopic = Subtopic.query.get(p.subtopic_id)
        if not subtopic:
            continue
        
        topic = Topic.query.get(subtopic.topic_id)
        subject = Subject.query.get(topic.subject_id) if topic else None
        
        result.append({
            "progress": p.to_dict(),
            "subtopic": subtopic.to_dict(),
            "topic_name": topic.name if topic else None,
            "subject_name": subject.name if subject else None
        })
    
    return jsonify({
        "weak_subtopics": result,
        "count": len(result)
    }), 200


# ============================================================================
# Question Bank Endpoints
# ============================================================================

@topics_bp.route("/question-banks", methods=["GET"])
@require_auth
def list_question_banks():
    """List question banks for a classroom"""
    from app.models.curriculum import QuestionBank
    
    classroom_id = request.args.get("classroom_id")
    
    query = QuestionBank.query
    if classroom_id:
        query = query.filter_by(classroom_id=classroom_id)
    
    banks = query.order_by(QuestionBank.created_at.desc()).all()
    
    return jsonify({
        "question_banks": [b.to_dict() for b in banks],
        "count": len(banks)
    }), 200


@topics_bp.route("/question-banks", methods=["POST"])
@require_teacher
def create_question_bank():
    """Create a new question bank"""
    from app.models.curriculum import QuestionBank
    
    data = request.get_json()
    
    if not data.get("classroom_id") or not data.get("name"):
        return jsonify({"error": "classroom_id and name are required"}), 400
    
    bank = QuestionBank(
        id=str(uuid4()),
        classroom_id=data["classroom_id"],
        subject_id=data.get("subject_id"),
        name=data["name"],
        description=data.get("description"),
        source_type=data.get("source_type", "generated"),
        source_document_id=data.get("source_document_id"),
        created_by=data.get("created_by") or request.user_id
    )
    
    db.session.add(bank)
    db.session.commit()
    
    return jsonify(bank.to_dict()), 201


@topics_bp.route("/question-banks/<bank_id>", methods=["GET"])
@require_auth
def get_question_bank(bank_id):
    """Get a question bank with its questions"""
    from app.models.curriculum import QuestionBank, Question
    
    bank = QuestionBank.query.get(bank_id)
    
    if not bank:
        return jsonify({"error": "Question bank not found"}), 404
    
    questions = Question.query.filter_by(question_bank_id=bank_id, is_active=True).all()
    
    bank_data = bank.to_dict()
    bank_data["questions"] = [q.to_dict(include_answer=True) for q in questions]
    
    return jsonify(bank_data), 200


# ============================================================================
# Question Endpoints
# ============================================================================

@topics_bp.route("/questions", methods=["GET"])
@require_auth
def list_questions():
    """List questions with optional filters"""
    from app.models.curriculum import Question
    
    question_bank_id = request.args.get("question_bank_id")
    topic_id = request.args.get("topic_id")
    subtopic_id = request.args.get("subtopic_id")
    question_type = request.args.get("question_type")
    difficulty = request.args.get("difficulty")
    include_answers = request.args.get("include_answers", "false").lower() == "true"
    limit = int(request.args.get("limit", 50))
    
    query = Question.query.filter_by(is_active=True)
    
    if question_bank_id:
        query = query.filter_by(question_bank_id=question_bank_id)
    if topic_id:
        query = query.filter_by(topic_id=topic_id)
    if subtopic_id:
        query = query.filter_by(subtopic_id=subtopic_id)
    if question_type:
        query = query.filter_by(question_type=question_type)
    if difficulty:
        query = query.filter_by(difficulty=difficulty)
    
    questions = query.limit(limit).all()
    
    return jsonify({
        "questions": [q.to_dict(include_answer=include_answers) for q in questions],
        "count": len(questions)
    }), 200


@topics_bp.route("/questions", methods=["POST"])
@require_teacher
def create_question():
    """Create a new question"""
    from app.models.curriculum import Question
    
    data = request.get_json()
    
    if not data.get("question_type") or not data.get("question_text"):
        return jsonify({"error": "question_type and question_text are required"}), 400
    
    question = Question(
        id=str(uuid4()),
        question_bank_id=data.get("question_bank_id"),
        topic_id=data.get("topic_id"),
        subtopic_id=data.get("subtopic_id"),
        question_type=data["question_type"],
        question_text=data["question_text"],
        options=data.get("options", []),
        correct_answer=data.get("correct_answer"),
        explanation=data.get("explanation"),
        key_points=data.get("key_points", []),
        difficulty=data.get("difficulty", "medium"),
        marks=data.get("marks", 1),
        time_estimate_seconds=data.get("time_estimate_seconds", 60),
        source_chunk_id=data.get("source_chunk_id"),
        source_content_preview=data.get("source_content_preview"),
        created_by=data.get("created_by") or request.user_id
    )
    
    db.session.add(question)
    
    # Update question bank count if linked
    if question.question_bank_id:
        from app.models.curriculum import QuestionBank
        bank = QuestionBank.query.get(question.question_bank_id)
        if bank:
            bank.total_questions = (bank.total_questions or 0) + 1
    
    db.session.commit()
    
    return jsonify(question.to_dict(include_answer=True)), 201


@topics_bp.route("/questions/<question_id>", methods=["GET"])
@require_auth
def get_question(question_id):
    """Get a single question"""
    from app.models.curriculum import Question
    
    question = Question.query.get(question_id)
    
    if not question:
        return jsonify({"error": "Question not found"}), 404
    
    # Teachers see answers, students don't
    include_answer = hasattr(request, "user_role") and request.user_role == "teacher"
    
    return jsonify({"question": question.to_dict(include_answer=include_answer)}), 200


@topics_bp.route("/questions/<question_id>", methods=["PUT"])
@require_teacher
def update_question(question_id):
    """Update a question"""
    from app.models.curriculum import Question
    
    question = Question.query.get(question_id)
    
    if not question:
        return jsonify({"error": "Question not found"}), 404
    
    data = request.get_json()
    
    # Update fields
    updatable = [
        "question_text", "options", "correct_answer", "explanation",
        "key_points", "difficulty", "marks", "time_estimate_seconds",
        "is_active", "review_status"
    ]
    
    for field in updatable:
        if field in data:
            setattr(question, field, data[field])
    
    db.session.commit()
    
    return jsonify({"question": question.to_dict(include_answer=True)}), 200


@topics_bp.route("/questions/<question_id>", methods=["DELETE"])
@require_teacher
def delete_question(question_id):
    """Delete a question (soft delete)"""
    from app.models.curriculum import Question
    
    question = Question.query.get(question_id)
    
    if not question:
        return jsonify({"error": "Question not found"}), 404
    
    question.is_active = False
    db.session.commit()
    
    return jsonify({"message": "Question deleted"}), 200


@topics_bp.route("/questions/<question_id>/analytics", methods=["POST"])
@require_auth
def update_question_analytics(question_id):
    """Update question analytics after student answers"""
    from app.models.curriculum import Question
    
    question = Question.query.get(question_id)
    
    if not question:
        return jsonify({"error": "Question not found"}), 404
    
    data = request.get_json()
    was_correct = data.get("was_correct", False)
    time_taken_seconds = data.get("time_taken_seconds", 0)
    
    question.update_analytics(was_correct, time_taken_seconds)
    db.session.commit()
    
    return jsonify({
        "times_used": question.times_used,
        "times_correct": question.times_correct,
        "times_incorrect": question.times_incorrect,
        "difficulty_rating": question.difficulty_rating
    }), 200


@topics_bp.route("/questions/random", methods=["GET"])
@require_auth
def get_random_questions():
    """Get random questions for a quiz"""
    from app.models.curriculum import Question
    from sqlalchemy.sql.expression import func
    
    topic_id = request.args.get("topic_id")
    subtopic_id = request.args.get("subtopic_id")
    question_type = request.args.get("question_type")
    difficulty = request.args.get("difficulty")
    count = int(request.args.get("count", 10))
    
    query = Question.query.filter_by(is_active=True, review_status="approved")
    
    if topic_id:
        query = query.filter_by(topic_id=topic_id)
    if subtopic_id:
        query = query.filter_by(subtopic_id=subtopic_id)
    if question_type:
        query = query.filter_by(question_type=question_type)
    if difficulty:
        query = query.filter_by(difficulty=difficulty)
    
    # Random selection
    questions = query.order_by(func.random()).limit(count).all()
    
    return jsonify({
        "questions": [q.to_dict(include_answer=False) for q in questions],
        "count": len(questions)
    }), 200
