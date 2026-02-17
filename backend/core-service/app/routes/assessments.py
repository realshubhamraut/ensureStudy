"""
Assessment Routes
Enhanced with assessment creation, challenges, and AI integration
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
from uuid import uuid4
from app import db
from app.models.user import Assessment, AssessmentResult, Progress, AssessmentChallenge, User
from app.routes.users import require_auth, require_teacher

assessments_bp = Blueprint("assessments", __name__, url_prefix="/api/assessments")


# ============================================================================
# Assessment CRUD Endpoints
# ============================================================================

@assessments_bp.route("/", methods=["GET"])
@require_auth
def list_assessments():
    """
    List available assessments with filtering.
    Query params:
    - subject: Filter by subject
    - difficulty: Filter by difficulty
    - assessment_type: Filter by type (teacher_created, self_practice, student_challenge)
    - classroom_id: Filter by classroom
    - created_by_me: If 'true', show only assessments created by current user
    """
    subject = request.args.get("subject")
    difficulty = request.args.get("difficulty")
    assessment_type = request.args.get("assessment_type")
    classroom_id = request.args.get("classroom_id")
    created_by_me = request.args.get("created_by_me") == "true"
    
    query = Assessment.query
    
    if subject:
        query = query.filter_by(subject=subject)
    if difficulty:
        query = query.filter_by(difficulty=difficulty)
    if assessment_type:
        query = query.filter_by(assessment_type=assessment_type)
    if classroom_id:
        query = query.filter_by(classroom_id=classroom_id)
    if created_by_me:
        query = query.filter_by(created_by=request.user_id)
    
    assessments = query.order_by(Assessment.created_at.desc()).limit(100).all()
    
    return jsonify({
        "assessments": [a.to_dict(include_questions=False) for a in assessments],
        "count": len(assessments)
    }), 200


@assessments_bp.route("/<assessment_id>", methods=["GET"])
@require_auth
def get_assessment(assessment_id):
    """Get assessment details with questions"""
    assessment = Assessment.query.get(assessment_id)
    
    if not assessment:
        return jsonify({"error": "Assessment not found"}), 404
    
    return jsonify({"assessment": assessment.to_dict(include_questions=True)}), 200


@assessments_bp.route("/<assessment_id>/start", methods=["GET"])
@require_auth
def start_assessment(assessment_id):
    """
    Start taking an assessment.
    Returns assessment with questions but WITHOUT correct answers visible.
    """
    assessment = Assessment.query.get(assessment_id)
    
    if not assessment:
        return jsonify({"error": "Assessment not found"}), 404
    
    # Build questions without revealing correct answers
    questions_for_taking = []
    for i, q in enumerate(assessment.questions or []):
        question_data = {
            "id": q.get("id", str(i)),
            "question_type": q.get("question_type", "mcq"),
            "question_text": q.get("question_text", ""),
            "marks": q.get("marks", 1),
            "difficulty": q.get("difficulty", "medium"),
        }
        
        # For MCQ: include options WITHOUT is_correct flag or explanation
        if q.get("question_type") == "mcq" and q.get("options"):
            raw_options = q.get("options", [])
            formatted_options = []
            for idx, opt in enumerate(raw_options):
                # Handle string options (from AI generation)
                if isinstance(opt, str):
                    letter = chr(65 + idx)  # A, B, C, D...
                    formatted_options.append({"id": letter, "text": opt})
                # Handle dict options (traditional format)
                elif isinstance(opt, dict):
                    formatted_options.append({
                        "id": opt.get("id", chr(65 + idx)), 
                        "text": opt.get("text", str(opt))
                    })
            question_data["options"] = formatted_options
        
        questions_for_taking.append(question_data)
    
    # Calculate total marks
    total_marks = sum(q.get("marks", 1) for q in assessment.questions or [])
    
    return jsonify({
        "assessment": {
            "id": assessment.id,
            "title": assessment.title,
            "topic": assessment.topic,
            "subject": assessment.subject,
            "time_limit_minutes": assessment.time_limit_minutes or 30,
            "questions": questions_for_taking,
            "total_marks": total_marks
        }
    }), 200


@assessments_bp.route("/", methods=["POST"])
@require_auth
def create_assessment():
    """
    Create new assessment.
    Accepts new fields for enhanced assessment creation:
    - assessment_type: teacher_created | self_practice | student_challenge
    - classroom_id: Optional classroom link
    - use_ai_questions: Boolean flag for AI generation
    - source_topics: List of topic IDs to pull questions from
    - source_chapters: List of chapter IDs
    - include_weak_topics: Boolean to include user's weak topics
    """
    data = request.get_json()
    user = User.query.get(request.user_id)
    
    # For teacher_created, require teacher role
    assessment_type = data.get("assessment_type", "teacher_created")
    if assessment_type == "teacher_created" and user.role != "teacher":
        return jsonify({"error": "Only teachers can create teacher assessments"}), 403
    
    # Validate required fields
    if not data.get("questions") and not data.get("use_ai_questions"):
        return jsonify({"error": "Either questions or use_ai_questions is required"}), 400
    
    # Support multiple classroom IDs for mixed-subject assessments
    classroom_ids = data.get("classroom_ids", [])
    primary_classroom_id = data.get("classroom_id")
    if primary_classroom_id and primary_classroom_id not in classroom_ids:
        classroom_ids.insert(0, primary_classroom_id)
    
    # Build assessment
    assessment = Assessment(
        id=str(uuid4()),
        topic=data.get("topic", "General"),
        subject=data.get("subject", "General"),
        title=data.get("title", f"{data.get('topic', 'General')} Assessment"),
        description=data.get("description"),
        questions=data.get("questions", []),
        difficulty=data.get("difficulty", "medium"),
        time_limit_minutes=data.get("time_limit_minutes", 30),
        is_adaptive=data.get("is_adaptive", False),
        scheduled_date=datetime.fromisoformat(data["scheduled_date"]) if data.get("scheduled_date") else None,
        created_by=request.user_id,
        # New fields
        assessment_type=assessment_type,
        classroom_id=classroom_ids[0] if classroom_ids else primary_classroom_id,  # Primary classroom
        use_ai_questions=data.get("use_ai_questions", False),
        source_topics=data.get("source_topics") or data.get("topic_ids", []),
        source_chapters=data.get("source_chapters", []),
        include_weak_topics=data.get("include_weak_topics", False),
        is_challenge=data.get("is_challenge", False)
    )
    
    # Store additional classroom IDs in description if mixed assessment
    if len(classroom_ids) > 1:
        assessment.description = f"Mixed assessment from {len(classroom_ids)} subjects. Classrooms: {','.join(classroom_ids)}"
    
    db.session.add(assessment)
    db.session.commit()
    
    return jsonify({"assessment": assessment.to_dict()}), 201



@assessments_bp.route("/<assessment_id>", methods=["DELETE"])
@require_auth
def delete_assessment(assessment_id):
    """Delete an assessment (owner or teacher only)"""
    assessment = Assessment.query.get(assessment_id)
    
    if not assessment:
        return jsonify({"error": "Assessment not found"}), 404
    
    user = User.query.get(request.user_id)
    if assessment.created_by != request.user_id and user.role != "teacher":
        return jsonify({"error": "Not authorized to delete this assessment"}), 403
    
    db.session.delete(assessment)
    db.session.commit()
    
    return jsonify({"message": "Assessment deleted"}), 200


# ============================================================================
# Assessment Submission
# ============================================================================

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
        id=str(uuid4()),
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
    
    # Update challenge if this is part of one
    if assessment.is_challenge:
        _update_challenge_score(assessment, request.user_id, score)
    
    # =========================================================================
    # Update StudentTopicScore for each topic involved in the assessment
    # This tracks per-topic mastery based on question performance
    # =========================================================================
    from app.models.curriculum import StudentTopicScore
    
    # Group questions by topic_id and update scores
    topic_scores_updated = set()
    for i, question in enumerate(assessment.questions):
        topic_id = question.get("topic_id") or question.get("classroom_topic_id")
        if not topic_id:
            continue
        
        # Get or create StudentTopicScore for this topic
        topic_score = StudentTopicScore.query.filter_by(
            user_id=request.user_id,
            classroom_topic_id=topic_id
        ).first()
        
        if not topic_score:
            topic_score = StudentTopicScore(
                user_id=request.user_id,
                classroom_topic_id=topic_id,
                first_activity_at=datetime.utcnow()
            )
            db.session.add(topic_score)
        
        # Update MCQ score for this question
        is_correct = feedback[i]["is_correct"]
        marks = question.get("marks", 1)
        topic_score.update_mcq_score(correct=is_correct, marks=marks)
        topic_score.last_activity_at = datetime.utcnow()
        topic_scores_updated.add(topic_id)
        
        # Create StudentQuestionResponse record if question is linked to TopicQuestion
        question_id = question.get("id")
        # Note: Skip if question_id is just a generated UUID (not from TopicQuestions table)
        # Only create response records when questions are stored in topic_questions table
        # For AI-generated assessment questions, we track score via StudentTopicScore only
    
    db.session.commit()
    
    # Build full assessment with answers for review
    questions_with_answers = []
    for i, q in enumerate(assessment.questions or []):
        question_data = {
            "id": q.get("id", str(i)),
            "question_type": q.get("question_type", "mcq"),
            "question_text": q.get("question_text", ""),
            "marks": q.get("marks", 1),
            "difficulty": q.get("difficulty", "medium"),
            "correct_answer": q.get("correct_answer"),
            "explanation": q.get("explanation", ""),
        }
        
        # Include options WITH is_correct and explanation
        if q.get("question_type") == "mcq" and q.get("options"):
            question_data["options"] = q.get("options")
        
        questions_with_answers.append(question_data)
    
    # Calculate total marks
    total_marks = sum(q.get("marks", 1) for q in assessment.questions or [])
    
    return jsonify({
        "result": {
            "score": correct,
            "total_marks": total_marks,
            "percentage": round(score, 1),
            "correct_count": correct,
            "total_questions": total,
        },
        "assessment_with_answers": {
            "id": assessment.id,
            "title": assessment.title,
            "topic": assessment.topic,
            "time_limit_minutes": assessment.time_limit_minutes or 30,
            "questions": questions_with_answers,
            "total_marks": total_marks
        },
        "feedback": feedback
    }), 200


def _update_challenge_score(assessment, user_id, score):
    """Update challenge scores when an assessment is completed"""
    # Find challenge where this assessment is used
    challenge = AssessmentChallenge.query.filter(
        (AssessmentChallenge.assessment_id == assessment.id) |
        (AssessmentChallenge.recipient_assessment_id == assessment.id)
    ).first()
    
    if not challenge:
        return
    
    if challenge.sender_id == user_id:
        challenge.sender_score = score
    elif challenge.recipient_id == user_id:
        challenge.recipient_score = score
    
    # Check if both have completed
    if challenge.sender_score is not None and challenge.recipient_score is not None:
        challenge.status = "completed"
        challenge.completed_at = datetime.utcnow()


# ============================================================================
# User Results
# ============================================================================

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


# ============================================================================
# My Assessments (Created by user)
# ============================================================================

@assessments_bp.route("/my-assessments", methods=["GET"])
@require_auth
def get_my_assessments():
    """Get assessments created by current user"""
    assessments = Assessment.query.filter_by(
        created_by=request.user_id
    ).order_by(Assessment.created_at.desc()).limit(50).all()
    
    return jsonify({
        "assessments": [a.to_dict(include_questions=False) for a in assessments],
        "count": len(assessments)
    }), 200


# ============================================================================
# Weak Topics for Assessment Targeting
# ============================================================================

@assessments_bp.route("/weak-topics", methods=["GET"])
@require_auth
def get_weak_topics():
    """Get user's weak topics for assessment targeting"""
    classroom_id = request.args.get("classroom_id")
    
    # Get progress records where user is weak
    weak_progress = Progress.query.filter_by(
        user_id=request.user_id,
        is_weak=True
    ).all()
    
    weak_topics = []
    for p in weak_progress:
        weak_topics.append({
            "topic": p.topic,
            "subject": p.subject,
            "confidence_score": p.confidence_score,
            "last_studied": p.last_studied.isoformat() if p.last_studied else None
        })
    
    # Also check StudentTopicScore for classroom topics if classroom_id provided
    if classroom_id:
        try:
            from app.models.curriculum import StudentTopicScore, ClassroomTopic
            
            low_scores = StudentTopicScore.query.filter(
                StudentTopicScore.user_id == request.user_id,
                StudentTopicScore.mastery_percentage < 50
            ).all()
            
            for score in low_scores:
                topic = ClassroomTopic.query.get(score.classroom_topic_id)
                if topic and topic.classroom_id == classroom_id:
                    weak_topics.append({
                        "topic_id": topic.id,
                        "topic": topic.name,
                        "chapter_id": topic.chapter_id,
                        "mastery_percentage": score.mastery_percentage,
                        "status": score.status
                    })
        except Exception:
            pass  # Models may not exist in test
    
    return jsonify({
        "weak_topics": weak_topics,
        "count": len(weak_topics)
    }), 200


# ============================================================================
# Challenge System
# ============================================================================

@assessments_bp.route("/challenge", methods=["POST"])
@require_auth
def send_challenge():
    """
    Send an assessment challenge to another student.
    Request body:
    - assessment_id: ID of the assessment to challenge with
    - recipient_id: ID of the student to challenge
    - message: Optional challenge message
    """
    data = request.get_json()
    
    assessment_id = data.get("assessment_id")
    recipient_id = data.get("recipient_id")
    message = data.get("message")
    
    if not assessment_id or not recipient_id:
        return jsonify({"error": "assessment_id and recipient_id are required"}), 400
    
    # Validate assessment
    assessment = Assessment.query.get(assessment_id)
    if not assessment:
        return jsonify({"error": "Assessment not found"}), 404
    
    # Can't challenge yourself
    if recipient_id == request.user_id:
        return jsonify({"error": "Cannot challenge yourself"}), 400
    
    # Validate recipient
    recipient = User.query.get(recipient_id)
    if not recipient:
        return jsonify({"error": "Recipient not found"}), 404
    
    # Check if challenge already exists
    existing = AssessmentChallenge.query.filter_by(
        assessment_id=assessment_id,
        sender_id=request.user_id,
        recipient_id=recipient_id,
        status="pending"
    ).first()
    
    if existing:
        return jsonify({"error": "Challenge already sent"}), 409
    
    # Get sender's score if they've completed the assessment
    sender_result = AssessmentResult.query.filter_by(
        user_id=request.user_id,
        assessment_id=assessment_id
    ).order_by(AssessmentResult.completed_at.desc()).first()
    
    challenge = AssessmentChallenge(
        id=str(uuid4()),
        assessment_id=assessment_id,
        sender_id=request.user_id,
        recipient_id=recipient_id,
        status="pending",
        sender_score=sender_result.score if sender_result else None,
        challenge_message=message
    )
    
    db.session.add(challenge)
    db.session.commit()
    
    return jsonify({
        "challenge": challenge.to_dict(include_assessment=True),
        "message": "Challenge sent successfully"
    }), 201


@assessments_bp.route("/challenges/sent", methods=["GET"])
@require_auth
def get_sent_challenges():
    """Get challenges sent by current user"""
    challenges = AssessmentChallenge.query.filter_by(
        sender_id=request.user_id
    ).order_by(AssessmentChallenge.sent_at.desc()).limit(50).all()
    
    return jsonify({
        "challenges": [c.to_dict(include_assessment=True) for c in challenges],
        "count": len(challenges)
    }), 200


@assessments_bp.route("/challenges/received", methods=["GET"])
@require_auth
def get_received_challenges():
    """Get challenges received by current user"""
    status = request.args.get("status")  # Optional filter: pending, accepted, declined, completed
    
    query = AssessmentChallenge.query.filter_by(recipient_id=request.user_id)
    
    if status:
        query = query.filter_by(status=status)
    
    challenges = query.order_by(AssessmentChallenge.sent_at.desc()).limit(50).all()
    
    return jsonify({
        "challenges": [c.to_dict(include_assessment=True) for c in challenges],
        "count": len(challenges)
    }), 200


@assessments_bp.route("/challenges/<challenge_id>/accept", methods=["POST"])
@require_auth
def accept_challenge(challenge_id):
    """Accept a challenge - clones the assessment for the recipient"""
    challenge = AssessmentChallenge.query.get(challenge_id)
    
    if not challenge:
        return jsonify({"error": "Challenge not found"}), 404
    
    if challenge.recipient_id != request.user_id:
        return jsonify({"error": "Not authorized"}), 403
    
    if challenge.status != "pending":
        return jsonify({"error": f"Challenge already {challenge.status}"}), 400
    
    # Clone the assessment for the recipient
    original = Assessment.query.get(challenge.assessment_id)
    if not original:
        return jsonify({"error": "Original assessment not found"}), 404
    
    cloned = Assessment(
        id=str(uuid4()),
        topic=original.topic,
        subject=original.subject,
        title=f"Challenge: {original.title}",
        description=original.description,
        questions=original.questions,
        difficulty=original.difficulty,
        time_limit_minutes=original.time_limit_minutes,
        is_adaptive=original.is_adaptive,
        created_by=request.user_id,
        assessment_type="student_challenge",
        classroom_id=original.classroom_id,
        use_ai_questions=original.use_ai_questions,
        source_topics=original.source_topics,
        source_chapters=original.source_chapters,
        is_challenge=True,
        original_assessment_id=original.id
    )
    
    db.session.add(cloned)
    
    # Update challenge
    challenge.status = "accepted"
    challenge.responded_at = datetime.utcnow()
    challenge.recipient_assessment_id = cloned.id
    
    db.session.commit()
    
    return jsonify({
        "challenge": challenge.to_dict(include_assessment=True),
        "assessment": cloned.to_dict(include_questions=False),
        "message": "Challenge accepted"
    }), 200


@assessments_bp.route("/challenges/<challenge_id>/decline", methods=["POST"])
@require_auth
def decline_challenge(challenge_id):
    """Decline a challenge"""
    challenge = AssessmentChallenge.query.get(challenge_id)
    
    if not challenge:
        return jsonify({"error": "Challenge not found"}), 404
    
    if challenge.recipient_id != request.user_id:
        return jsonify({"error": "Not authorized"}), 403
    
    if challenge.status != "pending":
        return jsonify({"error": f"Challenge already {challenge.status}"}), 400
    
    challenge.status = "declined"
    challenge.responded_at = datetime.utcnow()
    
    db.session.commit()
    
    return jsonify({
        "challenge": challenge.to_dict(),
        "message": "Challenge declined"
    }), 200


@assessments_bp.route("/challenges/<challenge_id>", methods=["GET"])
@require_auth
def get_challenge(challenge_id):
    """Get challenge details including comparison if completed"""
    challenge = AssessmentChallenge.query.get(challenge_id)
    
    if not challenge:
        return jsonify({"error": "Challenge not found"}), 404
    
    # Only sender or recipient can view
    if challenge.sender_id != request.user_id and challenge.recipient_id != request.user_id:
        return jsonify({"error": "Not authorized"}), 403
    
    data = challenge.to_dict(include_assessment=True)
    
    # Add comparison data if completed
    if challenge.status == "completed":
        data["comparison"] = {
            "sender_score": challenge.sender_score,
            "recipient_score": challenge.recipient_score,
            "winner": "sender" if (challenge.sender_score or 0) > (challenge.recipient_score or 0) else "recipient" if (challenge.recipient_score or 0) > (challenge.sender_score or 0) else "tie",
            "difference": abs((challenge.sender_score or 0) - (challenge.recipient_score or 0))
        }
    
    return jsonify({"challenge": data}), 200


# ============================================================================
# Available Students for Challenging
# ============================================================================

@assessments_bp.route("/challenge/students", methods=["GET"])
@require_auth
def get_challengeable_students():
    """Get list of students that can be challenged (classmates)"""
    classroom_id = request.args.get("classroom_id")
    
    try:
        from app.models.classroom import StudentEnrollment
        
        if classroom_id:
            enrollments = StudentEnrollment.query.filter(
                StudentEnrollment.classroom_id == classroom_id,
                StudentEnrollment.user_id != request.user_id,
                StudentEnrollment.status == "active"
            ).all()
            
            student_ids = [e.user_id for e in enrollments]
        else:
            # Get all classmates from any enrolled classroom
            my_enrollments = StudentEnrollment.query.filter_by(
                user_id=request.user_id,
                status="active"
            ).all()
            
            my_classroom_ids = [e.classroom_id for e in my_enrollments]
            
            enrollments = StudentEnrollment.query.filter(
                StudentEnrollment.classroom_id.in_(my_classroom_ids),
                StudentEnrollment.user_id != request.user_id,
                StudentEnrollment.status == "active"
            ).all()
            
            student_ids = list(set([e.user_id for e in enrollments]))
        
        students = User.query.filter(User.id.in_(student_ids)).all()
        
        return jsonify({
            "students": [{
                "id": s.id,
                "username": s.username,
                "name": f"{s.first_name or ''} {s.last_name or ''}".strip() or s.username,
                "avatar_url": s.avatar_url
            } for s in students],
            "count": len(students)
        }), 200
        
    except Exception as e:
        return jsonify({
            "students": [],
            "count": 0,
            "error": str(e)
        }), 200


# ============================================================================
# Daily Revision Assessment
# ============================================================================

@assessments_bp.route("/daily-revision", methods=["GET"])
@require_auth
def get_daily_revision():
    """
    Get the revision assessment for a specific date.
    Query params:
    - date: ISO date string (defaults to today)
    """
    from datetime import date as date_type
    
    date_str = request.args.get("date", date_type.today().isoformat())
    try:
        target_date = date_type.fromisoformat(date_str)
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    
    # Find revision assessment for this user and date
    assessment = Assessment.query.filter_by(
        created_by=request.user_id,
        is_revision_assessment=True,
        revision_date=target_date
    ).first()
    
    if not assessment:
        return jsonify({
            "assessment": None,
            "message": "No revision assessment for this date"
        }), 200
    
    # Check if user has completed it
    result = AssessmentResult.query.filter_by(
        user_id=request.user_id,
        assessment_id=assessment.id
    ).first()
    
    response_data = assessment.to_dict(include_questions=True)
    response_data["completed"] = result is not None
    response_data["score"] = result.score if result else None
    
    return jsonify({"assessment": response_data}), 200


@assessments_bp.route("/generate-daily-revision", methods=["POST"])
@require_auth
def generate_daily_revision():
    """
    Trigger generation of daily revision assessment.
    Request body (optional):
    - date: ISO date string (defaults to today)
    - force: Boolean to regenerate even if assessment exists
    
    This calls the AI service to generate questions based on today's revision schedule.
    """
    from datetime import date as date_type
    import httpx
    import os
    
    data = request.get_json() or {}
    date_str = data.get("date", date_type.today().isoformat())
    force = data.get("force", False)
    
    try:
        target_date = date_type.fromisoformat(date_str)
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    
    # Check if assessment already exists (unless force=True)
    if not force:
        existing = Assessment.query.filter_by(
            created_by=request.user_id,
            is_revision_assessment=True,
            revision_date=target_date
        ).first()
        
        if existing:
            return jsonify({
                "assessment": existing.to_dict(include_questions=False),
                "message": "Revision assessment already exists",
                "already_exists": True
            }), 200
    
    # Get revision topics from the actual revision calendar (SM-2 algorithm)
    from app.routes.revision import get_todays_revision_topics
    
    print(f"[Revision Assessment] Generating for user {request.user_id}, date {target_date}")
    
    revision_topics = get_todays_revision_topics(request.user_id, target_date, max_topics=5)
    print(f"[Revision Assessment] Found {len(revision_topics)} revision topics from calendar")
    for t in revision_topics:
        print(f"  - {t['topic_name']} (mastery: {t['mastery_percentage']}%, reason: {t['reason']}, priority: {t.get('priority', 'N/A')})")
    
    if not revision_topics:
        return jsonify({
            "assessment": None,
            "message": "No topics scheduled for revision today"
        }), 200
    
    # Generate questions using Groq API
    import os
    import json
    import httpx
    
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    generated_questions = []
    questions_per_topic = 3
    
    for topic in revision_topics[:5]:  # Max 5 topics
        topic_name = topic.get("topic_name", "General")
        subject = topic.get("subject_name", "")
        chapter_name = topic.get("chapter_name", "")
        mastery = topic.get("mastery_percentage", 50)
        
        # Determine difficulty based on mastery
        if mastery < 40:
            difficulty = "easy"
        elif mastery < 70:
            difficulty = "medium"
        else:
            difficulty = "hard"
        
        prompt = f"""You are a quiz generator. Generate exactly {questions_per_topic} multiple choice questions for students revising the topic: "{topic_name}" in {subject}.

Topic: {topic_name}
Subject: {subject}
Chapter: {chapter_name}
Student's current mastery: {mastery}%
Difficulty level: {difficulty}

Return ONLY a valid JSON array with no other text. Each question must have this exact structure:
[
  {{
    "question_text": "Clear, specific question about the topic?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "A",
    "explanation": "Brief explanation why this is correct"
  }}
]

Important:
- Make questions specific to {topic_name}
- The correct_answer must be just a single letter (A, B, C, or D)
- Options should be plausible but only one correct
- Questions should test understanding, not just memorization
- Return ONLY the JSON array, no markdown, no extra text"""

        try:
            if GROQ_API_KEY:
                response = httpx.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 1500
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"].strip()
                    
                    # Clean up markdown code blocks if present
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    
                    questions = json.loads(content)
                    
                    for idx, q in enumerate(questions):
                        if "question_text" in q and "options" in q and "correct_answer" in q:
                            # Normalize correct_answer to single letter
                            correct = q["correct_answer"].strip().upper()
                            if len(correct) > 1:
                                correct = correct[0]
                            
                            generated_questions.append({
                                "id": f"{topic.get('topic_id', 'q')}_{idx}",
                                "question_type": "mcq",
                                "question_text": q["question_text"],
                                "options": q["options"],
                                "correct_answer": correct,
                                "explanation": q.get("explanation", ""),
                                "marks": 1,
                                "difficulty": difficulty,
                                "topic_id": topic.get("topic_id"),
                                "topic": topic_name
                            })
        except Exception as e:
            print(f"[Revision Assessment] Error generating questions for {topic_name}: {str(e)}")
            # Continue to next topic, don't fail completely
            continue
    
    # If no questions generated, create fallback questions
    if not generated_questions:
        for idx, topic in enumerate(revision_topics[:3]):
            topic_name = topic.get("topic_name", "Topic")
            mastery = topic.get("mastery_percentage", 50)
            difficulty = "easy" if mastery < 40 else "medium" if mastery < 70 else "hard"
            
            generated_questions.append({
                "id": f"fallback_{idx}",
                "question_type": "mcq", 
                "question_text": f"Which of the following best describes {topic_name}?",
                "options": [
                    f"A core concept in {topic_name}",
                    f"An unrelated topic",
                    f"A different subject entirely", 
                    f"None of the above"
                ],
                "correct_answer": "A",
                "explanation": f"Review the fundamentals of {topic_name}.",
                "marks": 1,
                "difficulty": difficulty,
                "topic_id": topic.get("topic_id"),
                "topic": topic_name
            })
    
    if not generated_questions:
        return jsonify({
            "error": "Failed to generate questions",
            "topics_found": len(revision_topics)
        }), 500
    
    # Create the assessment
    topic_names = [t.get("topic_name", "") for t in revision_topics[:3]]
    topics_str = ", ".join(topic_names)
    if len(revision_topics) > 3:
        topics_str += f" +{len(revision_topics) - 3} more"
    
    assessment = Assessment(
        id=str(uuid4()),
        title=f"Automated Revision Assessment - {date_str}",
        topic=topics_str,
        subject="Revision",
        description=f"Daily revision quiz covering {len(revision_topics)} topic(s): {topics_str}",
        questions=generated_questions,
        difficulty="mixed",
        time_limit_minutes=max(len(generated_questions) * 2, 10),
        assessment_type="self_practice",
        use_ai_questions=True,
        created_by=request.user_id,
        is_revision_assessment=True,
        revision_date=target_date
    )
    
    db.session.add(assessment)
    db.session.commit()
    
    return jsonify({
        "assessment": assessment.to_dict(include_questions=False),
        "topics_covered": [t.get("topic_name") for t in revision_topics],
        "questions_generated": len(generated_questions),
        "message": "Revision assessment created successfully"
    }), 201


@assessments_bp.route("/<assessment_id>/append-questions", methods=["PATCH"])
@require_auth
def append_questions(assessment_id):
    """
    Append new questions to an existing assessment.
    Request body:
    - questions: List of question objects to append
    """
    assessment = Assessment.query.get(assessment_id)
    
    if not assessment:
        return jsonify({"error": "Assessment not found"}), 404
    
    if assessment.created_by != request.user_id:
        return jsonify({"error": "Not authorized to modify this assessment"}), 403
    
    data = request.get_json()
    new_questions = data.get("questions", [])
    
    if not new_questions:
        return jsonify({"error": "No questions provided"}), 400
    
    # Append new questions
    existing = assessment.questions or []
    assessment.questions = existing + new_questions
    
    # Update time limit
    assessment.time_limit_minutes = len(assessment.questions) * 2
    
    db.session.commit()
    
    return jsonify({
        "assessment": assessment.to_dict(include_questions=False),
        "questions_added": len(new_questions),
        "total_questions": len(assessment.questions)
    }), 200
