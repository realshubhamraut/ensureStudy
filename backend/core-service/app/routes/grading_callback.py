"""
Grading Callback Routes
Receives grading results from AI service and updates submission
"""
from flask import Blueprint, request, jsonify
from app import db
from app.models.assignment import Submission
from app.models.notification import create_notification
from datetime import datetime

grading_bp = Blueprint('grading', __name__, url_prefix='/api/grading')


@grading_bp.route('/callback', methods=['POST'])
def grading_callback():
    """
    Callback endpoint for AI grading service.
    
    Called by ai-service when grading completes.
    Updates submission with grade and sends notification to student.
    """
    try:
        data = request.json
        
        submission_id = data.get('submission_id')
        if not submission_id:
            return jsonify({'error': 'submission_id required'}), 400
        
        submission = Submission.query.get(submission_id)
        if not submission:
            return jsonify({'error': 'Submission not found'}), 404
        
        # Update submission with grading results
        submission.grade = data.get('grade', 0)
        submission.feedback = data.get('feedback', '')
        submission.status = data.get('status', 'graded')
        
        # Store detailed feedback as JSON if provided
        if hasattr(submission, 'detailed_feedback') and data.get('detailed_feedback'):
            submission.detailed_feedback = data.get('detailed_feedback')
        
        # Mark as AI graded
        if hasattr(submission, 'ai_graded'):
            submission.ai_graded = data.get('ai_graded', True)
        if hasattr(submission, 'ai_confidence'):
            submission.ai_confidence = data.get('confidence', 0.0)
        if hasattr(submission, 'graded_at'):
            submission.graded_at = datetime.utcnow()
        
        db.session.commit()
        
        # Create notification for student
        student_id = data.get('student_id') or submission.student_id
        assignment = submission.assignment
        
        grade_text = f"{submission.grade}/{assignment.points}" if assignment.points else f"{submission.grade}"
        
        # Only notify for successful grading
        if data.get('status') != 'failed_grading':
            try:
                create_notification(
                    user_id=student_id,
                    type='result',
                    title=f'Assignment Graded: {assignment.title}',
                    message=f'You scored {grade_text}. {data.get("feedback", "")[:100]}',
                    source_id=submission.id,
                    source_type='submission',
                    action_url=f'/student/classroom/{assignment.classroom_id}'
                )
                print(f"[Grading] Notification sent to student {student_id}")
            except Exception as e:
                print(f"[Grading] Failed to send notification: {e}")
        
        print(f"[Grading] Submission {submission_id} graded: {submission.grade}/{assignment.points}")
        
        return jsonify({
            'success': True,
            'message': 'Grading results saved',
            'submission_id': submission_id,
            'grade': submission.grade,
            'status': submission.status
        }), 200
        
    except Exception as e:
        db.session.rollback()
        print(f"[Grading] Callback error: {e}")
        return jsonify({'error': str(e)}), 500


@grading_bp.route('/regrade/<submission_id>', methods=['POST'])
def request_regrade(submission_id):
    """
    Request a regrade for a submission.
    Teacher can trigger this to re-run AI grading.
    """
    try:
        from app.utils.jwt_handler import verify_token
        
        # Verify authorization
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "Authentication required"}), 401
        
        token = auth_header.split()[1]
        payload = verify_token(token)
        
        submission = Submission.query.get(submission_id)
        if not submission:
            return jsonify({'error': 'Submission not found'}), 404
        
        # Verify teacher owns the classroom
        assignment = submission.assignment
        if assignment.classroom.teacher_id != payload.get('user_id'):
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Reset status and trigger re-grading
        submission.status = 'grading'
        db.session.commit()
        
        # Get PDFs and trigger AI service
        import os
        import requests
        
        ai_service_url = os.getenv('AI_SERVICE_URL', 'http://localhost:9001')
        teacher_pdfs = [att.url for att in assignment.attachments if att.filename and att.filename.endswith('.pdf')]
        student_pdfs = [f.url for f in submission.files if f.filename and f.filename.endswith('.pdf')]
        
        if teacher_pdfs and student_pdfs:
            try:
                requests.post(f"{ai_service_url}/api/grade/submission", json={
                    'assignment_id': assignment.id,
                    'submission_id': submission.id,
                    'teacher_pdf_url': teacher_pdfs[0],
                    'student_pdf_urls': student_pdfs,
                    'max_points': assignment.points or 100,
                    'classroom_id': assignment.classroom_id,
                    'student_id': submission.student_id
                }, timeout=5)
                
                return jsonify({
                    'success': True,
                    'message': 'Regrade requested',
                    'submission_id': submission_id
                }), 200
            except Exception as e:
                submission.status = 'submitted'
                db.session.commit()
                return jsonify({'error': f'Failed to trigger AI service: {e}'}), 500
        else:
            submission.status = 'submitted'
            db.session.commit()
            return jsonify({'error': 'No PDFs available for grading'}), 400
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
