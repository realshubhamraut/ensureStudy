"""
Assignment Routes
Handles assignment creation, viewing, and submission endpoints
"""
from flask import Blueprint, request, jsonify
from functools import wraps
from app import db
from app.models.assignment import Assignment, AssignmentAttachment, Submission, SubmissionFile
from app.models.user import User
from app.models.classroom import Classroom
from app.utils.jwt_handler import verify_token
from datetime import datetime

assignment_bp = Blueprint('assignment', __name__, url_prefix='/api')


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


def token_required(f):
    """Decorator to require any authenticated user"""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        request.current_user = user
        return f(*args, **kwargs)
    return decorated


def teacher_required(f):
    """Decorator to require teacher role"""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user or user.role != "teacher":
            return jsonify({"error": "Teacher access required"}), 403
        request.current_user = user
        return f(*args, **kwargs)
    return decorated


def student_required(f):
    """Decorator to require student role"""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user or user.role != "student":
            return jsonify({"error": "Student access required"}), 403
        request.current_user = user
        return f(*args, **kwargs)
    return decorated


# ========== TEACHER ENDPOINTS ==========

@assignment_bp.route('/classroom/<classroom_id>/assignments', methods=['POST'])
@token_required
@teacher_required
def create_assignment(classroom_id):
    """Create a new assignment for a classroom"""
    try:
        current_user = request.current_user
        # Verify teacher owns this classroom
        classroom = Classroom.query.filter_by(id=classroom_id, teacher_id=current_user.id).first()
        if not classroom:
            return jsonify({'error': 'Classroom not found or unauthorized'}), 404
        
        data = request.json
        
        # Create assignment
        assignment = Assignment(
            classroom_id=classroom_id,
            title=data.get('title'),
            description=data.get('description'),
            due_date=datetime.fromisoformat(data['due_date']) if data.get('due_date') else None,
            points=data.get('points'),
            status=data.get('status', 'published')
        )
        
        db.session.add(assignment)
        db.session.flush()  # Get assignment ID
        
        # Add attachments
        attachments = data.get('attachments', [])
        for att_data in attachments:
            attachment = AssignmentAttachment(
                assignment_id=assignment.id,
                type=att_data.get('type'),
                url=att_data.get('url'),
                filename=att_data.get('filename'),
                file_size=att_data.get('file_size')
            )
            db.session.add(attachment)
        
        db.session.commit()
        
        return jsonify({
            'message': 'Assignment created successfully',
            'assignment': assignment.to_dict(include_attachments=True)
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@assignment_bp.route('/classroom/<classroom_id>/assignments', methods=['GET'])
@token_required
def list_assignments(classroom_id):
    """List all assignments for a classroom"""
    try:
        current_user = request.current_user
        # Check if user has access to this classroom
        classroom = Classroom.query.get(classroom_id)
        if not classroom:
            return jsonify({'error': 'Classroom not found'}), 404
        
        # Check if teacher or enrolled student
        is_teacher = classroom.teacher_id == current_user.id
        is_student = any(e.student_id == current_user.id for e in classroom.student_enrollments)
        
        if not (is_teacher or is_student):
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Get assignments
        assignments = Assignment.query.filter_by(classroom_id=classroom_id).order_by(Assignment.created_at.desc()).all()
        
        # Format based on role
        if is_teacher:
            result = [a.to_dict(include_attachments=True, include_submissions=True) for a in assignments]
        else:
            # Student view - include their submission
            result = [a.to_dict(include_attachments=True, include_submissions=False, student_id=current_user.id) for a in assignments]
        
        return jsonify({'assignments': result}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@assignment_bp.route('/assignment/<assignment_id>', methods=['GET'])
@token_required
def get_assignment(assignment_id):
    """Get assignment details"""
    try:
        current_user = request.current_user
        assignment = Assignment.query.get(assignment_id)
        if not assignment:
            return jsonify({'error': 'Assignment not found'}), 404
        
        # Check access
        classroom = assignment.classroom
        is_teacher = classroom.teacher_id == current_user.id
        is_student = any(e.student_id == current_user.id for e in classroom.student_enrollments)
        
        if not (is_teacher or is_student):
            return jsonify({'error': 'Unauthorized'}), 403
        
        if is_teacher:
            data = assignment.to_dict(include_attachments=True, include_submissions=True)
        else:
            data = assignment.to_dict(include_attachments=True, student_id=current_user.id)
        
        return jsonify({'assignment': data}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@assignment_bp.route('/assignment/<assignment_id>', methods=['PUT'])
@token_required
@teacher_required
def update_assignment(assignment_id):
    """Update an assignment"""
    try:
        current_user = request.current_user
        assignment = Assignment.query.get(assignment_id)
        if not assignment:
            return jsonify({'error': 'Assignment not found'}), 404
        
        # Verify teacher owns this assignment
        if assignment.classroom.teacher_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.json
        
        # Update fields
        if 'title' in data:
            assignment.title = data['title']
        if 'description' in data:
            assignment.description = data['description']
        if 'due_date' in data:
            assignment.due_date = datetime.fromisoformat(data['due_date']) if data['due_date'] else None
        if 'points' in data:
            assignment.points = data['points']
        if 'status' in data:
            assignment.status = data['status']
        
        assignment.updated_at = datetime.utcnow()
        
        # Update attachments if provided
        if 'attachments' in data:
            # Remove old attachments
            AssignmentAttachment.query.filter_by(assignment_id=assignment.id).delete()
            
            # Add new attachments
            for att_data in data['attachments']:
                attachment = AssignmentAttachment(
                    assignment_id=assignment.id,
                    type=att_data.get('type'),
                    url=att_data.get('url'),
                    filename=att_data.get('filename'),
                    file_size=att_data.get('file_size')
                )
                db.session.add(attachment)
        
        db.session.commit()
        
        return jsonify({
            'message': 'Assignment updated successfully',
            'assignment': assignment.to_dict(include_attachments=True)
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@assignment_bp.route('/assignment/<assignment_id>', methods=['DELETE'])
@token_required
@teacher_required
def delete_assignment(assignment_id):
    """Delete an assignment"""
    try:
        current_user = request.current_user
        assignment = Assignment.query.get(assignment_id)
        if not assignment:
            return jsonify({'error': 'Assignment not found'}), 404
        
        # Verify teacher owns this assignment
        if assignment.classroom.teacher_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        db.session.delete(assignment)
        db.session.commit()
        
        return jsonify({'message': 'Assignment deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@assignment_bp.route('/assignment/<assignment_id>/submissions', methods=['GET'])
@token_required
@teacher_required
def get_submissions(assignment_id):
    """Get all submissions for an assignment (teacher only)"""
    try:
        current_user = request.current_user
        assignment = Assignment.query.get(assignment_id)
        if not assignment:
            return jsonify({'error': 'Assignment not found'}), 404
        
        # Verify teacher owns this assignment
        if assignment.classroom.teacher_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        submissions = Submission.query.filter_by(assignment_id=assignment_id).all()
        
        return jsonify({
            'submissions': [s.to_dict(include_student=True, include_files=True) for s in submissions]
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@assignment_bp.route('/submission/<submission_id>/grade', methods=['PUT'])
@token_required
@teacher_required
def grade_submission(submission_id):
    """Grade a student submission"""
    try:
        current_user = request.current_user
        submission = Submission.query.get(submission_id)
        if not submission:
            return jsonify({'error': 'Submission not found'}), 404
        
        # Verify teacher owns the assignment's classroom
        if submission.assignment.classroom.teacher_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.json
        
        submission.grade = data.get('grade')
        submission.feedback = data.get('feedback')
        submission.status = 'graded'
        
        db.session.commit()
        
        return jsonify({
            'message': 'Submission graded successfully',
            'submission': submission.to_dict(include_student=True, include_files=True)
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


# ========== STUDENT ENDPOINTS ==========

@assignment_bp.route('/assignment/<assignment_id>/submit', methods=['POST'])
@token_required
@student_required
def submit_assignment(assignment_id):
    """Submit an assignment"""
    try:
        current_user = request.current_user
        assignment = Assignment.query.get(assignment_id)
        if not assignment:
            return jsonify({'error': 'Assignment not found'}), 404
        
        # Verify student is enrolled
        classroom = assignment.classroom
        is_enrolled = any(e.student_id == current_user.id for e in classroom.student_enrollments)
        if not is_enrolled:
            return jsonify({'error': 'Not enrolled in this classroom'}), 403
        
        # Check if already submitted
        existing = Submission.query.filter_by(
            assignment_id=assignment_id,
            student_id=current_user.id
        ).first()
        
        if existing:
            return jsonify({'error': 'Already submitted. Use update endpoint to modify.'}), 400
        
        data = request.json
        
        # Create submission
        submission = Submission(
            assignment_id=assignment_id,
            student_id=current_user.id,
            status='submitted'
        )
        
        db.session.add(submission)
        db.session.flush()
        
        # Add submission files
        files = data.get('files', [])
        for file_data in files:
            sub_file = SubmissionFile(
                submission_id=submission.id,
                url=file_data.get('url'),
                filename=file_data.get('filename'),
                file_size=file_data.get('file_size'),
                type=file_data.get('type', 'file')
            )
            db.session.add(sub_file)
        
        db.session.commit()
        
        return jsonify({
            'message': 'Assignment submitted successfully',
            'submission': submission.to_dict(include_files=True)
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@assignment_bp.route('/submission/<submission_id>', methods=['PUT'])
@token_required
@student_required
def update_submission(submission_id):
    """Update a submission (before grading)"""
    try:
        current_user = request.current_user
        submission = Submission.query.get(submission_id)
        if not submission:
            return jsonify({'error': 'Submission not found'}), 404
        
        # Verify student owns this submission
        if submission.student_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Can't update if already graded
        if submission.status == 'graded':
            return jsonify({'error': 'Cannot update graded submission'}), 400
        
        data = request.json
        
        # Update files
        if 'files' in data:
            # Remove old files
            SubmissionFile.query.filter_by(submission_id=submission.id).delete()
            
            # Add new files
            for file_data in data['files']:
                sub_file = SubmissionFile(
                    submission_id=submission.id,
                    url=file_data.get('url'),
                    filename=file_data.get('filename'),
                    file_size=file_data.get('file_size'),
                    type=file_data.get('type', 'file')
                )
                db.session.add(sub_file)
        
        submission.submitted_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'message': 'Submission updated successfully',
            'submission': submission.to_dict(include_files=True)
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@assignment_bp.route('/submission/<submission_id>', methods=['DELETE'])
@token_required
@student_required
def delete_submission(submission_id):
    """Delete/unsubmit an assignment"""
    try:
        current_user = request.current_user
        submission = Submission.query.get(submission_id)
        if not submission:
            return jsonify({'error': 'Submission not found'}), 404
        
        # Verify student owns this submission
        if submission.student_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Can't delete if already graded
        if submission.status == 'graded':
            return jsonify({'error': 'Cannot delete graded submission'}), 400
        
        db.session.delete(submission)
        db.session.commit()
        
        return jsonify({'message': 'Submission deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400
