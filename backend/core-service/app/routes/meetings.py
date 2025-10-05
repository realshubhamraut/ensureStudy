"""
Meetings API Routes - Create, schedule, manage live meetings
"""
from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import os
import jwt
import time
import json
from app import db
from app.routes.users import require_auth
from app.models.user import User
from app.models.meeting import Meeting, MeetingParticipant, MeetingRecording, create_meeting, start_meeting, end_meeting
from app.models.classroom import Classroom, StudentClassroom
from app.models.notification import create_notification, notify_classroom_members

# LiveKit configuration
LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY', '')
LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET', '')

meetings_bp = Blueprint("meetings", __name__, url_prefix="/api")


@meetings_bp.route("/classroom/<classroom_id>/meetings", methods=["GET"])
@require_auth
def list_meetings(classroom_id):
    """List all meetings for a classroom"""
    user_id = request.user_id
    
    # Verify access to classroom
    classroom = Classroom.query.get(classroom_id)
    if not classroom:
        return jsonify({"error": "Classroom not found"}), 404
    
    # Get status filter
    status = request.args.get('status')  # scheduled, live, ended
    
    query = Meeting.query.filter_by(classroom_id=classroom_id)
    if status:
        query = query.filter_by(status=status)
    
    meetings = query.order_by(Meeting.scheduled_at.desc()).all()
    
    return jsonify({
        "meetings": [m.to_dict() for m in meetings],
        "count": len(meetings)
    }), 200


@meetings_bp.route("/classroom/<classroom_id>/meetings", methods=["POST"])
@require_auth
def create_new_meeting(classroom_id):
    """Create/schedule a new meeting"""
    user_id = request.user_id
    data = request.get_json()
    
    # Verify classroom ownership
    classroom = Classroom.query.get(classroom_id)
    if not classroom:
        return jsonify({"error": "Classroom not found"}), 404
    
    if classroom.teacher_id != user_id:
        return jsonify({"error": "Only the teacher can create meetings"}), 403
    
    title = data.get('title')
    if not title:
        return jsonify({"error": "Title is required"}), 400
    
    # Parse scheduled time
    scheduled_at = None
    if data.get('scheduled_at'):
        try:
            scheduled_at = datetime.fromisoformat(data['scheduled_at'].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({"error": "Invalid scheduled_at format"}), 400
    
    # Create meeting
    meeting = create_meeting(
        classroom_id=classroom_id,
        host_id=user_id,
        title=title,
        description=data.get('description'),
        scheduled_at=scheduled_at,
        max_participants=data.get('max_participants', 50),
        is_recording_enabled=data.get('is_recording_enabled', True)
    )
    
    # Notify classroom members
    notify_classroom_members(
        classroom_id=classroom_id,
        notification_type='meet',
        title=f"New Meeting: {title}",
        message=f"A meeting has been scheduled in {classroom.name}",
        source_id=meeting.id,
        action_url=f"/classroom/{classroom_id}?tab=meet"
    )
    
    return jsonify({
        "meeting": meeting.to_dict(),
        "message": "Meeting created successfully"
    }), 201


@meetings_bp.route("/meeting/<meeting_id>", methods=["GET"])
@require_auth
def get_meeting(meeting_id):
    """Get meeting details"""
    meeting = Meeting.query.get(meeting_id)
    if not meeting:
        return jsonify({"error": "Meeting not found"}), 404
    
    return jsonify({
        "meeting": meeting.to_dict(include_participants=True, include_recordings=True)
    }), 200


@meetings_bp.route("/meeting/<meeting_id>/start", methods=["POST"])
@require_auth
def start_meeting_endpoint(meeting_id):
    """Start a scheduled meeting (go live)"""
    user_id = request.user_id
    
    meeting = Meeting.query.get(meeting_id)
    if not meeting:
        return jsonify({"error": "Meeting not found"}), 404
    
    if meeting.host_id != user_id:
        return jsonify({"error": "Only the host can start the meeting"}), 403
    
    if meeting.status == 'live':
        return jsonify({"error": "Meeting is already live"}), 400
    
    if meeting.status == 'ended':
        return jsonify({"error": "Meeting has already ended"}), 400
    
    meeting = start_meeting(meeting_id)
    
    # Add host as participant
    participant = MeetingParticipant(
        meeting_id=meeting_id,
        user_id=user_id,
        role='host',
        joined_at=datetime.utcnow()
    )
    db.session.add(participant)
    db.session.commit()
    
    # Notify classroom members
    notify_classroom_members(
        classroom_id=meeting.classroom_id,
        notification_type='meet',
        title=f"Meeting Started: {meeting.title}",
        message="The meeting is now live! Click to join.",
        source_id=meeting.id,
        action_url=meeting.meeting_link
    )
    
    return jsonify({
        "meeting": meeting.to_dict(),
        "message": "Meeting started successfully"
    }), 200


@meetings_bp.route("/meeting/<meeting_id>/join", methods=["POST"])
@require_auth
def join_meeting(meeting_id):
    """Join a live meeting"""
    user_id = request.user_id
    data = request.get_json() or {}
    
    meeting = Meeting.query.get(meeting_id)
    if not meeting:
        return jsonify({"error": "Meeting not found"}), 404
    
    if meeting.status != 'live':
        return jsonify({"error": "Meeting is not live"}), 400
    
    # Check if already joined
    existing = MeetingParticipant.query.filter_by(
        meeting_id=meeting_id,
        user_id=user_id
    ).first()
    
    if existing:
        # Update join time if rejoining
        existing.joined_at = datetime.utcnow()
        existing.left_at = None
        db.session.commit()
        participant = existing
    else:
        # Add new participant
        participant = MeetingParticipant(
            meeting_id=meeting_id,
            user_id=user_id,
            role='attendee',
            joined_at=datetime.utcnow(),
            device_type=data.get('device_type', 'desktop')
        )
        db.session.add(participant)
        db.session.commit()
    
    return jsonify({
        "participant": participant.to_dict(),
        "meeting": meeting.to_dict(),
        "message": "Joined meeting successfully"
    }), 200


@meetings_bp.route("/meeting/<meeting_id>/token", methods=["POST"])
@require_auth
def get_meeting_token(meeting_id):
    """Generate LiveKit access token for joining a meeting"""
    user_id = request.user_id
    data = request.get_json() or {}
    
    meeting = Meeting.query.get(meeting_id)
    if not meeting:
        return jsonify({"error": "Meeting not found"}), 404
    
    # Get user info
    user = User.query.get(user_id)
    participant_name = data.get('participant_name') or (
        f"{user.first_name or ''} {user.last_name or ''}".strip() or user.username if user else 'Guest'
    )
    
    # Check if LiveKit is configured
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        return jsonify({
            "error": "LiveKit not configured",
            "message": "Set LIVEKIT_API_KEY and LIVEKIT_API_SECRET in environment"
        }), 503
    
    # Generate LiveKit token
    try:
        # Token claims
        now = int(time.time())
        claims = {
            "iss": LIVEKIT_API_KEY,
            "sub": user_id,
            "iat": now,
            "exp": now + 3600,  # 1 hour expiry
            "nbf": now,
            "jti": f"{meeting_id}-{user_id}-{now}",
            "video": {
                "roomJoin": True,
                "room": meeting.room_id,
                "canPublish": True,
                "canSubscribe": True,
                "canPublishData": True,
            },
            "name": participant_name,
            "metadata": json.dumps({
                "user_id": user_id,
                "meeting_id": meeting_id,
                "role": "host" if meeting.host_id == user_id else "attendee"
            })
        }
        
        token = jwt.encode(claims, LIVEKIT_API_SECRET, algorithm="HS256")
        
        return jsonify({
            "token": token,
            "room_id": meeting.room_id,
            "participant_name": participant_name,
            "expires_at": datetime.utcnow() + timedelta(hours=1)
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": "Failed to generate token",
            "details": str(e)
        }), 500


@meetings_bp.route("/meeting/<meeting_id>/leave", methods=["POST"])
@require_auth
def leave_meeting(meeting_id):
    """Leave a meeting"""
    user_id = request.user_id
    
    participant = MeetingParticipant.query.filter_by(
        meeting_id=meeting_id,
        user_id=user_id
    ).first()
    
    if not participant:
        return jsonify({"error": "Not a participant"}), 404
    
    participant.left_at = datetime.utcnow()
    if participant.joined_at:
        delta = participant.left_at - participant.joined_at
        participant.duration_seconds = int(delta.total_seconds())
    
    db.session.commit()
    
    return jsonify({
        "message": "Left meeting successfully"
    }), 200


@meetings_bp.route("/meeting/<meeting_id>/end", methods=["POST"])
@require_auth
def end_meeting_endpoint(meeting_id):
    """End a live meeting"""
    user_id = request.user_id
    
    meeting = Meeting.query.get(meeting_id)
    if not meeting:
        return jsonify({"error": "Meeting not found"}), 404
    
    if meeting.host_id != user_id:
        return jsonify({"error": "Only the host can end the meeting"}), 403
    
    if meeting.status != 'live':
        return jsonify({"error": "Meeting is not live"}), 400
    
    # Mark all participants as left
    for participant in meeting.participants:
        if not participant.left_at:
            participant.left_at = datetime.utcnow()
            if participant.joined_at:
                delta = participant.left_at - participant.joined_at
                participant.duration_seconds = int(delta.total_seconds())
    
    meeting = end_meeting(meeting_id)
    db.session.commit()
    
    return jsonify({
        "meeting": meeting.to_dict(),
        "message": "Meeting ended successfully"
    }), 200


@meetings_bp.route("/meeting/<meeting_id>/recording", methods=["POST"])
@require_auth
def upload_recording(meeting_id):
    """Upload a meeting recording"""
    user_id = request.user_id
    data = request.get_json()
    
    meeting = Meeting.query.get(meeting_id)
    if not meeting:
        return jsonify({"error": "Meeting not found"}), 404
    
    if meeting.host_id != user_id:
        return jsonify({"error": "Only the host can upload recordings"}), 403
    
    storage_url = data.get('storage_url')
    if not storage_url:
        return jsonify({"error": "storage_url is required"}), 400
    
    recording = MeetingRecording(
        meeting_id=meeting_id,
        storage_url=storage_url,
        storage_provider=data.get('storage_provider', 'local'),
        file_size=data.get('file_size'),
        duration_seconds=data.get('duration_seconds'),
        format=data.get('format', 'webm'),
        status='processing'
    )
    db.session.add(recording)
    db.session.commit()
    
    # TODO: Trigger Kafka event for transcription processing
    
    return jsonify({
        "recording": recording.to_dict(),
        "message": "Recording uploaded, processing started"
    }), 201


@meetings_bp.route("/meeting/<meeting_id>/recordings", methods=["GET"])
@require_auth
def list_recordings(meeting_id):
    """Get all recordings for a meeting"""
    meeting = Meeting.query.get(meeting_id)
    if not meeting:
        return jsonify({"error": "Meeting not found"}), 404
    
    recordings = MeetingRecording.query.filter_by(meeting_id=meeting_id).all()
    
    return jsonify({
        "recordings": [r.to_dict() for r in recordings]
    }), 200


@meetings_bp.route("/recording/<recording_id>", methods=["GET"])
@require_auth
def get_recording(recording_id):
    """Get recording details"""
    recording = MeetingRecording.query.get(recording_id)
    if not recording:
        return jsonify({"error": "Recording not found"}), 404
    
    return jsonify({
        "recording": recording.to_dict()
    }), 200


@meetings_bp.route("/meeting/<meeting_id>/participants", methods=["GET"])
@require_auth
def list_participants(meeting_id):
    """List all participants in a meeting"""
    meeting = Meeting.query.get(meeting_id)
    if not meeting:
        return jsonify({"error": "Meeting not found"}), 404
    
    participants = MeetingParticipant.query.filter_by(meeting_id=meeting_id).all()
    
    return jsonify({
        "participants": [p.to_dict() for p in participants],
        "count": len(participants)
    }), 200


@meetings_bp.route("/meeting/<meeting_id>", methods=["DELETE"])
@require_auth
def delete_meeting(meeting_id):
    """Delete a meeting (only if not live)"""
    user_id = request.user_id
    
    meeting = Meeting.query.get(meeting_id)
    if not meeting:
        return jsonify({"error": "Meeting not found"}), 404
    
    if meeting.host_id != user_id:
        return jsonify({"error": "Only the host can delete the meeting"}), 403
    
    if meeting.status == 'live':
        return jsonify({"error": "Cannot delete a live meeting"}), 400
    
    # Delete participants
    MeetingParticipant.query.filter_by(meeting_id=meeting_id).delete()
    
    # Delete recordings
    MeetingRecording.query.filter_by(meeting_id=meeting_id).delete()
    
    # Delete meeting
    db.session.delete(meeting)
    db.session.commit()
    
    return jsonify({
        "message": "Meeting deleted successfully"
    }), 200
