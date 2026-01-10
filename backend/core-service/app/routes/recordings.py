"""
Recording Routes - Handle meeting recording uploads, streaming, and management
Supports chunked uploads, finalization, and byte-range video streaming
"""
import os
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file, Response
from functools import wraps
from app import db
from app.models.meeting import Meeting, MeetingRecording
from app.models.user import User
from app.utils.jwt_handler import verify_token

recordings_bp = Blueprint('recordings', __name__, url_prefix='/api/recordings')

# Upload directory for recordings
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'recordings')
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Temporary chunks directory
CHUNKS_DIR = os.path.join(RECORDINGS_DIR, 'chunks')
os.makedirs(CHUNKS_DIR, exist_ok=True)


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


def auth_required(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        request.current_user = user
        return f(*args, **kwargs)
    return decorated


@recordings_bp.route('/upload-chunk', methods=['POST'])
@auth_required
def upload_chunk():
    """
    Upload a chunk of recording data
    Used for progressive upload during recording
    """
    try:
        meeting_id = request.form.get('meeting_id')
        chunk_index = int(request.form.get('chunk_index', 0))
        is_final = request.form.get('is_final', 'false').lower() == 'true'
        
        print(f"[Recording] Upload chunk request: meeting={meeting_id}, chunk={chunk_index}, final={is_final}")
        
        if not meeting_id:
            print("[Recording] Error: meeting_id required")
            return jsonify({'error': 'meeting_id required'}), 400
        
        # Verify meeting exists
        meeting = Meeting.query.get(meeting_id)
        if not meeting:
            print(f"[Recording] Error: Meeting {meeting_id} not found")
            return jsonify({'error': 'Meeting not found'}), 404
        
        # Only host (teacher) can record
        if meeting.host_id != request.current_user.id:
            print(f"[Recording] Error: User {request.current_user.id} is not host {meeting.host_id}")
            return jsonify({'error': 'Only the host can record meetings'}), 403
        
        if 'chunk' not in request.files:
            print("[Recording] Error: No chunk file provided")
            return jsonify({'error': 'No chunk file provided'}), 400
        
        chunk_file = request.files['chunk']
        
        # Save chunk to temp directory
        chunk_dir = os.path.join(CHUNKS_DIR, meeting_id)
        os.makedirs(chunk_dir, exist_ok=True)
        
        chunk_path = os.path.join(chunk_dir, f'chunk_{chunk_index:05d}.webm')
        chunk_file.save(chunk_path)
        
        print(f"[Recording] Saved chunk {chunk_index} to {chunk_path}")
        
        return jsonify({
            'message': 'Chunk uploaded',
            'chunk_index': chunk_index,
            'is_final': is_final
        }), 200
        
    except Exception as e:
        print(f"[Recording] Upload chunk error: {e}")
        return jsonify({'error': str(e)}), 400


@recordings_bp.route('/finalize', methods=['POST'])
@auth_required
def finalize_recording():
    """
    Finalize recording by merging chunks and creating MeetingRecording entry
    """
    try:
        data = request.get_json()
        meeting_id = data.get('meeting_id')
        total_chunks = data.get('total_chunks', 0)
        duration_seconds = data.get('duration_seconds', 0)
        
        print(f"[Recording] Finalize request: meeting={meeting_id}, chunks={total_chunks}, duration={duration_seconds}s")
        
        if not meeting_id:
            print("[Recording] Error: meeting_id required")
            return jsonify({'error': 'meeting_id required'}), 400
        
        # Verify meeting
        meeting = Meeting.query.get(meeting_id)
        if not meeting:
            print(f"[Recording] Error: Meeting {meeting_id} not found")
            return jsonify({'error': 'Meeting not found'}), 404
        
        # Only host (teacher) can finalize recording
        if meeting.host_id != request.current_user.id:
            print(f"[Recording] Error: User {request.current_user.id} is not host {meeting.host_id}")
            return jsonify({'error': 'Only the host can finalize recordings'}), 403
        
        # Merge chunks
        chunk_dir = os.path.join(CHUNKS_DIR, meeting_id)
        if not os.path.exists(chunk_dir):
            print(f"[Recording] Error: No chunks directory found at {chunk_dir}")
            return jsonify({'error': 'No chunks found for this meeting'}), 400
        
        # Create final recording file
        recording_id = str(uuid.uuid4())
        final_filename = f'{recording_id}.webm'
        final_path = os.path.join(RECORDINGS_DIR, final_filename)
        
        # Merge all chunk files
        chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.startswith('chunk_')])
        
        with open(final_path, 'wb') as outfile:
            for chunk_name in chunk_files:
                chunk_path = os.path.join(chunk_dir, chunk_name)
                with open(chunk_path, 'rb') as infile:
                    outfile.write(infile.read())
        
        # Get file size
        file_size = os.path.getsize(final_path)
        
        # Clean up chunks
        for chunk_name in chunk_files:
            os.remove(os.path.join(chunk_dir, chunk_name))
        os.rmdir(chunk_dir)
        
        # Generate storage URL
        base_url = request.host_url.rstrip('/')
        storage_url = f'{base_url}/api/recordings/{recording_id}/stream'
        
        # Create MeetingRecording entry - set to ready so video is playable
        recording = MeetingRecording(
            id=recording_id,
            meeting_id=meeting_id,
            storage_url=storage_url,
            storage_provider='local',
            file_size=file_size,
            duration_seconds=duration_seconds,
            format='webm',
            status='ready'  # Video is playable immediately
        )
        
        db.session.add(recording)
        db.session.commit()
        
        print(f"[Recording] Recording {recording_id} saved and ready for playback")
        
        # Trigger transcription processing in AI service (async, non-blocking)
        try:
            import requests
            ai_service_url = os.getenv('AI_SERVICE_URL', 'http://localhost:8001')
            video_path = final_path
            
            requests.post(
                f'{ai_service_url}/api/process/recording',
                json={
                    'recording_id': recording_id,
                    'meeting_id': meeting_id,
                    'classroom_id': meeting.classroom_id,
                    'video_path': video_path
                },
                timeout=2.0  # Quick timeout - processing happens in background
            )
            print(f"[Recording] Triggered transcription for recording {recording_id}")
        except Exception as e:
            print(f"[Recording] Warning: Could not trigger transcription: {e}")
            # Non-fatal - recording is still ready for playback
        
        return jsonify({
            'message': 'Recording finalized',
            'recording': recording.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@recordings_bp.route('/<recording_id>', methods=['GET'])
@auth_required
def get_recording(recording_id):
    """Get recording metadata"""
    recording = MeetingRecording.query.get(recording_id)
    if not recording:
        return jsonify({'error': 'Recording not found'}), 404
    
    return jsonify({'recording': recording.to_dict()}), 200


@recordings_bp.route('/<recording_id>/stream', methods=['GET'])
def stream_recording(recording_id):
    """
    Stream video with byte-range support for seeking
    No auth required for simpler video player integration
    """
    recording = MeetingRecording.query.get(recording_id)
    if not recording:
        return jsonify({'error': 'Recording not found'}), 404
    
    # Get file path
    file_path = os.path.join(RECORDINGS_DIR, f'{recording_id}.webm')
    if not os.path.exists(file_path):
        return jsonify({'error': 'Recording file not found'}), 404
    
    file_size = os.path.getsize(file_path)
    
    # Handle byte-range requests for seeking
    range_header = request.headers.get('Range')
    
    if range_header:
        # Parse range header
        byte_range = range_header.replace('bytes=', '').split('-')
        start = int(byte_range[0])
        end = int(byte_range[1]) if byte_range[1] else file_size - 1
        
        length = end - start + 1
        
        def generate():
            with open(file_path, 'rb') as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk_size = min(8192, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data
        
        response = Response(
            generate(),
            status=206,
            mimetype='video/webm',
            headers={
                'Content-Range': f'bytes {start}-{end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': length,
                'Content-Type': 'video/webm'
            }
        )
        return response
    
    # Full file request
    return send_file(
        file_path,
        mimetype='video/webm',
        as_attachment=False
    )


@recordings_bp.route('/meeting/<meeting_id>', methods=['GET'])
@auth_required
def list_meeting_recordings(meeting_id):
    """Get all recordings for a meeting"""
    meeting = Meeting.query.get(meeting_id)
    if not meeting:
        return jsonify({'error': 'Meeting not found'}), 404
    
    recordings = MeetingRecording.query.filter_by(meeting_id=meeting_id).order_by(
        MeetingRecording.created_at.desc()
    ).all()
    
    return jsonify({
        'recordings': [r.to_dict() for r in recordings]
    }), 200


@recordings_bp.route('/classroom/<classroom_id>', methods=['GET'])
@auth_required
def list_classroom_recordings(classroom_id):
    """Get all recordings for a classroom (for VOD list)"""
    # Get all meetings in classroom
    meetings = Meeting.query.filter_by(classroom_id=classroom_id).all()
    meeting_ids = [m.id for m in meetings]
    
    if not meeting_ids:
        return jsonify({'recordings': []}), 200
    
    # Get recordings with meeting info
    recordings = MeetingRecording.query.filter(
        MeetingRecording.meeting_id.in_(meeting_ids),
        MeetingRecording.status.in_(['ready', 'processing'])  # Include processing ones
    ).order_by(MeetingRecording.created_at.desc()).all()
    
    # Add meeting info to each recording
    result = []
    for r in recordings:
        data = r.to_dict()
        meeting = next((m for m in meetings if m.id == r.meeting_id), None)
        if meeting:
            data['meeting'] = {
                'id': meeting.id,
                'title': meeting.title,
                'host_name': f"{meeting.host.first_name or ''} {meeting.host.last_name or ''}".strip() if meeting.host else 'Unknown',
                'started_at': meeting.started_at.isoformat() if meeting.started_at else None,
                'ended_at': meeting.ended_at.isoformat() if meeting.ended_at else None
            }
        result.append(data)
    
    return jsonify({'recordings': result}), 200


@recordings_bp.route('/<recording_id>', methods=['DELETE'])
@auth_required
def delete_recording(recording_id):
    """Delete a recording (host only)"""
    recording = MeetingRecording.query.get(recording_id)
    if not recording:
        return jsonify({'error': 'Recording not found'}), 404
    
    meeting = Meeting.query.get(recording.meeting_id)
    if not meeting or meeting.host_id != request.current_user.id:
        return jsonify({'error': 'Only host can delete recording'}), 403
    
    # Delete file
    file_path = os.path.join(RECORDINGS_DIR, f'{recording_id}.webm')
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Delete DB entry
    db.session.delete(recording)
    db.session.commit()
    
    return jsonify({'message': 'Recording deleted'}), 200


@recordings_bp.route('/<recording_id>/status', methods=['PATCH'])
def update_recording_status(recording_id):
    """
    Update recording status after processing (called by AI service)
    No auth required - internal API
    """
    try:
        recording = MeetingRecording.query.get(recording_id)
        if not recording:
            return jsonify({'error': 'Recording not found'}), 404
        
        data = request.get_json()
        
        # Update allowed fields
        if 'status' in data:
            recording.status = data['status']
        if 'processing_progress' in data:
            recording.processing_progress = data['processing_progress']
        if 'error_message' in data:
            recording.error_message = data['error_message']
        if 'has_transcript' in data:
            recording.has_transcript = data['has_transcript']
        if 'speaker_count' in data:
            recording.speaker_count = data['speaker_count']
        if 'word_count' in data:
            recording.word_count = data['word_count']
        if 'language' in data:
            recording.language = data['language']
        if 'key_topics' in data:
            import json
            recording.key_topics = json.dumps(data['key_topics'])
        if 'summary_brief' in data:
            recording.summary_brief = data['summary_brief']
        if 'transcript_text' in data:
            recording.transcript_text = data['transcript_text']
        if 'summary' in data:
            recording.summary_brief = data['summary']
        if 'is_indexed' in data:
            recording.is_indexed = data['is_indexed']
        if 'indexed_at' in data:
            recording.indexed_at = datetime.fromisoformat(data['indexed_at'].replace('Z', '+00:00'))
        
        if recording.status == 'ready':
            recording.processed_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({'message': 'Status updated', 'recording': recording.to_dict()}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@recordings_bp.route('/<recording_id>/trigger-processing', methods=['POST'])
@auth_required
def trigger_processing(recording_id):
    """
    Trigger transcription processing for a recording
    Calls AI service to start the pipeline
    """
    import requests
    
    recording = MeetingRecording.query.get(recording_id)
    if not recording:
        return jsonify({'error': 'Recording not found'}), 404
    
    meeting = Meeting.query.get(recording.meeting_id)
    if not meeting:
        return jsonify({'error': 'Meeting not found'}), 404
    
    # Only host can trigger
    if meeting.host_id != request.current_user.id:
        return jsonify({'error': 'Only host can trigger processing'}), 403
    
    # Get video file path
    video_path = os.path.join(RECORDINGS_DIR, f'{recording_id}.webm')
    if not os.path.exists(video_path):
        return jsonify({'error': 'Recording file not found'}), 404
    
    # Call AI service to process
    ai_service_url = os.getenv('AI_SERVICE_URL', 'http://localhost:8001')
    
    try:
        response = requests.post(
            f'{ai_service_url}/api/meeting/process',
            json={
                'recording_id': recording_id,
                'meeting_id': meeting.id,
                'classroom_id': meeting.classroom_id,
                'video_path': video_path,
                'meeting_title': meeting.title,
                'language': 'en'
            },
            headers={'Authorization': request.headers.get('Authorization')},
            timeout=300  # 5 min timeout for processing
        )
        
        return jsonify(response.json()), response.status_code
        
    except requests.Timeout:
        return jsonify({'message': 'Processing started in background'}), 202
    except Exception as e:
        return jsonify({'error': str(e)}), 500

