"""
File Upload Routes
Handles generic file uploads and serving
"""
import os
import uuid
from flask import Blueprint, request, jsonify, send_from_directory
from functools import wraps
from app.models.user import User
from app.utils.jwt_handler import verify_token

files_bp = Blueprint('files', __name__, url_prefix='/api/files')

# Upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'uploads')

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)


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


@files_bp.route('/upload', methods=['POST'])
@auth_required
def upload_file():
    """Upload a file (PDF, image, etc.)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate unique filename
        original_name = file.filename
        extension = os.path.splitext(original_name)[1].lower()
        unique_filename = f"{uuid.uuid4()}{extension}"
        
        # Allowed extensions
        allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.gif', '.doc', '.docx'}
        if extension not in allowed_extensions:
            return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(allowed_extensions)}'}), 400
        
        # Save file
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        file.save(file_path)
        
        # Generate URL for the file
        file_url = f"http://localhost:8000/api/files/{unique_filename}"
        
        return jsonify({
            'message': 'File uploaded successfully',
            'url': file_url,
            'filename': original_name,
            'stored_filename': unique_filename,
            'size': os.path.getsize(file_path)
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@files_bp.route('/<filename>', methods=['GET'])
def get_file(filename):
    """Serve an uploaded file"""
    try:
        return send_from_directory(UPLOAD_DIR, filename)
    except Exception as e:
        return jsonify({'error': 'File not found'}), 404
