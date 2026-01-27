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
        
        # Generate URL for the file - use request host for dynamic URL
        base_url = request.host_url.rstrip('/')
        file_url = f"{base_url}/api/files/{unique_filename}"
        
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


@files_bp.route('/material/<material_id>', methods=['GET'])
def get_material_file(material_id):
    """
    Serve a classroom material file by its ID.
    Looks up the material in the database and serves/redirects to the actual file.
    """
    try:
        from app.models.classroom import ClassroomMaterial
        from flask import redirect
        
        # Find the material
        material = ClassroomMaterial.query.get(material_id)
        
        if not material:
            return jsonify({'error': 'Material not found'}), 404
        
        if not material.file_url:
            return jsonify({'error': 'File URL not available'}), 404
        
        # If file_url is a local path, extract filename and serve
        file_url = material.file_url
        
        if '/api/files/' in file_url:
            # Extract filename from URL and serve directly
            filename = file_url.split('/api/files/')[-1]
            return send_from_directory(UPLOAD_DIR, filename)
        elif file_url.startswith('http'):
            # External URL - redirect to it
            return redirect(file_url)
        elif os.path.isfile(file_url):
            # Local file path - serve directory
            directory = os.path.dirname(file_url)
            filename = os.path.basename(file_url)
            return send_from_directory(directory, filename)
        else:
            # Try to find in uploads with material ID as prefix
            for f in os.listdir(UPLOAD_DIR):
                if f.startswith(material_id) or material_id in f:
                    return send_from_directory(UPLOAD_DIR, f)
            
            return jsonify({'error': 'File not found on disk'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
