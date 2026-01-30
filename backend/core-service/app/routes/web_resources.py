"""
Web Resources Routes - Store and retrieve web-downloaded materials
"""
from flask import Blueprint, request, jsonify
from app import db
from app.models.classroom import ClassroomMaterial
from app.models.user import User

web_resources_bp = Blueprint("web_resources", __name__, url_prefix="/api/web-resources")


def internal_service_check():
    """Check if request is from internal AI service"""
    service_key = request.headers.get("X-Service-Key")
    return service_key == "internal-ai-service"


@web_resources_bp.route("/register", methods=["POST"])
def register_web_material():
    """
    Register a web-downloaded PDF as a material.
    Called by AI service after downloading PDFs.
    """
    if not internal_service_check():
        return jsonify({"error": "Unauthorized - internal service only"}), 403
    
    data = request.get_json()
    
    required = ["name", "file_path", "user_id", "source"]
    for field in required:
        if not data.get(field):
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Verify user exists
    user = User.query.get(data["user_id"])
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    # Get optional classroom_id (if provided, store in that classroom)
    classroom_id = data.get("classroom_id")
    if classroom_id:
        from app.models.classroom import Classroom
        classroom = Classroom.query.get(classroom_id)
        if not classroom:
            print(f"[WEB-RES] Warning: Classroom {classroom_id} not found, storing as global")
            classroom_id = None
    
    # Create material record with source='web'
    material = ClassroomMaterial(
        classroom_id=classroom_id,  # Store in classroom if provided
        name=data["name"],
        file_url=data["file_path"],  # Local file path
        file_type="application/pdf",
        file_size=data.get("file_size", 0),
        subject=data.get("topic") or data.get("subject") or "Web Resource",
        description=f"Downloaded from web: {data.get('source_url', '')}",
        uploaded_by=data["user_id"],
        source="web",
        source_url=data.get("source_url", "")
    )
    
    db.session.add(material)
    db.session.commit()
    
    print(f"[WEB-RES] ✅ Registered: {material.name} (id={material.id})")
    
    # Trigger indexing via AI service
    try:
        import requests as http_requests
        import os
        ai_service_url = os.getenv('AI_SERVICE_URL', 'http://localhost:9001')
        http_requests.post(
            f'{ai_service_url}/api/index/material',
            json={
                'material_id': material.id,
                'file_url': material.file_url,
                'classroom_id': classroom_id,  # Pass actual classroom_id
                'subject': material.subject,
                'document_title': material.name,
                'uploaded_by': data["user_id"]
            },
            timeout=5
        )
        print(f"[WEB-RES] Triggered indexing for material {material.id}")
    except Exception as e:
        print(f"[WEB-RES] Warning: Failed to trigger indexing: {e}")
    
    return jsonify({
        "success": True,
        "material": material.to_dict(),
        "message": "Web material registered successfully"
    }), 201


@web_resources_bp.route("/user/<user_id>", methods=["GET"])
def list_user_web_resources(user_id):
    """
    List all web resources downloaded by a user.
    """
    # Allow internal service or authenticated user
    if not internal_service_check():
        # TODO: Add user authentication check
        pass
    
    materials = ClassroomMaterial.query.filter_by(
        uploaded_by=user_id,
        source="web",
        is_active=True
    ).order_by(ClassroomMaterial.uploaded_at.desc()).all()
    
    return jsonify({
        "resources": [m.to_dict() for m in materials],
        "count": len(materials)
    })


@web_resources_bp.route("/<material_id>", methods=["DELETE"])
def delete_web_resource(material_id):
    """
    Delete a web resource.
    """
    if not internal_service_check():
        return jsonify({"error": "Unauthorized"}), 403
    
    material = ClassroomMaterial.query.get(material_id)
    
    if not material:
        return jsonify({"error": "Resource not found"}), 404
    
    if material.source != "web":
        return jsonify({"error": "Not a web resource"}), 400
    
    # Soft delete
    material.is_active = False
    db.session.commit()
    
    return jsonify({
        "success": True,
        "message": "Web resource deleted"
    })
