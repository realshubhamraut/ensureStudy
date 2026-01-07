"""
Notification routes for managing user notifications
"""
from flask import Blueprint, request, jsonify
from datetime import datetime
from app import db
from app.routes.users import require_auth
from app.models.notification import Notification

notifications_bp = Blueprint("notifications", __name__, url_prefix="/api")


@notifications_bp.route("/notifications", methods=["GET"])
@require_auth
def get_notifications():
    """Get user's notifications with pagination"""
    user_id = request.user_id
    
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    unread_only = request.args.get("unread_only", "false").lower() == "true"
    type_filter = request.args.get("type")  # Optional filter by type
    
    query = Notification.query.filter_by(user_id=user_id)
    
    if unread_only:
        query = query.filter_by(is_read=False)
    
    if type_filter:
        query = query.filter_by(type=type_filter)
    
    # Order by newest first
    query = query.order_by(Notification.created_at.desc())
    
    # Paginate
    paginated = query.paginate(page=page, per_page=per_page, error_out=False)
    
    # Get unread count
    unread_count = Notification.query.filter_by(user_id=user_id, is_read=False).count()
    
    return jsonify({
        "notifications": [n.to_dict() for n in paginated.items],
        "page": page,
        "per_page": per_page,
        "total": paginated.total,
        "pages": paginated.pages,
        "unread_count": unread_count
    }), 200


@notifications_bp.route("/notifications/recent", methods=["GET"])
@require_auth
def get_recent_notifications():
    """Get recent notifications for dashboard widget (max 10)"""
    user_id = request.user_id
    limit = request.args.get("limit", 10, type=int)
    
    notifications = Notification.query.filter_by(user_id=user_id) \
        .order_by(Notification.created_at.desc()) \
        .limit(min(limit, 20)) \
        .all()
    
    unread_count = Notification.query.filter_by(user_id=user_id, is_read=False).count()
    
    return jsonify({
        "notifications": [n.to_dict() for n in notifications],
        "unread_count": unread_count
    }), 200


@notifications_bp.route("/notifications/<notification_id>/read", methods=["POST"])
@require_auth
def mark_as_read(notification_id):
    """Mark a specific notification as read"""
    user_id = request.user_id
    
    notification = Notification.query.filter_by(id=notification_id, user_id=user_id).first()
    if not notification:
        return jsonify({"error": "Notification not found"}), 404
    
    notification.is_read = True
    notification.read_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify(notification.to_dict()), 200


@notifications_bp.route("/notifications/read-all", methods=["POST"])
@require_auth
def mark_all_as_read():
    """Mark all notifications as read"""
    user_id = request.user_id
    
    now = datetime.utcnow()
    Notification.query.filter_by(user_id=user_id, is_read=False).update({
        "is_read": True,
        "read_at": now
    })
    db.session.commit()
    
    return jsonify({"message": "All notifications marked as read"}), 200


@notifications_bp.route("/notifications/<notification_id>", methods=["DELETE"])
@require_auth
def delete_notification(notification_id):
    """Delete a notification"""
    user_id = request.user_id
    
    notification = Notification.query.filter_by(id=notification_id, user_id=user_id).first()
    if not notification:
        return jsonify({"error": "Notification not found"}), 404
    
    db.session.delete(notification)
    db.session.commit()
    
    return jsonify({"message": "Notification deleted"}), 200


@notifications_bp.route("/notifications/unread-count", methods=["GET"])
@require_auth
def get_unread_count():
    """Get unread notification count"""
    user_id = request.user_id
    count = Notification.query.filter_by(user_id=user_id, is_read=False).count()
    return jsonify({"unread_count": count}), 200


@notifications_bp.route("/notifications/stream", methods=["GET"])
def notification_stream():
    """SSE endpoint for real-time notification push"""
    from flask import Response, stream_with_context
    import time
    import json
    
    # Get auth token from query param (SSE can't use headers easily)
    token = request.args.get("token")
    if not token:
        return jsonify({"error": "Token required"}), 401
    
    try:
        from app.utils.jwt_handler import verify_token
        payload = verify_token(token)
        user_id = payload["user_id"]
    except:
        return jsonify({"error": "Invalid token"}), 401
    
    def generate():
        last_check = datetime.utcnow()
        
        while True:
            try:
                # Check for new notifications since last check
                new_notifications = Notification.query.filter(
                    Notification.user_id == user_id,
                    Notification.created_at > last_check
                ).order_by(Notification.created_at.asc()).all()
                
                if new_notifications:
                    for notification in new_notifications:
                        data = json.dumps(notification.to_dict())
                        yield f"data: {data}\n\n"
                    last_check = new_notifications[-1].created_at
                else:
                    # Send heartbeat to keep connection alive
                    yield f": heartbeat\n\n"
                
                # Poll every 2 seconds
                time.sleep(2)
                
            except GeneratorExit:
                break
            except Exception as e:
                print(f"SSE error: {e}")
                break
    
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
