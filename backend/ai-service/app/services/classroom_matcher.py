"""
Classroom Matcher Service

Matches detected subject to user's classrooms using fuzzy matching.
E.g., "Physics" detected → matches "Physics Class 10-A"
"""
import os
import logging
import httpx
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Support both env var names, default to https://localhost:8000
CORE_API_URL = os.getenv("CORE_API_URL") or os.getenv("CORE_SERVICE_URL") or "https://localhost:8000"


async def match_classroom_by_subject(
    user_id: str,
    detected_subject: str
) -> Optional[Dict[str, Any]]:
    """
    Find user's classroom matching the detected subject.
    
    Uses fuzzy matching on classroom.name and classroom.subject fields.
    E.g., detected_subject="Physics" matches classroom "Physics Class 10-A"
    
    Args:
        user_id: The user's ID
        detected_subject: Subject string (e.g., "physics", "chemistry")
        
    Returns:
        Dict with classroom_id and name, or None if no match
    """
    if not detected_subject:
        return None
    
    try:
        # verify=False for self-signed certs in development
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            # Use internal endpoint for AI service
            response = await client.get(
                f"{CORE_API_URL}/api/classroom/internal/user/{user_id}/classrooms",
                headers={"X-Service-Key": "internal-ai-service"}
            )
            
            if response.status_code != 200:
                logger.warning(f"[CLASSROOM-MATCH] Could not fetch classrooms: {response.status_code}")
                return None
            
            data = response.json()
            classrooms = data.get("classrooms", [])
            
            if isinstance(classrooms, dict):
                classrooms = [classrooms]
            
            if not classrooms:
                logger.info(f"[CLASSROOM-MATCH] User {user_id} has no classrooms")
                return None
            
            # Fuzzy match by name or subject
            subject_lower = detected_subject.lower().strip()
            
            for classroom in classrooms:
                classroom_name = (classroom.get("name") or "").lower()
                classroom_subject = (classroom.get("subject") or "").lower()
                
                # Match if subject is in name or subject field
                if subject_lower in classroom_name or subject_lower in classroom_subject:
                    match = {
                        "classroom_id": classroom.get("id"),
                        "name": classroom.get("name"),
                        "subject": classroom.get("subject")
                    }
                    logger.info(f"[CLASSROOM-MATCH] ✅ Matched '{detected_subject}' → '{match['name']}' (ID: {match['classroom_id'][:8]}...)")
                    return match
            
            logger.info(f"[CLASSROOM-MATCH] No classroom matched for subject '{detected_subject}'")
            return None
            
    except Exception as e:
        logger.error(f"[CLASSROOM-MATCH] Error matching classroom: {e}")
        return None


async def store_web_material_in_classroom(
    classroom_id: str,
    material_name: str,
    file_url: str,
    file_type: str = "application/pdf",
    source_url: str = None,
    subject: str = None,
    description: str = None
) -> Optional[str]:
    """
    Store a web-sourced PDF/document in a classroom's materials.
    
    Args:
        classroom_id: Target classroom ID
        material_name: Name/title of the material
        file_url: URL where the file is stored (S3, etc.)
        file_type: MIME type (default: application/pdf)
        source_url: Original web URL where the PDF was found
        subject: Subject category
        description: Optional description
        
    Returns:
        Material ID if successful, None otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            payload = {
                "classroom_id": classroom_id,
                "name": material_name,
                "file_url": file_url,
                "file_type": file_type,
                "source": "web",  # Mark as web-sourced
                "source_url": source_url,
                "uploaded_by_role": "system",  # System/web crawler
                "visibility": "public",  # Visible to all students
                "subject": subject,
                "description": description or f"Web resource: {source_url or file_url}"
            }
            
            response = await client.post(
                f"{CORE_API_URL}/api/materials/web",
                json=payload
            )
            
            if response.status_code in (200, 201):
                result = response.json()
                material_id = result.get("id") or result.get("material_id")
                logger.info(f"[WEB-MATERIAL] ✅ Stored web material '{material_name}' in classroom {classroom_id[:8]}...")
                return material_id
            else:
                logger.warning(f"[WEB-MATERIAL] Failed to store material: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"[WEB-MATERIAL] Error storing web material: {e}")
        return None
