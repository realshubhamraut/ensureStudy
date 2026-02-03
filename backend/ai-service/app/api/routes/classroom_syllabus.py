"""
Classroom Syllabus API Routes

Handles:
- Syllabus upload with automatic topic extraction
- Topic hierarchy retrieval (Classroom → Chapters → Topics)
- Manual topic editing by teachers
- Student topic score retrieval
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query, Header
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import tempfile
import os
import httpx

from app.services.syllabus_hierarchy_extractor import (
    get_syllabus_hierarchy_extractor,
    ExtractedSyllabusHierarchy
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/classroom-syllabus", tags=["Classroom Syllabus"])


# ============================================================================
# Request/Response Models
# ============================================================================

class TopicCreate(BaseModel):
    name: str
    description: Optional[str] = None
    difficulty: str = "medium"
    estimated_hours: float = 1.0
    key_concepts: List[str] = []


class ChapterCreate(BaseModel):
    name: str
    description: Optional[str] = None
    color: str = "#3B82F6"
    estimated_hours: float = 2.0
    topics: List[TopicCreate] = []


class SyllabusHierarchyRequest(BaseModel):
    """Request to manually set syllabus hierarchy."""
    classroom_id: str
    chapters: List[ChapterCreate]


class ExtractSyllabusRequest(BaseModel):
    """Request to extract hierarchy from text."""
    classroom_id: str
    syllabus_text: str
    subject_name: Optional[str] = None


class TopicResponse(BaseModel):
    id: str
    chapter_id: str
    name: str
    description: Optional[str]
    difficulty: str
    estimated_hours: float
    key_concepts: List[str]
    order: int
    question_count: int = 0


class ChapterResponse(BaseModel):
    id: str
    classroom_id: str
    name: str
    description: Optional[str]
    color: str
    estimated_hours: float
    order: int
    topic_count: int
    topics: List[TopicResponse] = []


class HierarchyResponse(BaseModel):
    classroom_id: str
    subject_name: str
    chapters: List[ChapterResponse]
    total_chapters: int
    total_topics: int


class StudentTopicScoreResponse(BaseModel):
    topic_id: str
    topic_name: str
    chapter_name: str
    chapter_color: str
    mastery_percentage: float
    mcq_attempts: int
    mcq_correct: int
    descriptive_attempts: int
    status: str


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/extract", response_model=Dict[str, Any])
async def extract_syllabus_hierarchy(
    classroom_id: str = Form(...),
    subject_name: str = Form(""),
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None)
):
    """
    Extract chapter-topic hierarchy from syllabus.
    
    Accepts either:
    - PDF file upload
    - Plain text (already extracted)
    
    Returns extracted hierarchy for review before saving.
    Also saves the PDF to storage and updates classroom's syllabus URL.
    """
    extractor = get_syllabus_hierarchy_extractor()
    core_api_url = os.getenv("CORE_SERVICE_URL", os.getenv("CORE_API_URL", "http://localhost:8000"))
    
    syllabus_url = None
    syllabus_filename = None
    
    if file:
        # Handle PDF upload
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(400, "Only PDF files are supported")
        
        # Read file content
        content = await file.read()
        syllabus_filename = file.filename
        
        # Save to temp file for extraction
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Extract hierarchy
            hierarchy = extractor.extract_hierarchy(tmp_path, subject_name)
            
            # Upload PDF to core service file storage
            if authorization:
                try:
                    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
                        # Reset file position for upload
                        files = {"file": (file.filename, content, "application/pdf")}
                        headers = {"Authorization": authorization}
                        
                        upload_resp = await client.post(
                            f"{core_api_url}/api/files/upload",
                            files=files,
                            headers=headers
                        )
                        
                        if upload_resp.status_code in (200, 201):
                            upload_data = upload_resp.json()
                            syllabus_url = upload_data.get("url")
                            logger.info(f"Uploaded syllabus PDF: {syllabus_url}")
                            
                            # Update classroom with syllabus URL
                            syllabus_update = {
                                "syllabus_url": syllabus_url,
                                "syllabus_filename": syllabus_filename
                            }
                            
                            update_resp = await client.put(
                                f"{core_api_url}/api/classroom/{classroom_id}/syllabus",
                                json=syllabus_update,
                                headers=headers
                            )
                            
                            if update_resp.status_code == 200:
                                logger.info(f"Updated classroom {classroom_id} syllabus URL")
                            else:
                                logger.warning(f"Failed to update classroom syllabus: {update_resp.text}")
                        else:
                            logger.warning(f"Failed to upload syllabus PDF: {upload_resp.text}")
                            
                except Exception as e:
                    logger.error(f"Error uploading syllabus: {e}")
        finally:
            os.unlink(tmp_path)
    
    elif text:
        # Handle plain text
        hierarchy = extractor.extract_from_text(text, subject_name)
    
    else:
        raise HTTPException(400, "Either file or text must be provided")
    
    result = extractor.hierarchy_to_dict(hierarchy)
    result["classroom_id"] = classroom_id
    result["syllabus_url"] = syllabus_url
    result["syllabus_filename"] = syllabus_filename
    
    logger.info(f"Extracted {result['total_chapters']} chapters, {result['total_topics']} topics")
    
    return result



@router.post("/save", response_model=Dict[str, Any])
async def save_syllabus_hierarchy(
    request: SyllabusHierarchyRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Save extracted/edited hierarchy to database.
    
    Creates Chapter and ClassroomTopic records.
    Returns created IDs for confirmation.
    """
    # This will call the Core API to create records
    core_api_url = os.getenv("CORE_SERVICE_URL", os.getenv("CORE_API_URL", "http://localhost:8000"))
    
    chapters_created = []
    topics_created = []
    
    # Prepare headers with auth token
    headers = {}
    if authorization:
        headers["Authorization"] = authorization
    
    async with httpx.AsyncClient(verify=False) as client:
        for idx, chapter_data in enumerate(request.chapters):
            # Create chapter
            chapter_payload = {
                "classroom_id": request.classroom_id,
                "name": chapter_data.name,
                "description": chapter_data.description,
                "color": chapter_data.color,
                "estimated_hours": chapter_data.estimated_hours,
                "order": idx
            }
            
            # POST to Core API
            try:
                resp = await client.post(
                    f"{core_api_url}/api/classroom/{request.classroom_id}/chapters",
                    json=chapter_payload,
                    headers=headers,
                    timeout=30.0
                )
                
                if resp.status_code == 200 or resp.status_code == 201:
                    chapter_result = resp.json()
                    chapter_id = chapter_result.get("id")
                    chapters_created.append(chapter_result)
                    
                    # Create topics for this chapter
                    for t_idx, topic_data in enumerate(chapter_data.topics):
                        topic_payload = {
                            "chapter_id": chapter_id,
                            "classroom_id": request.classroom_id,
                            "name": topic_data.name,
                            "description": topic_data.description,
                            "difficulty": topic_data.difficulty,
                            "estimated_hours": topic_data.estimated_hours,
                            "key_concepts": topic_data.key_concepts,
                            "order": t_idx
                        }
                        
                        topic_resp = await client.post(
                            f"{core_api_url}/api/classroom/{request.classroom_id}/topics",
                            json=topic_payload,
                            headers=headers,
                            timeout=30.0
                        )
                        
                        if topic_resp.status_code in (200, 201):
                            topics_created.append(topic_resp.json())
                        else:
                            logger.warning(f"Failed to create topic: {topic_resp.text}")
                else:
                    logger.warning(f"Failed to create chapter: {resp.text}")
                    
            except Exception as e:
                logger.error(f"Error creating chapter: {e}")
                continue
    
    return {
        "success": True,
        "classroom_id": request.classroom_id,
        "chapters_created": len(chapters_created),
        "topics_created": len(topics_created),
        "chapters": chapters_created
    }


@router.get("/hierarchy/{classroom_id}", response_model=Dict[str, Any])
async def get_classroom_hierarchy(
    classroom_id: str,
    include_scores: bool = Query(False, description="Include student scores"),
    user_id: Optional[str] = Query(None, description="Student user ID for scores")
):
    """
    Get classroom's chapter-topic hierarchy.
    
    If include_scores=True and user_id provided, includes student's mastery scores.
    """
    core_api_url = os.getenv("CORE_SERVICE_URL", os.getenv("CORE_API_URL", "http://localhost:8000"))
    
    async with httpx.AsyncClient(verify=False) as client:
        # Get chapters and topics from Core API
        try:
            resp = await client.get(
                f"{core_api_url}/api/classroom/{classroom_id}/hierarchy",
                timeout=30.0
            )
            
            if resp.status_code != 200:
                raise HTTPException(resp.status_code, "Failed to fetch hierarchy")
            
            hierarchy = resp.json()
            
            # If scores requested, fetch them
            if include_scores and user_id:
                scores_resp = await client.get(
                    f"{core_api_url}/api/classroom/users/{user_id}/topic-scores",
                    params={"classroom_id": classroom_id},
                    timeout=30.0
                )
                
                if scores_resp.status_code == 200:
                    scores = {s["topic_id"]: s for s in scores_resp.json()}
                    
                    # Merge scores into hierarchy
                    for chapter in hierarchy.get("chapters", []):
                        for topic in chapter.get("topics", []):
                            topic_score = scores.get(topic["id"], {})
                            topic["mastery_percentage"] = topic_score.get("mastery_percentage", 0)
                            topic["status"] = topic_score.get("status", "not_started")
            
            return hierarchy
            
        except httpx.RequestError as e:
            logger.error(f"Error fetching hierarchy: {e}")
            raise HTTPException(500, f"Failed to fetch hierarchy: {str(e)}")


@router.get("/student-scores/{user_id}", response_model=List[StudentTopicScoreResponse])
async def get_student_topic_scores(
    user_id: str,
    classroom_id: Optional[str] = Query(None, description="Filter by classroom")
):
    """
    Get all topic scores for a student.
    
    Useful for curriculum page to show mastery across all enrolled classrooms.
    """
    core_api_url = os.getenv("CORE_SERVICE_URL", os.getenv("CORE_API_URL", "http://localhost:8000"))
    
    async with httpx.AsyncClient(verify=False) as client:
        try:
            params = {}
            if classroom_id:
                params["classroom_id"] = classroom_id
            
            resp = await client.get(
                f"{core_api_url}/api/classroom/users/{user_id}/topic-scores",
                params=params,
                timeout=30.0
            )
            
            if resp.status_code != 200:
                raise HTTPException(resp.status_code, "Failed to fetch scores")
            
            return resp.json()
            
        except httpx.RequestError as e:
            logger.error(f"Error fetching scores: {e}")
            raise HTTPException(500, f"Failed to fetch scores: {str(e)}")


@router.post("/sync-student/{user_id}", response_model=Dict[str, Any])
async def sync_student_topics(
    user_id: str,
    classroom_id: str = Query(..., description="Classroom to sync from")
):
    """
    Sync classroom topics to student's curriculum.
    
    Called when:
    - Student enrolls in a classroom
    - Teacher updates classroom syllabus
    """
    core_api_url = os.getenv("CORE_SERVICE_URL", os.getenv("CORE_API_URL", "http://localhost:8000"))
    
    async with httpx.AsyncClient(verify=False) as client:
        try:
            resp = await client.post(
                f"{core_api_url}/api/classroom/users/{user_id}/sync-topics",
                json={"classroom_id": classroom_id},
                timeout=30.0
            )
            
            if resp.status_code not in (200, 201):
                raise HTTPException(resp.status_code, "Failed to sync topics")
            
            return resp.json()
            
        except httpx.RequestError as e:
            logger.error(f"Error syncing topics: {e}")
            raise HTTPException(500, f"Failed to sync topics: {str(e)}")
