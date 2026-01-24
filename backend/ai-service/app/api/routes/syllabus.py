"""
Syllabus Processing API Routes

Endpoints for:
- Processing syllabus PDFs
- Extracting topics
- Searching syllabus content
"""
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import os
import tempfile
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/syllabus", tags=["Syllabus"])


class ProcessSyllabusRequest(BaseModel):
    """Request body for syllabus processing"""
    syllabus_id: str
    file_url: str  # URL to download PDF from
    classroom_id: str
    subject_name: str
    title: Optional[str] = None


class ProcessSyllabusResponse(BaseModel):
    """Response from syllabus processing"""
    success: bool
    syllabus_id: str
    chunks_stored: int
    topics_extracted: int
    lessons_created: int
    processing_time_ms: int
    error: Optional[str] = None


class SearchSyllabusRequest(BaseModel):
    """Request for syllabus content search"""
    query: str
    classroom_id: Optional[str] = None
    subject: Optional[str] = None
    top_k: int = 5


class SearchResult(BaseModel):
    """Single search result"""
    id: str
    score: float
    text: str
    chapter: str
    subject: str
    syllabus_id: str
    page_number: int


class SearchSyllabusResponse(BaseModel):
    """Response from syllabus search"""
    results: List[SearchResult]
    count: int


@router.post("/process", response_model=ProcessSyllabusResponse)
async def process_syllabus(request: ProcessSyllabusRequest, background_tasks: BackgroundTasks):
    """
    Process a syllabus PDF.
    
    Pipeline:
    1. Download PDF from file_url
    2. Extract text and detect chapters
    3. Store chunks in 'syllabus_content' Qdrant collection
    4. Extract topics using LLM
    5. Populate curriculum models (Subject, Topic, Subtopic)
    
    Returns processing results.
    """
    try:
        from ...services.syllabus_extractor import get_syllabus_extractor
        
        print(f"\n{'='*70}")
        print(f"[SYLLABUS API] 📚 PROCESSING SYLLABUS")
        print(f"{'='*70}")
        print(f"[SYLLABUS API] Syllabus ID: {request.syllabus_id}")
        print(f"[SYLLABUS API] Classroom ID: {request.classroom_id}")
        print(f"[SYLLABUS API] Subject: {request.subject_name}")
        print(f"[SYLLABUS API] File URL: {request.file_url}")
        print(f"{'='*70}")
        
        extractor = get_syllabus_extractor()
        
        # Download PDF to temp file
        print(f"[SYLLABUS API] Step 1: Downloading PDF...")
        pdf_path = await _download_pdf(request.file_url)
        
        if not pdf_path:
            print(f"[SYLLABUS API] ❌ FAILED: Could not download PDF")
            raise HTTPException(status_code=400, detail="Failed to download PDF from URL")
        
        print(f"[SYLLABUS API] ✓ PDF downloaded to: {pdf_path}")
        
        try:
            # Process the syllabus
            print(f"[SYLLABUS API] Step 2: Processing syllabus...")
            result = await extractor.process_syllabus(
                syllabus_id=request.syllabus_id,
                pdf_path=pdf_path,
                classroom_id=request.classroom_id,
                subject_name=request.subject_name,
                title=request.title
            )
            
            print(f"\n{'='*70}")
            if result.success:
                print(f"[SYLLABUS API] ✅ PROCESSING COMPLETE")
                print(f"[SYLLABUS API]   Chunks stored: {result.chunks_stored}")
                print(f"[SYLLABUS API]   Topics extracted: {result.topics_extracted}")
                print(f"[SYLLABUS API]   Topics created in DB: {result.lessons_created}")
            else:
                print(f"[SYLLABUS API] ❌ PROCESSING FAILED: {result.error}")
            print(f"{'='*70}\n")
            
            return ProcessSyllabusResponse(
                success=result.success,
                syllabus_id=result.syllabus_id,
                chunks_stored=result.chunks_stored,
                topics_extracted=result.topics_extracted,
                lessons_created=result.lessons_created,
                processing_time_ms=result.processing_time_ms,
                error=result.error
            )
            
        finally:
            # Clean up temp file
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                print(f"[SYLLABUS API] Cleaned up temp file")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SYLLABUS API] Processing error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class ProcessFromClassroomRequest(BaseModel):
    """Request to process syllabus from classroom data"""
    classroom_id: str
    subject_name: Optional[str] = None  # If not provided, use classroom.subject


@router.post("/process-from-classroom", response_model=ProcessSyllabusResponse)
async def process_syllabus_from_classroom(request: ProcessFromClassroomRequest):
    """
    Process syllabus from classroom data.
    
    Fetches syllabus_url from the classroom and processes it.
    This is the endpoint to call after a teacher uploads a syllabus to a classroom.
    """
    import httpx
    from uuid import uuid4
    
    core_service_url = os.getenv("CORE_SERVICE_URL", "http://localhost:9000")
    
    print(f"\n{'='*70}")
    print(f"[SYLLABUS API] 📚 PROCESSING FROM CLASSROOM: {request.classroom_id}")
    print(f"{'='*70}")
    
    try:
        # Fetch classroom data from core service (internal service call)
        async with httpx.AsyncClient(timeout=30.0) as client:
            print(f"[SYLLABUS API] Fetching classroom data from {core_service_url}...")
            resp = await client.get(
                f"{core_service_url}/api/classroom/{request.classroom_id}/syllabus",
                headers={"X-Service-Key": "internal-ai-service"}  # Internal service auth
            )
            
            if resp.status_code != 200:
                print(f"[SYLLABUS API] ❌ Failed to fetch classroom: {resp.status_code} - {resp.text}")
                raise HTTPException(status_code=404, detail=f"Classroom or syllabus not found: {resp.text}")
            
            classroom_data = resp.json()
            print(f"[SYLLABUS API] ✓ Classroom data retrieved: {classroom_data.get('classroom_name')}")
        
        syllabus_url = classroom_data.get("syllabus_url")
        if not syllabus_url:
            print(f"[SYLLABUS API] ❌ No syllabus URL found in classroom")
            raise HTTPException(status_code=400, detail="No syllabus uploaded for this classroom")
        
        # Determine subject name (prefer explicit subject, then classroom.subject, then classroom_name)
        subject_name = (
            request.subject_name or 
            classroom_data.get("subject") or 
            classroom_data.get("classroom_name") or 
            "General"
        )
        print(f"[SYLLABUS API] Subject determined: {subject_name}")
        
        # Generate syllabus ID
        syllabus_id = str(uuid4())
        
        print(f"[SYLLABUS API] Syllabus URL: {syllabus_url}")
        print(f"[SYLLABUS API] Subject: {subject_name}")
        
        # Process using existing logic
        from ...services.syllabus_extractor import get_syllabus_extractor
        
        extractor = get_syllabus_extractor()
        
        # Download PDF
        pdf_path = await _download_pdf(syllabus_url)
        
        if not pdf_path:
            raise HTTPException(status_code=400, detail="Failed to download syllabus PDF")
        
        try:
            result = await extractor.process_syllabus(
                syllabus_id=syllabus_id,
                pdf_path=pdf_path,
                classroom_id=request.classroom_id,
                subject_name=subject_name,
                title=classroom_data.get("syllabus_filename", "Syllabus")
            )
            
            print(f"\n{'='*70}")
            if result.success:
                print(f"[SYLLABUS API] ✅ CLASSROOM SYLLABUS PROCESSED")
                print(f"[SYLLABUS API]   Chunks: {result.chunks_stored}")
                print(f"[SYLLABUS API]   Topics: {result.topics_extracted}")
                print(f"[SYLLABUS API]   DB Records: {result.lessons_created}")
            else:
                print(f"[SYLLABUS API] ❌ FAILED: {result.error}")
            print(f"{'='*70}\n")
            
            return ProcessSyllabusResponse(
                success=result.success,
                syllabus_id=result.syllabus_id,
                chunks_stored=result.chunks_stored,
                topics_extracted=result.topics_extracted,
                lessons_created=result.lessons_created,
                processing_time_ms=result.processing_time_ms,
                error=result.error
            )
            
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SYLLABUS API] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-upload", response_model=ProcessSyllabusResponse)
async def process_syllabus_upload(
    file: UploadFile = File(...),
    syllabus_id: str = Form(...),
    classroom_id: str = Form(...),
    subject_name: str = Form(...),
    title: str = Form(None)
):
    """
    Process an uploaded syllabus PDF directly.
    
    Use this endpoint for direct file uploads (not URLs).
    """
    try:
        from ...services.syllabus_extractor import get_syllabus_extractor
        
        print(f"\n[SYLLABUS API] Processing uploaded file: {file.filename}")
        print(f"[SYLLABUS API] Syllabus ID: {syllabus_id}")
        print(f"[SYLLABUS API] Subject: {subject_name}")
        
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            pdf_path = tmp.name
        
        try:
            extractor = get_syllabus_extractor()
            
            result = await extractor.process_syllabus(
                syllabus_id=syllabus_id,
                pdf_path=pdf_path,
                classroom_id=classroom_id,
                subject_name=subject_name,
                title=title
            )
            
            return ProcessSyllabusResponse(
                success=result.success,
                syllabus_id=result.syllabus_id,
                chunks_stored=result.chunks_stored,
                topics_extracted=result.topics_extracted,
                lessons_created=result.lessons_created,
                processing_time_ms=result.processing_time_ms,
                error=result.error
            )
            
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                
    except Exception as e:
        logger.error(f"[SYLLABUS API] Upload processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchSyllabusResponse)
async def search_syllabus(request: SearchSyllabusRequest):
    """
    Search syllabus content in vector database.
    
    Searches the 'syllabus_content' Qdrant collection.
    Optionally filter by classroom_id or subject.
    """
    try:
        from ...services.syllabus_extractor import get_syllabus_extractor
        
        extractor = get_syllabus_extractor()
        
        results = extractor.search_syllabus_content(
            query=request.query,
            classroom_id=request.classroom_id,
            subject=request.subject,
            top_k=request.top_k
        )
        
        return SearchSyllabusResponse(
            results=[SearchResult(**r) for r in results],
            count=len(results)
        )
        
    except Exception as e:
        logger.error(f"[SYLLABUS API] Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collection-info")
async def get_collection_info():
    """Get information about the syllabus Qdrant collection."""
    try:
        from ...services.syllabus_extractor import SYLLABUS_COLLECTION, get_syllabus_extractor
        from qdrant_client import QdrantClient
        
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(c.name == SYLLABUS_COLLECTION for c in collections)
        
        if not exists:
            return {
                "collection_name": SYLLABUS_COLLECTION,
                "exists": False,
                "points_count": 0,
                "message": "Collection not created yet. Process a syllabus to create it."
            }
        
        # Get collection info
        info = client.get_collection(SYLLABUS_COLLECTION)
        
        return {
            "collection_name": SYLLABUS_COLLECTION,
            "exists": True,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "vector_size": info.config.params.vectors.size if info.config.params.vectors else 384,
            "status": info.status
        }
        
    except Exception as e:
        logger.error(f"[SYLLABUS API] Collection info error: {e}")
        return {
            "collection_name": SYLLABUS_COLLECTION,
            "exists": False,
            "error": str(e)
        }


async def _download_pdf(url: str) -> Optional[str]:
    """Download PDF from URL to temp file."""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to download PDF: HTTP {response.status}")
                    return None
                
                content = await response.read()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(content)
                    return tmp.name
                    
    except Exception as e:
        logger.error(f"Download error: {e}")
        return None
