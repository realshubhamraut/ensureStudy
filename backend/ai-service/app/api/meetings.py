"""
Meeting Transcription and Query API
- Transcribe audio recordings using Whisper (OpenAI API)
- Generate summaries using LLM (Gemini)
- Answer questions from meeting content using RAG (Qdrant)
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
import json
import httpx
from datetime import datetime

# Try to import Gemini (optional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    print("⚠️ google-generativeai not installed, install with: pip install google-generativeai")

# Initialize router
router = APIRouter(prefix="/api/meetings", tags=["meetings"])

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/ensure_study_meetings")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Initialize Gemini
gemini_model = None
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# MongoDB client (lazy init)
mongo_client = None

def get_mongo_db():
    global mongo_client
    if mongo_client is None:
        try:
            from pymongo import MongoClient
            mongo_client = MongoClient(MONGODB_URI)
        except Exception as e:
            print(f"MongoDB not available: {e}")
            return None
    try:
        return mongo_client.get_database("ensure_study_meetings")
    except:
        return None


# ============ Request/Response Schemas ============

class TranscribeRequest(BaseModel):
    meeting_id: str
    recording_url: str
    duration_seconds: Optional[int] = None


class TranscribeResponse(BaseModel):
    meeting_id: str
    transcript: str
    segments: List[dict]
    language: str
    word_count: int
    processing_time_seconds: float


class SummarizeRequest(BaseModel):
    meeting_id: str
    transcript: str


class SummarizeResponse(BaseModel):
    meeting_id: str
    brief: str
    detailed: str
    key_points: List[str]
    topics_discussed: List[str]
    action_items: List[str]


class QueryRequest(BaseModel):
    meeting_id: Optional[str] = None
    classroom_id: Optional[str] = None
    query: str
    max_results: int = 5


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[dict]
    confidence: float


# ============ Endpoints ============

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_recording(request: TranscribeRequest):
    """
    Transcribe a meeting recording using OpenAI Whisper API
    """
    start_time = datetime.utcnow()
    
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503, 
            detail="OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
        )
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Download audio from storage URL
            audio_response = await client.get(request.recording_url)
            if audio_response.status_code != 200:
                raise HTTPException(status_code=404, detail="Could not download recording from storage URL")
            
            audio_data = audio_response.content
            
            # Call OpenAI Whisper API
            files = {
                "file": ("recording.webm", audio_data, "audio/webm"),
                "model": (None, "whisper-1"),
                "response_format": (None, "verbose_json"),
                "timestamp_granularities[]": (None, "segment")
            }
            
            whisper_response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files=files,
                timeout=300.0
            )
            
            if whisper_response.status_code != 200:
                raise HTTPException(
                    status_code=502, 
                    detail=f"Whisper API error: {whisper_response.text}"
                )
            
            result = whisper_response.json()
            transcript = result.get("text", "")
            segments = []
            
            for seg in result.get("segments", []):
                segments.append({
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", ""),
                    "confidence": seg.get("avg_logprob", 0)
                })
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Store in MongoDB
            db = get_mongo_db()
            if db:
                db.transcripts.update_one(
                    {"meeting_id": request.meeting_id},
                    {"$set": {
                        "transcript": transcript,
                        "segments": segments,
                        "language": result.get("language", "en"),
                        "created_at": datetime.utcnow(),
                        "processing_time": processing_time
                    }},
                    upsert=True
                )
            
            return TranscribeResponse(
                meeting_id=request.meeting_id,
                transcript=transcript,
                segments=segments,
                language=result.get("language", "en"),
                word_count=len(transcript.split()),
                processing_time_seconds=processing_time
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_transcript(request: SummarizeRequest):
    """
    Generate summary from meeting transcript using Gemini LLM
    """
    if not gemini_model:
        raise HTTPException(
            status_code=503, 
            detail="Gemini not configured. Install google-generativeai and set GEMINI_API_KEY."
        )
    
    try:
        prompt = f"""Analyze this meeting transcript and provide a structured summary.

TRANSCRIPT:
{request.transcript}

Respond in this exact JSON format:
{{
    "brief": "A one-sentence summary of the meeting",
    "detailed": "A detailed 3-5 sentence summary covering main discussion points",
    "key_points": ["list", "of", "key", "points", "discussed"],
    "topics_discussed": ["topic1", "topic2", "topic3"],
    "action_items": ["action items or homework assigned"]
}}

Only respond with valid JSON, no other text."""

        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        result = json.loads(response_text.strip())
        
        # Store in MongoDB
        db = get_mongo_db()
        if db:
            db.summaries.update_one(
                {"meeting_id": request.meeting_id},
                {"$set": {
                    "summary": result,
                    "updated_at": datetime.utcnow()
                }},
                upsert=True
            )
        
        return SummarizeResponse(
            meeting_id=request.meeting_id,
            brief=result.get("brief", ""),
            detailed=result.get("detailed", ""),
            key_points=result.get("key_points", []),
            topics_discussed=result.get("topics_discussed", []),
            action_items=result.get("action_items", [])
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_meeting_content(request: QueryRequest):
    """
    Answer questions about meeting content using RAG (Qdrant + Gemini)
    """
    context_chunks = []
    
    # Search Qdrant for relevant content
    try:
        from qdrant_client import QdrantClient
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=503, detail="OpenAI API key required for embeddings")
        
        # Get embedding for query
        async with httpx.AsyncClient() as client:
            embed_response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"model": "text-embedding-3-small", "input": request.query}
            )
            
            if embed_response.status_code != 200:
                raise HTTPException(status_code=502, detail="Failed to generate query embedding")
            
            query_embedding = embed_response.json()["data"][0]["embedding"]
            
            # Build filter
            filter_conditions = []
            if request.meeting_id:
                filter_conditions.append({"key": "meeting_id", "match": {"value": request.meeting_id}})
            if request.classroom_id:
                filter_conditions.append({"key": "classroom_id", "match": {"value": request.classroom_id}})
            
            search_filter = {"must": filter_conditions} if filter_conditions else None
            
            # Search Qdrant
            results = qdrant.search(
                collection_name="meeting_chunks",
                query_vector=query_embedding,
                limit=request.max_results,
                query_filter=search_filter
            )
            
            context_chunks = [
                {
                    "meeting_id": r.payload.get("meeting_id", ""), 
                    "text": r.payload.get("text", ""),
                    "timestamp": r.payload.get("timestamp", ""),
                    "score": r.score
                }
                for r in results
            ]
            
    except ImportError:
        raise HTTPException(status_code=503, detail="Qdrant client not installed")
    except Exception as e:
        print(f"Qdrant search error: {e}")
    
    if not context_chunks:
        return QueryResponse(
            query=request.query,
            answer="No relevant meeting content found. Make sure meetings have been transcribed and indexed.",
            sources=[],
            confidence=0.0
        )
    
    # Generate answer using Gemini
    if not gemini_model:
        raise HTTPException(status_code=503, detail="Gemini not configured for answer generation")
    
    try:
        context_text = "\n".join([f"- {c['text']}" for c in context_chunks])
        prompt = f"""Answer this question based on the meeting transcript context:

QUESTION: {request.query}

CONTEXT FROM MEETINGS:
{context_text}

Provide a helpful, accurate answer based only on the context provided. If the answer isn't in the context, say so."""

        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=[
                {
                    "meeting_id": c["meeting_id"], 
                    "text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"], 
                    "timestamp": c["timestamp"]
                } 
                for c in context_chunks[:3]
            ],
            confidence=sum(c["score"] for c in context_chunks) / len(context_chunks) if context_chunks else 0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.get("/transcript/{meeting_id}")
async def get_transcript(meeting_id: str):
    """
    Retrieve stored transcript for a meeting from MongoDB
    """
    db = get_mongo_db()
    if not db:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    
    try:
        transcript_doc = db.transcripts.find_one({"meeting_id": meeting_id})
        summary_doc = db.summaries.find_one({"meeting_id": meeting_id})
        
        if not transcript_doc:
            return {
                "meeting_id": meeting_id,
                "has_transcript": False,
                "transcript": None,
                "summary": None
            }
        
        return {
            "meeting_id": meeting_id,
            "has_transcript": True,
            "transcript": transcript_doc.get("transcript", ""),
            "segments": transcript_doc.get("segments", []),
            "language": transcript_doc.get("language", "en"),
            "summary": summary_doc.get("summary") if summary_doc else None,
            "created_at": transcript_doc.get("created_at", datetime.utcnow()).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
