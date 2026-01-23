"""
Semantic Retrieval Service using Qdrant

Uses Sentence-Transformers (Hugging Face) for embeddings.
Embedding Model: sentence-transformers/all-mpnet-base-v2 (FREE, LOCAL)

Returns:
- chunk_text
- document_id
- chunk_id
- similarity_score
"""
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from ..api.schemas.tutor import RetrievedChunk
from ..config import settings


# ============================================================================
# Embedding Model (Singleton, Hugging Face)
# ============================================================================

_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """
    Lazy-load Sentence-Transformers model.
    Uses: sentence-transformers/all-mpnet-base-v2
    Runs locally, no API cost.
    """
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedding_model


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding vector for text using Sentence-Transformers.
    
    Args:
        text: Query or document text
        
    Returns:
        List of 768 floats (all-mpnet-base-v2 dimension)
    """
    model = get_embedding_model()
    # Encode returns numpy array, convert to list
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


# ============================================================================
# Qdrant Client (Singleton)
# ============================================================================

_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
    return _qdrant_client


# ============================================================================
# Semantic Search
# ============================================================================

def semantic_search(
    query: str,
    user_id: str,
    subject: Optional[str] = None,
    top_k: int = None,
    threshold: float = None
) -> List[RetrievedChunk]:
    """
    Search Qdrant for semantically similar content.
    
    Args:
        query: The student's question
        user_id: Student ID (for future personalization/filtering)
        subject: Optional subject filter (physics, math, etc.)
        top_k: Number of results (default: 8)
        threshold: Minimum similarity score (default: 0.5)
        
    Returns:
        List of RetrievedChunk with:
        - chunk_text
        - document_id
        - chunk_id
        - similarity_score
    """
    top_k = top_k or settings.TOP_K_RESULTS
    threshold = threshold or settings.SIMILARITY_THRESHOLD
    
    # Step 1: Generate query embedding
    query_embedding = generate_embedding(query)
    
    # Step 2: Build optional filter
    search_filter = None
    if subject:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="subject",
                    match=MatchValue(value=subject)
                )
            ]
        )
    
    # Step 3: Query Qdrant
    client = get_qdrant_client()
    
    try:
        query_result = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_embedding,
            query_filter=search_filter,
            limit=top_k,
            score_threshold=threshold
        )
        # Extract points from QueryResponse
        results = query_result.points if hasattr(query_result, 'points') else []
    except Exception as e:
        # If Qdrant is not available, return empty list (no mock data!)
        print(f"[Retrieval] Qdrant query failed: {e}")
        return []
    
    # Step 4: Convert to RetrievedChunk objects
    chunks = []
    for result in results:
        chunk = RetrievedChunk(
            document_id=result.payload.get("document_id", "unknown"),
            chunk_id=result.id if isinstance(result.id, str) else f"chunk_{result.id}",
            text=result.payload.get("text", ""),
            similarity_score=result.score,
            metadata={
                k: v for k, v in result.payload.items()
                if k not in ["document_id", "text"]
            }
        )
        chunks.append(chunk)
    
    return chunks


# ============================================================================
# Meeting Transcript Search (for AI Tutor)
# ============================================================================

def search_meeting_transcripts(
    query: str,
    classroom_id: str,
    meeting_ids: Optional[List[str]] = None,
    top_k: int = 5,
    threshold: float = 0.5
) -> List[RetrievedChunk]:
    """
    Search meeting transcripts for semantically similar content.
    
    Used by AI Tutor to include class lecture context in responses.
    
    Args:
        query: The student's question
        classroom_id: Filter by classroom (required)
        meeting_ids: Optional list of specific meetings to search
        top_k: Number of results (default: 5)
        threshold: Minimum similarity score (default: 0.5)
        
    Returns:
        List of RetrievedChunk with transcript content and speaker info
    """
    # Generate query embedding
    query_embedding = generate_embedding(query)
    
    # Build filter for classroom and optionally specific meetings
    must_conditions = [
        FieldCondition(
            key="classroom_id",
            match=MatchValue(value=classroom_id)
        )
    ]
    
    if meeting_ids:
        # Add meeting filter if specific meetings requested
        must_conditions.append(
            FieldCondition(
                key="meeting_id",
                match=MatchValue(value=meeting_ids[0])  # Qdrant doesn't support IN by default
            )
        )
    
    search_filter = Filter(must=must_conditions)
    
    # Query Qdrant meeting_transcripts collection
    client = get_qdrant_client()
    
    try:
        query_result = client.query_points(
            collection_name="meeting_transcripts",  # Different collection!
            query=query_embedding,
            query_filter=search_filter,
            limit=top_k,
            score_threshold=threshold
        )
        # Extract points from QueryResponse
        results = query_result.points if hasattr(query_result, 'points') else []
    except Exception as e:
        print(f"[MeetingSearch] Error searching transcripts: {e}")
        return []
    
    # Convert to RetrievedChunk with transcript-specific metadata
    chunks = []
    for result in results:
        payload = result.payload
        
        # Build speaker info string
        speaker_names = payload.get("speaker_names", [])
        speaker_info = ", ".join(speaker_names) if speaker_names else "Unknown speaker"
        
        # Build time info
        start_time = payload.get("start_time", 0)
        end_time = payload.get("end_time", 0)
        time_info = f"{int(start_time // 60)}:{int(start_time % 60):02d} - {int(end_time // 60)}:{int(end_time % 60):02d}"
        
        chunk = RetrievedChunk(
            document_id=payload.get("recording_id", "unknown"),
            chunk_id=result.id if isinstance(result.id, str) else f"transcript_{result.id}",
            text=payload.get("chunk_text", ""),
            similarity_score=result.score,
            metadata={
                "source_type": "meeting_transcript",
                "meeting_id": payload.get("meeting_id", ""),
                "classroom_id": payload.get("classroom_id", ""),
                "speaker": speaker_info,
                "time_range": time_info,
                "start_time": start_time,
                "end_time": end_time
            }
        )
        chunks.append(chunk)
    
    print(f"[MeetingSearch] Found {len(chunks)} transcript chunks for query")
    return chunks

# Note: Mock data removed - only real classroom materials are shown now
