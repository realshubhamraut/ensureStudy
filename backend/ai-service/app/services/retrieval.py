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
        results = client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=top_k,
            score_threshold=threshold
        )
    except Exception as e:
        # If Qdrant is not available, return mock data for development
        if settings.DEBUG:
            return _get_mock_chunks(query, subject)
        raise e
    
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
        results = client.search(
            collection_name="meeting_transcripts",  # Different collection!
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=top_k,
            score_threshold=threshold
        )
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


# ============================================================================
# Mock Data (for development without Qdrant)
# ============================================================================

def _get_mock_chunks(query: str, subject: Optional[str] = None) -> List[RetrievedChunk]:
    """Return mock chunks for development/testing."""
    
    mock_data = {
        "physics": [
            {
                "document_id": "doc_physics_101",
                "chunk_id": "ch_001",
                "text": "Newton's first law of motion, also known as the law of inertia, states that an object at rest will remain at rest, and an object in motion will remain in motion at a constant velocity, unless acted upon by an unbalanced external force. This principle was revolutionary because it contradicted the Aristotelian view that objects naturally tend toward rest.",
                "similarity_score": 0.89,
                "metadata": {"subject": "physics", "topic": "mechanics", "chapter": "3"}
            },
            {
                "document_id": "doc_physics_101",
                "chunk_id": "ch_002",
                "text": "Newton's second law of motion states that the acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass. Mathematically, this is expressed as F = ma, where F is force, m is mass, and a is acceleration.",
                "similarity_score": 0.82,
                "metadata": {"subject": "physics", "topic": "mechanics", "chapter": "3"}
            },
            {
                "document_id": "doc_physics_101",
                "chunk_id": "ch_003",
                "text": "Example: A 5 kg object experiences a force of 10 N. Using F = ma, the acceleration is a = F/m = 10/5 = 2 m/s². This demonstrates how to apply Newton's second law in practice.",
                "similarity_score": 0.75,
                "metadata": {"subject": "physics", "topic": "mechanics", "chapter": "3", "type": "example"}
            },
        ],
        "biology": [
            {
                "document_id": "doc_bio_ch4",
                "chunk_id": "ch_101",
                "text": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water. It occurs in the chloroplasts using chlorophyll pigment. The overall equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.",
                "similarity_score": 0.91,
                "metadata": {"subject": "biology", "topic": "plant biology", "chapter": "4"}
            },
            {
                "document_id": "doc_bio_ch4",
                "chunk_id": "ch_102",
                "text": "The light-dependent reactions occur in the thylakoid membranes, producing ATP and NADPH. The light-independent reactions (Calvin cycle) occur in the stroma, using ATP and NADPH to fix carbon dioxide into glucose.",
                "similarity_score": 0.84,
                "metadata": {"subject": "biology", "topic": "plant biology", "chapter": "4"}
            },
        ],
        "math": [
            {
                "document_id": "doc_math_algebra",
                "chunk_id": "ch_201",
                "text": "The quadratic formula is used to solve equations of the form ax² + bx + c = 0. The solution is given by x = (-b ± √(b² - 4ac)) / 2a. The discriminant (b² - 4ac) determines the nature of roots: positive gives two real roots, zero gives one real root, negative gives complex roots.",
                "similarity_score": 0.88,
                "metadata": {"subject": "math", "topic": "algebra", "chapter": "5"}
            },
            {
                "document_id": "doc_math_algebra",
                "chunk_id": "ch_202",
                "text": "Example: Solve x² - 5x + 6 = 0. Here a=1, b=-5, c=6. Discriminant = 25-24 = 1 > 0. Solutions: x = (5 ± 1)/2. So x = 3 or x = 2.",
                "similarity_score": 0.79,
                "metadata": {"subject": "math", "topic": "algebra", "chapter": "5", "type": "example"}
            },
        ]
    }
    
    # Return subject-specific or general chunks
    if subject and subject in mock_data:
        chunks = mock_data[subject]
    else:
        # If no subject, filter by relevance/threshold instead of returning everything
        chunks = []
        similarity_threshold = settings.SIMILARITY_THRESHOLD or 0.6
        
        # Simple keyword matching to simulate semantic search on mock data
        query_terms = set(query.lower().split())
        
        for subj, subj_chunks in mock_data.items():
            for chunk in subj_chunks:
                # Mock similarity: check if any query term is in the chunk text
                text_lower = chunk["text"].lower()
                matches = sum(1 for term in query_terms if term in text_lower and len(term) > 3)
                
                # Boost score if matches found, otherwise low score
                mock_score = 0.85 if matches > 0 else 0.2
                
                # Only include if score meets threshold
                if mock_score >= similarity_threshold:
                    chunks.append({**chunk, "similarity_score": mock_score})
        
        # If no strict matches, return nothing (prevent junk pollution)
        if not chunks and "test" in query.lower():
             # Only strictly fallback for "test" queries
             for subject_chunks in mock_data.values():
                 chunks.extend(subject_chunks)

        chunks.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        chunks = chunks[:settings.TOP_K_RESULTS]
    
    return [
        RetrievedChunk(
            document_id=c["document_id"],
            chunk_id=c["chunk_id"],
            text=c["text"],
            similarity_score=c["similarity_score"],
            metadata=c["metadata"]
        )
        for c in chunks
    ]
