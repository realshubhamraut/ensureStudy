"""
Meeting Embedding Service
Generates vector embeddings for meeting transcripts and stores in Qdrant
Enables semantic search across all meeting content

Uses Sentence-Transformers (FREE, LOCAL) - same as AI tutor retrieval
"""
import os
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer

# Configuration
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
COLLECTION_NAME = 'meeting_transcripts'
# Use same model as AI tutor retrieval for consistency
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
EMBEDDING_DIMENSIONS = 768  # For sentence-transformers/all-mpnet-base-v2

# Lazy-loaded embedding model (shared)
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load Sentence-Transformers model (FREE, LOCAL)"""
    global _embedding_model
    if _embedding_model is None:
        model_name = EMBEDDING_MODEL.replace('sentence-transformers/', '')
        print(f"[MeetingEmbedding] Loading model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


@dataclass
class TranscriptChunk:
    """A chunk of transcript for embedding"""
    id: str
    recording_id: str
    meeting_id: str
    classroom_id: str
    chunk_text: str
    start_time: float
    end_time: float
    speaker_ids: List[int]
    speaker_names: List[str]
    created_at: str


class MeetingEmbeddingService:
    """
    Service for generating and storing meeting transcript embeddings
    
    Uses Sentence-Transformers (FREE, LOCAL) for embeddings.
    Same model as AI tutor retrieval for consistent semantic search.
    
    Chunks transcripts by:
    - Speaker changes
    - Semantic boundaries (~500 tokens per chunk)
    - Time boundaries (~30 seconds)
    """
    
    def __init__(self):
        self._client = None
        self._initialized = False
    
    @property
    def client(self):
        """Lazy-load Qdrant client only when needed"""
        if self._client is None:
            try:
                self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                self._ensure_collection()
                self._initialized = True
            except Exception as e:
                print(f"Warning: Qdrant not available - {e}")
                self._client = None
        return self._client
    
    def _ensure_collection(self):
        """Ensure Qdrant collection exists with correct dimensions"""
        if self._client is None:
            return
            
        try:
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if COLLECTION_NAME not in collection_names:
                self._client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMENSIONS,  # 768 for sentence-transformers
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection: {COLLECTION_NAME} (dim={EMBEDDING_DIMENSIONS})")
        except Exception as e:
            print(f"Warning: Could not ensure Qdrant collection - {e}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Sentence-Transformers (FREE, LOCAL)"""
        def _embed():
            model = get_embedding_model()
            embedding = model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        
        return await asyncio.to_thread(_embed)
    
    def chunk_transcript(
        self,
        segments: List[Dict],
        max_tokens: int = 500,
        max_duration: float = 60.0
    ) -> List[Dict]:
        """
        Chunk transcript segments into embedding-sized pieces
        
        Preserves:
        - Speaker continuity where possible
        - Semantic coherence
        - Reasonable time boundaries
        """
        chunks = []
        current_chunk = {
            'texts': [],
            'speakers': set(),
            'speaker_names': set(),
            'start_time': None,
            'end_time': None,
            'token_count': 0
        }
        
        for seg in segments:
            text = seg.get('text', '')
            # Rough token estimate
            token_count = len(text.split()) * 1.3
            
            # Check if we should start new chunk
            should_break = False
            
            if current_chunk['texts']:
                # Break if too many tokens
                if current_chunk['token_count'] + token_count > max_tokens:
                    should_break = True
                # Break if too long duration
                elif current_chunk['start_time'] is not None:
                    duration = seg.get('end', 0) - current_chunk['start_time']
                    if duration > max_duration:
                        should_break = True
            
            if should_break and current_chunk['texts']:
                # Save current chunk
                chunks.append({
                    'text': ' '.join(current_chunk['texts']),
                    'speakers': list(current_chunk['speakers']),
                    'speaker_names': list(current_chunk['speaker_names']),
                    'start_time': current_chunk['start_time'],
                    'end_time': current_chunk['end_time']
                })
                
                # Start new chunk
                current_chunk = {
                    'texts': [],
                    'speakers': set(),
                    'speaker_names': set(),
                    'start_time': None,
                    'end_time': None,
                    'token_count': 0
                }
            
            # Add segment to current chunk
            current_chunk['texts'].append(text)
            current_chunk['speakers'].add(seg.get('speaker_id', 0))
            if seg.get('speaker_name'):
                current_chunk['speaker_names'].add(seg['speaker_name'])
            
            if current_chunk['start_time'] is None:
                current_chunk['start_time'] = seg.get('start', 0)
            current_chunk['end_time'] = seg.get('end', 0)
            current_chunk['token_count'] += token_count
        
        # Don't forget last chunk
        if current_chunk['texts']:
            chunks.append({
                'text': ' '.join(current_chunk['texts']),
                'speakers': list(current_chunk['speakers']),
                'speaker_names': list(current_chunk['speaker_names']),
                'start_time': current_chunk['start_time'],
                'end_time': current_chunk['end_time']
            })
        
        return chunks
    
    async def embed_transcript(
        self,
        recording_id: str,
        meeting_id: str,
        classroom_id: str,
        segments: List[Dict],
        meeting_title: str = ""
    ) -> int:
        """
        Generate embeddings for all transcript chunks and store in Qdrant
        
        Returns number of chunks indexed
        """
        # Chunk the transcript
        chunks = self.chunk_transcript(segments)
        
        if not chunks:
            return 0
        
        # Generate embeddings for all chunks
        points = []
        
        for i, chunk in enumerate(chunks):
            # Prepend meeting title for context
            text_to_embed = f"{meeting_title}: {chunk['text']}" if meeting_title else chunk['text']
            
            # Generate embedding
            embedding = await self.generate_embedding(text_to_embed)
            
            # Create point
            point_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    'recording_id': recording_id,
                    'meeting_id': meeting_id,
                    'classroom_id': classroom_id,
                    'chunk_index': i,
                    'chunk_text': chunk['text'],
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                    'speaker_ids': chunk['speakers'],
                    'speaker_names': chunk['speaker_names'],
                    'created_at': datetime.utcnow().isoformat()
                }
            ))
        
        # Upsert to Qdrant
        if self.client is not None:
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
        else:
            print("Warning: Qdrant not available, embeddings not stored")
            return 0
        
        return len(points)
    
    async def search(
        self,
        query: str,
        classroom_id: Optional[str] = None,
        meeting_ids: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across meeting transcripts
        
        Returns matching chunks with metadata
        """
        # Generate query embedding
        query_embedding = await self.generate_embedding(query)
        
        # Build filter
        filter_conditions = []
        
        if classroom_id:
            filter_conditions.append(
                FieldCondition(
                    key="classroom_id",
                    match=MatchValue(value=classroom_id)
                )
            )
        
        if meeting_ids:
            # Filter by multiple meeting IDs
            for mid in meeting_ids:
                filter_conditions.append(
                    FieldCondition(
                        key="meeting_id",
                        match=MatchValue(value=mid)
                    )
                )
        
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Check if Qdrant is available
        if self.client is None:
            print("Warning: Qdrant not available for search")
            return []
        
        # Search
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                'score': hit.score,
                **hit.payload
            }
            for hit in results
        ]
    
    def delete_recording_embeddings(self, recording_id: str):
        """Delete all embeddings for a recording"""
        if self.client is None:
            print("Warning: Qdrant not available for delete")
            return
            
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="recording_id",
                        match=MatchValue(value=recording_id)
                    )
                ]
            )
        )


# Singleton instance
meeting_embedding_service = MeetingEmbeddingService()
