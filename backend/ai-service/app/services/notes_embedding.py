"""
Notes Embedding Service
Generates embeddings for semantic search in digitized notes

Technologies used:
- Sentence Transformers for embeddings
- Qdrant vector database
- Document chunking
- RAG (Retrieval Augmented Generation)
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import uuid
import os

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with metadata"""
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    page_id: str
    job_id: str
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """A search result from semantic search"""
    chunk: TextChunk
    score: float
    page_number: int
    job_title: str


class NotesEmbeddingService:
    """
    Handles text chunking, embedding generation, and vector search for notes
    
    Uses:
    - sentence-transformers for embedding generation
    - Qdrant for vector storage and search
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "notes_embeddings",
        qdrant_host: str = None,
        qdrant_port: int = 6333,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        self.model_name = model_name
        self.collection_name = collection_name
        self.qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = qdrant_port
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.embedder = None
        self.qdrant_client = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        self._initialize()
    
    def _initialize(self):
        """Initialize embedding model and Qdrant client"""
        # Load embedding model
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.embedder = SentenceTransformer(self.model_name)
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Could not load embedding model: {e}")
            raise
        
        # Connect to Qdrant
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance
            
            logger.info(f"Connecting to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
            self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            
            # Create collection if not exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
            
            logger.info("Qdrant connected successfully")
            
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant: {e}. Vector search will be disabled.")
            self.qdrant_client = None
    
    def chunk_text(
        self,
        text: str,
        page_id: str,
        job_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks for embedding
        
        Uses sentence-aware chunking to avoid breaking mid-sentence
        """
        if not text or not text.strip():
            return []
        
        # Clean text
        text = text.strip()
        
        # Split into sentences (simple approach)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        start_char = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_length + sentence_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    page_id=page_id,
                    job_id=job_id,
                    metadata=metadata or {}
                ))
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
                start_char = start_char + len(chunk_text) - overlap_length
                chunk_index += 1
            
            current_chunk.append(sentence)
            current_length += sentence_len
        
        # Save last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                page_id=page_id,
                job_id=job_id,
                metadata=metadata or {}
            ))
        
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not self.embedder:
            raise RuntimeError("Embedding model not loaded")
        
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def index_chunks(
        self,
        chunks: List[TextChunk],
        student_id: str,
        classroom_id: str
    ) -> List[str]:
        """
        Index chunks in Qdrant vector database
        
        Returns:
            List of Qdrant point IDs
        """
        if not self.qdrant_client:
            logger.warning("Qdrant not available, skipping indexing")
            return []
        
        if not chunks:
            return []
        
        from qdrant_client.models import PointStruct
        
        # Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = self.generate_embeddings(texts)
        
        # Create points
        points = []
        point_ids = []
        
        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "page_id": chunk.page_id,
                    "job_id": chunk.job_id,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "student_id": student_id,
                    "classroom_id": classroom_id,
                    **chunk.metadata
                }
            ))
        
        # Upsert to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Indexed {len(points)} chunks to Qdrant")
        return point_ids
    
    def search(
        self,
        query: str,
        student_id: str,
        classroom_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across indexed notes
        
        Returns:
            List of search results with scores
        """
        if not self.qdrant_client:
            logger.warning("Qdrant not available")
            return []
        
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        # Build filter
        filter_conditions = [
            FieldCondition(key="student_id", match=MatchValue(value=student_id))
        ]
        
        if classroom_id:
            filter_conditions.append(
                FieldCondition(key="classroom_id", match=MatchValue(value=classroom_id))
            )
        
        search_filter = Filter(must=filter_conditions)
        
        # Search
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=limit,
            with_payload=True
        )
        
        # Format results
        formatted = []
        for result in results:
            formatted.append({
                "score": result.score,
                "text": result.payload.get("text", ""),
                "page_id": result.payload.get("page_id"),
                "job_id": result.payload.get("job_id"),
                "chunk_index": result.payload.get("chunk_index"),
                "metadata": {k: v for k, v in result.payload.items() 
                           if k not in ["text", "page_id", "job_id", "chunk_index", "student_id", "classroom_id"]}
            })
        
        return formatted
    
    def delete_job_embeddings(self, job_id: str) -> int:
        """
        Delete all embeddings for a job
        
        Returns:
            Number of deleted points
        """
        if not self.qdrant_client:
            return 0
        
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Delete points matching job_id
        result = self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="job_id", match=MatchValue(value=job_id))]
            )
        )
        
        logger.info(f"Deleted embeddings for job {job_id}")
        return 1  # Qdrant doesn't return count


class RAGService:
    """
    RAG (Retrieval Augmented Generation) for Q&A on notes
    
    Combines semantic search with LLM for answering questions
    """
    
    def __init__(self, embedding_service: NotesEmbeddingService):
        self.embedding_service = embedding_service
        self.llm = None
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM for answer generation"""
        try:
            from transformers import pipeline
            
            # Use a small, fast model for answer generation
            self.llm = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                max_length=512
            )
            logger.info("LLM loaded for RAG")
        except Exception as e:
            logger.warning(f"Could not load LLM: {e}. RAG will return raw search results.")
    
    def answer_question(
        self,
        question: str,
        student_id: str,
        classroom_id: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG
        
        1. Retrieve relevant chunks
        2. Generate answer using LLM with context
        """
        # Retrieve relevant chunks
        search_results = self.embedding_service.search(
            query=question,
            student_id=student_id,
            classroom_id=classroom_id,
            limit=top_k
        )
        
        if not search_results:
            return {
                "answer": "I couldn't find any relevant information in your notes.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Build context from search results
        context = "\n\n".join([r["text"] for r in search_results])
        
        # Generate answer
        if self.llm:
            prompt = f"""Based on the following notes, answer the question.

Notes:
{context}

Question: {question}

Answer:"""
            
            try:
                response = self.llm(prompt)[0]["generated_text"]
                answer = response.strip()
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                answer = f"Based on your notes: {search_results[0]['text'][:500]}..."
        else:
            # No LLM, return top result
            answer = f"From your notes: {search_results[0]['text']}"
        
        return {
            "answer": answer,
            "sources": [
                {
                    "page_id": r["page_id"],
                    "job_id": r["job_id"],
                    "text_snippet": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                    "relevance_score": r["score"]
                }
                for r in search_results
            ],
            "confidence": search_results[0]["score"] if search_results else 0.0
        }
