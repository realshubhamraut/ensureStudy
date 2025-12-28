"""
Material Indexer Service
Orchestrates the indexing of classroom materials into Qdrant for RAG.

Pipeline:
1. Download PDF from URL
2. Extract text (PDF/OCR)
3. Chunk text into segments
4. Generate embeddings
5. Store in Qdrant with classroom_id metadata
"""
import os
import logging
import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import uuid

from .pdf_processor import PDFProcessor, PDFExtractionResult
from .chunking_service import create_semantic_chunks
from .qdrant_service import QdrantService, ChunkMetadata, SourceType

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Result of material indexing"""
    success: bool
    material_id: str
    chunks_indexed: int
    total_words: int
    processing_time_ms: int
    error: Optional[str] = None


class MaterialIndexer:
    """
    Index classroom materials into Qdrant for semantic search.
    
    Supports:
    - PDF documents (text and scanned)
    - Automatic chunking with overlap
    - Embedding generation
    - Classroom-aware indexing
    """
    
    def __init__(self, collection_name: str = "classroom_materials"):
        """
        Initialize the material indexer.
        
        Args:
            collection_name: Qdrant collection for materials
        """
        self.collection_name = collection_name
        self.pdf_processor = PDFProcessor()
        self.qdrant_service = None
        self.embedding_model = None
        
    def _ensure_services(self):
        """Lazy load services on first use."""
        if self.qdrant_service is None:
            self.qdrant_service = QdrantService(collection_name=self.collection_name)
            
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                logger.info(f"[INDEXER] Loading embedding model: {model_name}")
                self.embedding_model = SentenceTransformer(model_name)
            except ImportError:
                logger.error("[INDEXER] sentence-transformers not installed")
                raise
    
    async def index_material(
        self,
        material_id: str,
        file_url: str,
        classroom_id: str,
        subject: Optional[str] = None,
        document_title: Optional[str] = None,
        uploaded_by: Optional[str] = None
    ) -> IndexResult:
        """
        Index a classroom material into Qdrant.
        
        Args:
            material_id: Unique ID of the material
            file_url: URL to download the PDF from
            classroom_id: Classroom this material belongs to
            subject: Optional subject/topic
            document_title: Original filename or title
            uploaded_by: Teacher ID who uploaded
            
        Returns:
            IndexResult with indexing status
        """
        start_time = datetime.now()
        
        try:
            self._ensure_services()
            
            logger.info(f"[INDEXER] Starting indexing for material: {material_id}")
            logger.info(f"[INDEXER] Classroom: {classroom_id}, Subject: {subject}")
            
            # Step 1: Extract text from PDF
            logger.info("[INDEXER] Step 1: Extracting text from PDF...")
            extraction_result = self.pdf_processor.extract_from_url(file_url)
            
            if not extraction_result.success:
                return IndexResult(
                    success=False,
                    material_id=material_id,
                    chunks_indexed=0,
                    total_words=0,
                    processing_time_ms=self._elapsed_ms(start_time),
                    error=extraction_result.error or "PDF extraction failed"
                )
            
            if extraction_result.total_words < 10:
                return IndexResult(
                    success=False,
                    material_id=material_id,
                    chunks_indexed=0,
                    total_words=extraction_result.total_words,
                    processing_time_ms=self._elapsed_ms(start_time),
                    error="PDF contains insufficient text"
                )
            
            logger.info(f"[INDEXER] Extracted {extraction_result.total_words} words from {extraction_result.total_pages} pages")
            
            # Step 2: Chunk the text
            logger.info("[INDEXER] Step 2: Chunking text...")
            full_text = self.pdf_processor.get_full_text(extraction_result)
            chunks = create_semantic_chunks(
                full_text,
                chunk_size=500,
                overlap=100
            )
            
            if not chunks:
                return IndexResult(
                    success=False,
                    material_id=material_id,
                    chunks_indexed=0,
                    total_words=extraction_result.total_words,
                    processing_time_ms=self._elapsed_ms(start_time),
                    error="Failed to create text chunks"
                )
            
            logger.info(f"[INDEXER] Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            logger.info("[INDEXER] Step 3: Generating embeddings...")
            chunk_texts = [c["text"] for c in chunks]
            embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=False)
            
            # Step 4: Prepare metadata and index
            logger.info("[INDEXER] Step 4: Indexing into Qdrant...")
            indexed_count = 0
            batch = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{material_id}_chunk_{i}"
                
                # Detect page number from chunk text if available
                page_number = self._extract_page_number(chunk["text"])
                
                metadata = ChunkMetadata(
                    document_id=material_id,
                    chunk_id=chunk_id,
                    chunk_index=i,
                    chunk_text=chunk["text"],
                    source_type=SourceType.TEACHER_MATERIAL.value,
                    source_confidence=0.95,  # High trust for teacher materials
                    student_id="",  # Not applicable for teacher materials
                    classroom_id=classroom_id,
                    page_number=page_number,
                    title=document_title or "Classroom Material",
                    subject=subject,
                    url=file_url,
                    created_at=datetime.now().isoformat()
                )
                
                batch.append((embedding.tolist(), metadata))
            
            # Batch index
            point_ids = self.qdrant_service.index_batch(batch)
            indexed_count = len(point_ids)
            
            processing_time = self._elapsed_ms(start_time)
            logger.info(f"[INDEXER] ✅ Indexed {indexed_count} chunks in {processing_time}ms")
            
            return IndexResult(
                success=True,
                material_id=material_id,
                chunks_indexed=indexed_count,
                total_words=extraction_result.total_words,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"[INDEXER] ❌ Indexing failed: {e}", exc_info=True)
            return IndexResult(
                success=False,
                material_id=material_id,
                chunks_indexed=0,
                total_words=0,
                processing_time_ms=self._elapsed_ms(start_time),
                error=str(e)
            )
    
    async def delete_material(self, material_id: str) -> bool:
        """
        Delete all indexed chunks for a material.
        
        Args:
            material_id: Material ID to delete
            
        Returns:
            True if successful
        """
        try:
            self._ensure_services()
            deleted_count = self.qdrant_service.delete_document(material_id)
            logger.info(f"[INDEXER] Deleted {deleted_count} chunks for material {material_id}")
            return True
        except Exception as e:
            logger.error(f"[INDEXER] Delete failed: {e}")
            return False
    
    def search_classroom_materials(
        self,
        query: str,
        classroom_id: str,
        top_k: int = 5,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search indexed materials for a classroom.
        
        Args:
            query: Search query
            classroom_id: Classroom to search within
            top_k: Number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of matching chunks with metadata
        """
        try:
            self._ensure_services()
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search with classroom filter
            results = self.qdrant_service.search_semantic(
                query_embedding=query_embedding,
                classroom_id=classroom_id,
                top_k=top_k,
                score_threshold=score_threshold,
                filters={"source_type": SourceType.TEACHER_MATERIAL.value}
            )
            
            return [
                {
                    "chunk_text": r.payload.get("chunk_text", ""),
                    "document_id": r.payload.get("document_id", ""),
                    "title": r.payload.get("title", ""),
                    "page_number": r.payload.get("page_number", 0),
                    "similarity_score": r.final_score,
                    "url": r.payload.get("url", ""),
                    "subject": r.payload.get("subject", "")
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"[INDEXER] Search failed: {e}")
            return []
    
    def _elapsed_ms(self, start_time: datetime) -> int:
        """Calculate elapsed milliseconds."""
        return int((datetime.now() - start_time).total_seconds() * 1000)
    
    def _extract_page_number(self, text: str) -> int:
        """Extract page number from chunk text if present."""
        import re
        match = re.search(r'\[Page (\d+)\]', text)
        return int(match.group(1)) if match else 0


# Singleton instance
_indexer_instance: Optional[MaterialIndexer] = None


def get_material_indexer() -> MaterialIndexer:
    """Get or create the material indexer singleton."""
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = MaterialIndexer()
    return _indexer_instance
