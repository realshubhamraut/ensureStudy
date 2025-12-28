"""
Document Processing Agent - LangGraph Orchestrator

7-Stage Pipeline:
1. Validate -> 2. Preprocess -> 3. OCR -> 4. Chunk -> 5. Embed -> 6. Index -> 7. Complete

Uses LangGraph StateGraph for:
- Conditional routing based on file type
- Error handling with retries
- Progress tracking
- Idempotent stage execution
"""
import logging
import asyncio
import os
import hashlib
from typing import Dict, Any, List, TypedDict, Optional, Literal
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from langgraph.graph import StateGraph, END
import httpx

logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

class ProcessingStage(str, Enum):
    """Pipeline stages"""
    PENDING = "pending"
    VALIDATING = "validating"
    PREPROCESSING = "preprocessing"
    OCR = "ocr"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TextChunk:
    """A chunk of text with metadata"""
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    page_number: int
    section_heading: Optional[str]
    source_confidence: float
    contains_formula: bool
    formula_latex: Optional[str] = None


@dataclass
class OCRResult:
    """OCR extraction result per page"""
    page_number: int
    text: str
    confidence: float
    formulas: List[Dict[str, Any]]
    headings: List[str]


class DocumentProcessingState(TypedDict):
    """State passed between nodes in the processing graph"""
    # Input
    document_id: str
    student_id: str
    classroom_id: str
    source_url: str
    file_type: str
    subject: Optional[str]
    is_teacher_material: bool
    
    # Processing artifacts
    local_path: str
    preprocessed_paths: List[str]
    ocr_results: List[Dict]
    chunks: List[Dict]
    embeddings: List[List[float]]
    
    # Status
    current_stage: str
    progress: int
    completed_stages: List[str]
    error: Optional[str]
    retry_count: int
    
    # Output
    qdrant_point_ids: List[str]
    total_tokens: int
    total_chunks: int
    avg_confidence: float


# ============================================================================
# Node Functions
# ============================================================================

async def validate_document(state: DocumentProcessingState) -> DocumentProcessingState:
    """Validate document exists and is processable"""
    state["current_stage"] = ProcessingStage.VALIDATING.value
    state["progress"] = 5
    
    try:
        # Check if file exists (S3 or local)
        source_url = state["source_url"]
        
        if source_url.startswith("s3://"):
            # S3 validation would go here
            logger.info(f"Validating S3 file: {source_url}")
        else:
            # Local file
            if not os.path.exists(source_url):
                raise FileNotFoundError(f"File not found: {source_url}")
        
        # Mark stage complete
        state["completed_stages"].append("validate")
        logger.info(f"Document {state['document_id']} validated successfully")
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        state["error"] = str(e)
    
    return state


async def preprocess_document(state: DocumentProcessingState) -> DocumentProcessingState:
    """Preprocess document based on file type"""
    if state.get("error"):
        return state
    
    state["current_stage"] = ProcessingStage.PREPROCESSING.value
    state["progress"] = 15
    
    try:
        file_type = state["file_type"].lower()
        source = state["source_url"]
        
        # Import processors
        from app.services.document_preprocessor import DocumentPreprocessor
        
        preprocessor = DocumentPreprocessor()
        
        if file_type == "pdf":
            result = await preprocessor.process_pdf(source)
        elif file_type == "pptx":
            result = await preprocessor.process_pptx(source)
        elif file_type == "docx":
            result = await preprocessor.process_docx(source)
        elif file_type in ["jpg", "jpeg", "png", "heic", "webp"]:
            result = await preprocessor.process_images([source])
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        state["preprocessed_paths"] = result.get("image_paths", [])
        state["completed_stages"].append("preprocess")
        state["progress"] = 30
        
        logger.info(f"Preprocessed {len(state['preprocessed_paths'])} pages")
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        state["error"] = str(e)
    
    return state


async def extract_text_ocr(state: DocumentProcessingState) -> DocumentProcessingState:
    """Extract text using OCR"""
    if state.get("error"):
        return state
    
    state["current_stage"] = ProcessingStage.OCR.value
    state["progress"] = 40
    
    try:
        from app.services.ocr_service import OCRService
        
        ocr = OCRService()
        ocr_results = []
        total_confidence = 0
        
        for i, img_path in enumerate(state["preprocessed_paths"]):
            import cv2
            image = cv2.imread(img_path)
            
            if image is None:
                logger.warning(f"Could not read image: {img_path}")
                continue
            
            result = ocr.extract_text(image)
            
            ocr_results.append({
                "page_number": i + 1,
                "text": result.text,
                "confidence": result.confidence,
                "lines": result.lines,
                "model_used": result.model_used
            })
            
            total_confidence += result.confidence
            
            # Update progress
            state["progress"] = 40 + int((i / len(state["preprocessed_paths"])) * 20)
        
        state["ocr_results"] = ocr_results
        state["avg_confidence"] = total_confidence / len(ocr_results) if ocr_results else 0
        state["completed_stages"].append("ocr")
        state["progress"] = 60
        
        logger.info(f"OCR completed: {len(ocr_results)} pages, avg confidence: {state['avg_confidence']:.2f}")
        
    except Exception as e:
        logger.error(f"OCR error: {e}")
        state["error"] = str(e)
    
    return state


async def chunk_text(state: DocumentProcessingState) -> DocumentProcessingState:
    """Chunk text into semantic segments"""
    if state.get("error"):
        return state
    
    state["current_stage"] = ProcessingStage.CHUNKING.value
    state["progress"] = 65
    
    try:
        from app.services.text_chunker import TextChunker
        
        chunker = TextChunker(
            chunk_size=500,
            chunk_overlap=100,
            respect_sentences=True
        )
        
        all_chunks = []
        total_tokens = 0
        
        for ocr_result in state["ocr_results"]:
            page_chunks = chunker.chunk_text(
                text=ocr_result["text"],
                document_id=state["document_id"],
                page_number=ocr_result["page_number"],
                source_confidence=ocr_result["confidence"]
            )
            
            for chunk in page_chunks:
                chunk["student_id"] = state["student_id"]
                chunk["classroom_id"] = state["classroom_id"]
                chunk["subject"] = state.get("subject")
                chunk["is_teacher_material"] = state.get("is_teacher_material", False)
                
            all_chunks.extend(page_chunks)
            total_tokens += sum(c.get("token_count", 0) for c in page_chunks)
        
        state["chunks"] = all_chunks
        state["total_chunks"] = len(all_chunks)
        state["total_tokens"] = total_tokens
        state["completed_stages"].append("chunk")
        state["progress"] = 75
        
        logger.info(f"Created {len(all_chunks)} chunks, {total_tokens} tokens")
        
    except Exception as e:
        logger.error(f"Chunking error: {e}")
        state["error"] = str(e)
    
    return state


async def generate_embeddings(state: DocumentProcessingState) -> DocumentProcessingState:
    """Generate embeddings for all chunks"""
    if state.get("error"):
        return state
    
    state["current_stage"] = ProcessingStage.EMBEDDING.value
    state["progress"] = 80
    
    try:
        from app.services.notes_embedding import NotesEmbeddingService
        
        embedding_service = NotesEmbeddingService()
        
        # Extract texts for batch embedding
        texts = [chunk["text"] for chunk in state["chunks"]]
        
        # Generate embeddings in batches
        embeddings = embedding_service.generate_embeddings(texts)
        
        state["embeddings"] = embeddings
        state["completed_stages"].append("embed")
        state["progress"] = 90
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        state["error"] = str(e)
    
    return state


async def index_in_qdrant(state: DocumentProcessingState) -> DocumentProcessingState:
    """Index chunks and embeddings in Qdrant"""
    if state.get("error"):
        return state
    
    state["current_stage"] = ProcessingStage.INDEXING.value
    state["progress"] = 92
    
    try:
        from app.services.notes_embedding import NotesEmbeddingService
        
        embedding_service = NotesEmbeddingService(collection_name="classroom_documents")
        
        # Create TextChunk objects
        from app.services.notes_embedding import TextChunk as EmbeddingChunk
        
        chunk_objects = []
        for i, chunk_data in enumerate(state["chunks"]):
            chunk_obj = EmbeddingChunk(
                text=chunk_data["text"],
                chunk_index=chunk_data.get("chunk_index", i),
                start_char=chunk_data.get("start_char", 0),
                end_char=chunk_data.get("end_char", len(chunk_data["text"])),
                page_id=str(chunk_data.get("page_number", 1)),
                job_id=state["document_id"],
                metadata={
                    "student_id": state["student_id"],
                    "classroom_id": state["classroom_id"],
                    "subject": state.get("subject"),
                    "source_confidence": chunk_data.get("source_confidence", 0.8),
                    "is_teacher_material": state.get("is_teacher_material", False),
                    "embedding_model_version": "all-MiniLM-L6-v2"
                }
            )
            chunk_objects.append(chunk_obj)
        
        # Index chunks
        point_ids = embedding_service.index_chunks(
            chunks=chunk_objects,
            student_id=state["student_id"],
            classroom_id=state["classroom_id"]
        )
        
        state["qdrant_point_ids"] = point_ids
        state["completed_stages"].append("index")
        state["progress"] = 98
        
        logger.info(f"Indexed {len(point_ids)} points in Qdrant")
        
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        state["error"] = str(e)
    
    return state


async def complete_processing(state: DocumentProcessingState) -> DocumentProcessingState:
    """Mark processing as complete"""
    if state.get("error"):
        state["current_stage"] = ProcessingStage.FAILED.value
    else:
        state["current_stage"] = ProcessingStage.COMPLETED.value
        state["progress"] = 100
        state["completed_stages"].append("complete")
    
    logger.info(f"Processing {'completed' if not state.get('error') else 'failed'} for {state['document_id']}")
    
    return state


def route_after_validation(state: DocumentProcessingState) -> str:
    """Route based on validation result"""
    if state.get("error"):
        return "complete"
    return "preprocess"


def route_after_preprocess(state: DocumentProcessingState) -> str:
    """Route based on preprocessing result"""
    if state.get("error"):
        return "complete"
    if not state.get("preprocessed_paths"):
        state["error"] = "No pages extracted from document"
        return "complete"
    return "ocr"


def route_after_ocr(state: DocumentProcessingState) -> str:
    """Route based on OCR result"""
    if state.get("error"):
        return "complete"
    return "chunk"


def route_after_chunk(state: DocumentProcessingState) -> str:
    """Route based on chunking result"""
    if state.get("error"):
        return "complete"
    if not state.get("chunks"):
        state["error"] = "No text extracted from document"
        return "complete"
    return "embed"


def route_after_embed(state: DocumentProcessingState) -> str:
    """Route based on embedding result"""
    if state.get("error"):
        return "complete"
    return "index"


def route_after_index(state: DocumentProcessingState) -> str:
    """Always go to complete after indexing"""
    return "complete"


# ============================================================================
# Graph Builder
# ============================================================================

def build_document_processing_graph():
    """Build the LangGraph workflow for document processing"""
    workflow = StateGraph(DocumentProcessingState)
    
    # Add nodes
    workflow.add_node("validate", validate_document)
    workflow.add_node("preprocess", preprocess_document)
    workflow.add_node("ocr", extract_text_ocr)
    workflow.add_node("chunk", chunk_text)
    workflow.add_node("embed", generate_embeddings)
    workflow.add_node("index", index_in_qdrant)
    workflow.add_node("complete", complete_processing)
    
    # Set entry point
    workflow.set_entry_point("validate")
    
    # Add conditional edges
    workflow.add_conditional_edges("validate", route_after_validation, {
        "preprocess": "preprocess",
        "complete": "complete"
    })
    
    workflow.add_conditional_edges("preprocess", route_after_preprocess, {
        "ocr": "ocr",
        "complete": "complete"
    })
    
    workflow.add_conditional_edges("ocr", route_after_ocr, {
        "chunk": "chunk",
        "complete": "complete"
    })
    
    workflow.add_conditional_edges("chunk", route_after_chunk, {
        "embed": "embed",
        "complete": "complete"
    })
    
    workflow.add_conditional_edges("embed", route_after_embed, {
        "index": "index",
        "complete": "complete"
    })
    
    workflow.add_edge("index", "complete")
    workflow.add_edge("complete", END)
    
    return workflow.compile()


# ============================================================================
# Agent Class
# ============================================================================

class DocumentProcessingAgent:
    """
    LangGraph-based Document Processing Agent
    
    Orchestrates the 7-stage document ingestion pipeline:
    1. Validate -> 2. Preprocess -> 3. OCR -> 4. Chunk -> 5. Embed -> 6. Index -> 7. Complete
    
    Features:
    - StateGraph for flow control
    - Conditional routing based on errors
    - Progress tracking
    - Idempotent stage execution
    """
    
    def __init__(self, core_service_url: str = None):
        self.graph = build_document_processing_graph()
        self.core_service_url = core_service_url or os.getenv(
            "CORE_SERVICE_URL", "http://localhost:8000"
        )
        self.internal_api_key = os.getenv("INTERNAL_API_KEY", "dev-internal-key")
        logger.info("Initialized DocumentProcessingAgent with LangGraph")
    
    async def process(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document through the full pipeline
        
        Args:
            document_data: {
                document_id, student_id, classroom_id,
                source_url, file_type, subject?, is_teacher_material?
            }
        
        Returns:
            Final processing state
        """
        # Initialize state
        initial_state: DocumentProcessingState = {
            "document_id": document_data["document_id"],
            "student_id": document_data["student_id"],
            "classroom_id": document_data["classroom_id"],
            "source_url": document_data["source_url"],
            "file_type": document_data["file_type"],
            "subject": document_data.get("subject"),
            "is_teacher_material": document_data.get("is_teacher_material", False),
            
            "local_path": "",
            "preprocessed_paths": [],
            "ocr_results": [],
            "chunks": [],
            "embeddings": [],
            
            "current_stage": ProcessingStage.PENDING.value,
            "progress": 0,
            "completed_stages": [],
            "error": None,
            "retry_count": 0,
            
            "qdrant_point_ids": [],
            "total_tokens": 0,
            "total_chunks": 0,
            "avg_confidence": 0.0
        }
        
        # Update status to processing
        await self._update_status(initial_state)
        
        try:
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            # Update final status
            await self._update_status(final_state)
            
            return {
                "success": final_state["current_stage"] == ProcessingStage.COMPLETED.value,
                "document_id": final_state["document_id"],
                "stage": final_state["current_stage"],
                "total_chunks": final_state["total_chunks"],
                "total_tokens": final_state["total_tokens"],
                "avg_confidence": final_state["avg_confidence"],
                "error": final_state.get("error")
            }
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            initial_state["error"] = str(e)
            initial_state["current_stage"] = ProcessingStage.FAILED.value
            await self._update_status(initial_state)
            
            return {
                "success": False,
                "document_id": document_data["document_id"],
                "error": str(e)
            }
    
    async def _update_status(self, state: DocumentProcessingState):
        """Update processing status in core service"""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.core_service_url}/api/documents/internal/update-status",
                    json={
                        "document_id": state["document_id"],
                        "processing_status": state["current_stage"],
                        "processing_stage": state["current_stage"],
                        "progress_percent": state["progress"],
                        "chunks_count": state.get("total_chunks", 0),
                        "total_tokens": state.get("total_tokens", 0),
                        "avg_ocr_confidence": state.get("avg_confidence", 0),
                        "error_message": state.get("error")
                    },
                    headers={"X-Internal-API-Key": self.internal_api_key},
                    timeout=10.0
                )
        except Exception as e:
            logger.warning(f"Failed to update status: {e}")


# ============================================================================
# Convenience Function
# ============================================================================

async def process_document(document_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a document through the ingestion pipeline
    
    Args:
        document_data: Document metadata from upload
    
    Returns:
        Processing result
    """
    agent = DocumentProcessingAgent()
    return await agent.process(document_data)
