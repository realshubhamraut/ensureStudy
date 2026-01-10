"""
Notes Processing Agent - Simplified Version
Converts video/images/documents into a searchable PDF

Pipeline:
1. Extract frames from video (or use uploaded images, or process documents)
2. Enhance images (brightness, contrast)
3. Generate searchable PDF with OCR layer

No text extraction display - just a downloadable searchable PDF.
"""
import logging
import asyncio
from typing import Dict, Any, List, TypedDict, Optional
from pathlib import Path
import os
import httpx
import cv2
import numpy as np
import io
import json

logger = logging.getLogger(__name__)


class NotesProcessingState(TypedDict):
    """State for the processing pipeline"""
    job_id: str
    source_type: str  # 'video', 'images', or 'document'
    source_path: str
    files: List[str]
    
    # Pipeline outputs
    extracted_frames: List[str]
    enhanced_images: List[str]
    pdf_path: Optional[str]
    
    # Status
    status: str
    progress: int  # 0-100
    current_step: str
    error: Optional[str]
    
    # Metadata
    student_id: str
    classroom_id: str


class NotesProcessingAgent:
    """
    Simplified notes processing agent
    
    Pipeline:
    1. Extract frames (video) or load images or process documents (PDF/PPTX/DOCX)
    2. Enhance images  
    3. Generate searchable PDF with embedded OCR
    """
    
    def __init__(
        self,
        core_service_url: str = None,
        internal_api_key: str = None
    ):
        self.core_service_url = core_service_url or os.getenv(
            "CORE_SERVICE_URL", "http://localhost:9000"
        )
        self.internal_api_key = internal_api_key or os.getenv(
            "INTERNAL_API_KEY", "dev-internal-key"
        )
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the simplified processing pipeline
        
        Args:
            state: Initial state with job details
        
        Returns:
            Final state with pdf_path
        """
        # Initialize state
        processing_state: NotesProcessingState = {
            "job_id": state["job_id"],
            "source_type": state["source_type"],
            "source_path": state["source_path"],
            "files": state.get("files", []),
            "extracted_frames": [],
            "enhanced_images": [],
            "pdf_path": None,
            "status": "processing",
            "progress": 0,
            "current_step": "Starting...",
            "error": None,
            "student_id": state.get("student_id", ""),
            "classroom_id": state.get("classroom_id", "")
        }
        
        await self._update_job_status(processing_state)
        
        try:
            # Step 1: Extract frames or load images
            processing_state = await self._extract_frames(processing_state)
            
            # Step 2: Enhance images
            processing_state = await self._enhance_images(processing_state)
            
            # Step 3: OCR and save pages
            processing_state = await self._ocr_and_save_pages(processing_state)
            
            # Step 4: Generate searchable PDF
            processing_state = await self._generate_searchable_pdf(processing_state)
            
            # Done
            processing_state["status"] = "completed"
            processing_state["progress"] = 100
            processing_state["current_step"] = "Complete!"
            await self._update_job_status(processing_state)
            
            logger.info(f"Job {processing_state['job_id']} completed successfully")
            return processing_state
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            processing_state["status"] = "failed"
            processing_state["error"] = str(e)
            await self._update_job_status(processing_state)
            return processing_state
    
    async def _extract_frames(self, state: NotesProcessingState) -> NotesProcessingState:
        """Extract frames from video, use uploaded images, or process documents"""
        state["current_step"] = "Extracting frames..."
        state["progress"] = 10
        await self._update_job_status(state)
        
        output_dir = os.path.join(state["source_path"], "frames")
        os.makedirs(output_dir, exist_ok=True)
        
        if state["source_type"] == "video":
            # Find video file
            video_file = None
            for f in state["files"]:
                if f.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
                    video_file = os.path.join(state["source_path"], f)
                    break
            
            if not video_file or not os.path.exists(video_file):
                raise ValueError("No video file found")
            
            # Extract frames
            frames = self._extract_video_frames(video_file, output_dir)
            state["extracted_frames"] = frames
            logger.info(f"Extracted {len(frames)} frames from video")
        
        # INTEGRATION NOTE: Document processing uses LangGraph DocumentProcessingAgent
        # Full 7-stage pipeline: Validate → Preprocess → OCR → Chunk → Embed → Index → Complete
        # This enables RAG search in AI Tutor via Qdrant vector database
        elif state["source_type"] == "document":
            # Find document file
            doc_file = None
            for f in state["files"]:
                if f.lower().endswith(('.pdf', '.pptx', '.docx', '.doc')):
                    doc_file = f
                    break
            
            if not doc_file or not os.path.exists(doc_file):
                # Try finding in source_path directory
                for f in os.listdir(state["source_path"]):
                    if f.lower().endswith(('.pdf', '.pptx', '.docx', '.doc')):
                        doc_file = os.path.join(state["source_path"], f)
                        break
            
            if not doc_file:
                raise ValueError("No document file found")
            
            ext = doc_file.rsplit('.', 1)[1].lower() if '.' in doc_file else ''
            state["current_step"] = f"Processing {ext.upper()} with LangGraph agent..."
            await self._update_job_status(state)
            
            # Use LangGraph DocumentProcessingAgent for full pipeline
            # This provides: chunking, embeddings, Qdrant indexing, RAG integration
            from app.agents.document_agent import DocumentProcessingAgent
            
            doc_agent = DocumentProcessingAgent()
            result = await doc_agent.process({
                "document_id": state["job_id"],
                "student_id": state.get("student_id", ""),
                "classroom_id": state.get("classroom_id", ""),
                "source_url": doc_file,
                "file_type": ext,
                "subject": None,
                "is_teacher_material": False
            })
            
            logger.info(f"LangGraph DocumentProcessingAgent result: {result}")
            
            # If agent succeeded, we still need to generate a PDF for viewing
            # Use the preprocessor to get images for PDF generation
            from app.services.document_preprocessor import preprocess_document
            
            preprocess_result = preprocess_document(doc_file, ext)
            
            # Copy extracted images to frames directory
            import shutil
            extracted_images = []
            for i, img_path in enumerate(preprocess_result.get("image_paths", [])):
                if os.path.exists(img_path):
                    dest_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
                    shutil.copy(img_path, dest_path)
                    extracted_images.append(dest_path)
            
            state["extracted_frames"] = extracted_images
            logger.info(f"Document indexed in Qdrant, {len(extracted_images)} pages for PDF")
            
        else:
            # Use uploaded images
            image_paths = []
            for f in state["files"]:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    full_path = os.path.join(state["source_path"], f)
                    if os.path.exists(full_path):
                        image_paths.append(full_path)
            
            state["extracted_frames"] = image_paths
            logger.info(f"Using {len(image_paths)} uploaded images")
        
        state["progress"] = 25
        return state
    
    def _extract_video_frames(
        self, 
        video_path: str, 
        output_dir: str,
        frame_interval: float = 1.0,
        max_frames: int = 30,
        blur_threshold: float = 80.0
    ) -> List[str]:
        """Extract best frames from video"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_skip = int(fps * frame_interval)
        saved_frames = []
        frame_idx = 0
        
        while len(saved_frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Check if frame is sharp enough
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if blur_score > blur_threshold:
                # Save frame
                frame_path = os.path.join(output_dir, f"frame_{len(saved_frames):03d}.png")
                cv2.imwrite(frame_path, frame)
                saved_frames.append(frame_path)
            
            frame_idx += frame_skip
        
        cap.release()
        return saved_frames
    
    async def _enhance_images(self, state: NotesProcessingState) -> NotesProcessingState:
        """
        MINIMAL image processing - preserve original handwriting quality
        
        For handwritten notes, less processing = better quality
        We only:
        1. Resize to standard A4 dimensions (if oversized)
        2. Very slight brightness adjustment if too dark
        """
        state["current_step"] = "Preparing images..."
        state["progress"] = 40
        await self._update_job_status(state)
        
        output_dir = os.path.join(state["source_path"], "enhanced")
        os.makedirs(output_dir, exist_ok=True)
        
        enhanced_paths = []
        
        # A4 at 200 DPI
        A4_WIDTH = 1654
        A4_HEIGHT = 2339
        
        for i, img_path in enumerate(state["extracted_frames"]):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # === DISABLED ALL PROCESSING (User Request) ===
                # Just save the original image to preserve maximum resolution/quality
                # for the VLM/OCR model. Downscaling was causing hallucinations.
                
                output_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
                cv2.imwrite(output_path, img) # Just save raw read
                enhanced_paths.append(output_path)
                logger.info(f"Processed image {i+1}: Original quality preserved")
                
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                # On error, just copy original
                import shutil
                output_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
                shutil.copy(img_path, output_path)
                enhanced_paths.append(output_path)
        
        state["enhanced_images"] = enhanced_paths
        state["progress"] = 50
        return state
    
    async def _ocr_and_save_pages(self, state: NotesProcessingState) -> NotesProcessingState:
        """
        Run OCR on each image and save pages to database.
        
        Uses HuggingFace Inference API with Nanonets-OCR2-3B or olmOCR-7B
        for high-quality handwritten text recognition.
        
        Requires HUGGINGFACE_API_KEY environment variable to be set.
        """
        state["current_step"] = "Extracting text from pages..."
        state["progress"] = 55
        await self._update_job_status(state)
        
        # Import the API-based OCR service
        from app.services.nanonets_ocr import get_ocr_service
        from PIL import Image
        
        ocr_service = get_ocr_service()
        
        # Check if API key is configured
        if not ocr_service.api_key:
            logger.warning("HUGGINGFACE_API_KEY not set - OCR will be skipped")
            state["error"] = "OCR requires HUGGINGFACE_API_KEY to be set"
            # Still continue to save pages without OCR text
        else:
            logger.info(f"Using HuggingFace API with model: {ocr_service.model_id}")
        
        total_confidence = 0
        page_count = 0
        
        for i, img_path in enumerate(state["enhanced_images"]):
            try:
                extracted_text = ""
                confidence = 0.0
                
                if os.path.exists(img_path):
                    # Load the image
                    img = Image.open(img_path).convert("RGB")
                    
                    logger.info(f"Page {i+1}: Running OCR on {img.size[0]}x{img.size[1]} image...")
                    
                    # Extract text using HuggingFace API
                    # The API takes the FULL page image - no segmentation needed!
                    if ocr_service.api_key:
                        extracted_text, confidence = await ocr_service.extract_text(
                            img,
                            prompt="Perform OCR on this handwritten document. Extract all text exactly as written, preserving line breaks.",
                        )
                        logger.info(f"Page {i+1}: Extracted {len(extracted_text)} chars, confidence: {confidence:.2f}")
                    else:
                        logger.warning(f"Page {i+1}: Skipped OCR (no API key)")
                        confidence = 0.0
                
                # Calculate relative image URL for the page
                enhanced_url = f"/api/notes/images/{state['job_id']}/enhanced/page_{i+1:03d}.png"
                
                # Quality gating based on confidence
                if confidence < 0.60:
                    page_status = "needs_review"
                elif confidence < 0.75:
                    page_status = "low_confidence" 
                else:
                    page_status = "ocr_done"
                
                logger.info(f"Page {i+1}: confidence={confidence:.2f}, status={page_status}")
                
                # Save page to database via internal API
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{self.core_service_url}/api/notes/internal/add-page",
                        json={
                            "job_id": state["job_id"],
                            "page_number": i + 1,
                            "enhanced_image_url": enhanced_url,
                            "extracted_text": extracted_text,
                            "confidence_score": confidence,
                            "status": page_status
                        },
                        headers={"X-Internal-API-Key": self.internal_api_key},
                        timeout=30.0
                    )
                
                total_confidence += confidence
                page_count += 1
                
                # Update progress
                progress = 55 + int((i + 1) / len(state["enhanced_images"]) * 10)
                state["progress"] = progress
                
            except Exception as e:
                logger.error(f"Error processing page {i+1}: {e}")
        
        # Calculate average confidence
        if page_count > 0:
            state["avg_confidence"] = total_confidence / page_count
        
        state["progress"] = 65
        logger.info(f"Saved {page_count} pages with OCR text")
        return state
    
    async def _generate_searchable_pdf(self, state: NotesProcessingState) -> NotesProcessingState:
        """Generate searchable PDF with OCR text layer"""
        state["current_step"] = "Generating searchable PDF..."
        state["progress"] = 70
        await self._update_job_status(state)
        
        if not state["enhanced_images"]:
            raise ValueError("No enhanced images to process")
        
        # Import the searchable PDF generator
        from app.services.searchable_pdf import generate_searchable_pdf
        
        output_pdf = os.path.join(state["source_path"], "notes.pdf")
        
        def progress_callback(progress, message):
            # Update progress (70-95 range for PDF generation)
            new_progress = int(70 + progress * 25)
            logger.info(f"PDF Progress: {message}")
        
        pdf_path = generate_searchable_pdf(
            state["enhanced_images"],
            output_pdf,
            progress_callback=progress_callback
        )
        
        state["pdf_path"] = pdf_path
        state["progress"] = 95
        
        logger.info(f"Generated searchable PDF: {pdf_path}")
        return state
    
    async def _update_job_status(self, state: NotesProcessingState):
        """Update job status in core service"""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.core_service_url}/api/notes/internal/update-job",
                    json={
                        "job_id": state["job_id"],
                        "status": state["status"],
                        "progress_percent": state["progress"],
                        "current_step": state["current_step"],
                        "pdf_path": state.get("pdf_path"),
                        "error_message": state.get("error")
                    },
                    headers={"X-Internal-API-Key": self.internal_api_key},
                    timeout=10.0
                )
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")


# Convenience function for API to call
async def process_notes_job(job_data: dict) -> dict:
    """
    Process a notes digitization job
    
    Args:
        job_data: Dict with job_id, source_type, source_path, files, etc.
    
    Returns:
        Final state dict with pdf_path
    """
    agent = NotesProcessingAgent()
    result = await agent.process(job_data)
    return result

