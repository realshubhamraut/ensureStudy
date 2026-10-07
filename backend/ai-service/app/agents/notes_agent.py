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
                
                # === MINIMAL PROCESSING ONLY ===
                
                # 1. Resize if too large (preserve aspect ratio)
                h, w = img.shape[:2]
                if w > A4_WIDTH or h > A4_HEIGHT:
                    scale = min(A4_WIDTH / w, A4_HEIGHT / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # 2. Only adjust brightness if image is too dark
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(gray)
                
                if avg_brightness < 100:  # Too dark
                    # Slight brightness boost
                    img = cv2.convertScaleAbs(img, alpha=1.0, beta=30)
                elif avg_brightness > 240:  # Too bright/washed out
                    # Slight contrast boost
                    img = cv2.convertScaleAbs(img, alpha=1.1, beta=-10)
                
                # NO CLAHE, NO DENOISE, NO SHARPEN - preserve original!
                
                # Save at maximum quality
                output_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
                cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                enhanced_paths.append(output_path)
                
                logger.info(f"Processed image {i+1}: brightness={avg_brightness:.0f}")
                
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
        """Run OCR on each image and save pages to database"""
        state["current_step"] = "Extracting text from pages..."
        state["progress"] = 55
        await self._update_job_status(state)
        
        # Try TrOCR first (better for handwritten text), fallback to Tesseract
        TROCR_AVAILABLE = False
        trocr_processor = None
        trocr_model = None
        
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            from PIL import Image
            import torch
            
            # Load TrOCR model for handwritten text
            logger.info("Loading TrOCR handwritten model...")
            trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
            TROCR_AVAILABLE = True
            logger.info("TrOCR model loaded successfully")
        except Exception as e:
            logger.warning(f"TrOCR not available ({e}), falling back to Tesseract")
        
        # Fallback to Tesseract
        TESSERACT_AVAILABLE = False
        if not TROCR_AVAILABLE:
            try:
                import pytesseract
                TESSERACT_AVAILABLE = True
            except ImportError:
                logger.warning("Neither TrOCR nor Tesseract available")
        
        from PIL import Image
        
        total_confidence = 0
        page_count = 0
        
        for i, img_path in enumerate(state["enhanced_images"]):
            try:
                # Extract text with OCR
                extracted_text = ""
                confidence = 0.0
                
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    
                    if TROCR_AVAILABLE:
                        try:
                            # Use TrOCR for handwritten text recognition
                            # IMPORTANT: Detect actual text lines using horizontal projection
                            # instead of fixed-height strips (which causes hallucinations)
                            import cv2 as cv2_ocr
                            
                            # Convert PIL to numpy for line detection
                            img_np = np.array(img)
                            gray = cv2_ocr.cvtColor(img_np, cv2_ocr.COLOR_RGB2GRAY)
                            
                            # Binary threshold for line detection
                            _, binary = cv2_ocr.threshold(gray, 0, 255, cv2_ocr.THRESH_BINARY_INV + cv2_ocr.THRESH_OTSU)
                            
                            # Horizontal projection to find text lines
                            horizontal_proj = np.sum(binary, axis=1)
                            
                            # Find line boundaries using projection profile
                            threshold = np.max(horizontal_proj) * 0.05  # 5% of max = text present
                            in_line = False
                            line_regions = []
                            start = 0
                            min_line_height = 20  # Minimum pixels for a valid line
                            
                            for y_idx, val in enumerate(horizontal_proj):
                                if val > threshold and not in_line:
                                    in_line = True
                                    start = y_idx
                                elif val <= threshold and in_line:
                                    in_line = False
                                    if y_idx - start > min_line_height:
                                        # Add padding around the line
                                        line_start = max(0, start - 5)
                                        line_end = min(len(horizontal_proj), y_idx + 5)
                                        line_regions.append((line_start, line_end))
                            
                            # Don't forget the last line if still in_line
                            if in_line and len(horizontal_proj) - start > min_line_height:
                                line_regions.append((max(0, start - 5), len(horizontal_proj)))
                            
                            # Merge nearby lines (within 15px of each other)
                            merged_regions = []
                            for region in line_regions:
                                if merged_regions and region[0] - merged_regions[-1][1] < 15:
                                    merged_regions[-1] = (merged_regions[-1][0], region[1])
                                else:
                                    merged_regions.append(region)
                            
                            logger.info(f"Page {i+1}: Detected {len(merged_regions)} text lines")
                            
                            # Process each detected line with TrOCR
                            all_text = []
                            width, height = img.size
                            
                            for line_start, line_end in merged_regions:
                                # Crop the detected line
                                line_img = img.crop((0, line_start, width, line_end))
                                
                                # Check if line has content (not blank)
                                line_np = np.array(line_img.convert('L'))
                                if np.mean(line_np) > 250:  # Nearly white = blank
                                    continue
                                
                                # Process with TrOCR
                                pixel_values = trocr_processor(images=line_img, return_tensors="pt").pixel_values
                                
                                with torch.no_grad():
                                    generated_ids = trocr_model.generate(
                                        pixel_values, 
                                        max_length=256,  # Allow longer output
                                        num_beams=4,     # Beam search for better quality
                                        early_stopping=True
                                    )
                                
                                line_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                                if line_text.strip():
                                    all_text.append(line_text.strip())
                            
                            extracted_text = '\n'.join(all_text)
                            confidence = 0.85  # TrOCR typically has high accuracy on handwriting
                            
                            logger.info(f"Page {i+1}: TrOCR extracted {len(extracted_text)} chars from {len(merged_regions)} lines")
                        except Exception as ocr_error:
                            logger.warning(f"TrOCR failed for page {i+1}: {ocr_error}")
                            # Try Tesseract as fallback
                            if TESSERACT_AVAILABLE:
                                import pytesseract
                                extracted_text = pytesseract.image_to_string(img)
                                confidence = 0.4
                    
                    elif TESSERACT_AVAILABLE:
                        import pytesseract
                        # Use Tesseract as fallback
                        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                        
                        texts = []
                        confidences = []
                        for j, text in enumerate(ocr_data['text']):
                            if text.strip():
                                texts.append(text)
                                conf = int(ocr_data['conf'][j])
                                if conf > 0:
                                    confidences.append(conf)
                        
                        extracted_text = ' '.join(texts)
                        if confidences:
                            confidence = sum(confidences) / len(confidences) / 100
                        
                        logger.info(f"Page {i+1}: Tesseract extracted {len(extracted_text)} chars, confidence: {confidence:.2f}")
                
                # Calculate relative image URL for the page
                # The enhanced images are stored relative to source_path
                enhanced_url = f"/api/notes/images/{state['job_id']}/enhanced/page_{i+1:03d}.png"
                
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
                            "status": "ocr_done"
                        },
                        headers={"X-Internal-API-Key": self.internal_api_key},
                        timeout=10.0
                    )
                
                total_confidence += confidence
                page_count += 1
                
                # Update progress
                progress = 55 + int((i + 1) / len(state["enhanced_images"]) * 10)
                state["progress"] = progress
                
            except Exception as e:
                logger.error(f"Error saving page {i+1}: {e}")
        
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

