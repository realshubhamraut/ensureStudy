"""
OCR Service - AI-Powered Handwriting Recognition
Uses Hugging Face API for fast cloud-based OCR with local fallback

Priority:
1. Hugging Face Inference API (TrOCR) - Fast, cloud-based
2. Local TrOCR model - If no API key or offline
3. Tesseract - Final fallback
"""
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import os
import base64
import io
import time

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result of OCR processing"""
    text: str
    confidence: float  # 0-1
    lines: List[str]
    processing_time: float  # seconds
    model_used: str


class OCRService:
    """
    AI-Powered OCR using Hugging Face models
    
    Uses the Hugging Face Inference API for TrOCR handwriting recognition.
    Falls back to local model or Tesseract if API unavailable.
    """
    
    # Hugging Face model endpoints
    HF_OCR_MODEL = "microsoft/trocr-base-handwritten"
    HF_API_URL = "https://api-inference.huggingface.co/models"
    
    def __init__(
        self,
        use_api: bool = True,
        api_key: Optional[str] = None,
        fallback_to_local: bool = True,
        line_height_threshold: int = 50
    ):
        self.use_api = use_api
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.fallback_to_local = fallback_to_local
        self.line_height_threshold = line_height_threshold
        
        # Local model (lazy loaded)
        self.processor = None
        self.model = None
        self.tesseract_available = False
        
        # Check available backends
        self._check_backends()
    
    def _check_backends(self):
        """Check which OCR backends are available"""
        self.api_available = bool(self.api_key) and self.use_api
        
        if self.api_available:
            logger.info("Hugging Face API available for OCR")
        else:
            logger.info("No HF API key found, will use local models")
        
        # Check Tesseract
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("Tesseract OCR available as fallback")
        except:
            logger.warning("Tesseract not available")
    
    def _load_local_model(self):
        """Load local TrOCR model (lazy loading)"""
        if self.model is not None:
            return True
        
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            logger.info(f"Loading local TrOCR model: {self.HF_OCR_MODEL}")
            self.processor = TrOCRProcessor.from_pretrained(self.HF_OCR_MODEL)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.HF_OCR_MODEL)
            
            # Use MPS for Apple Silicon
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Local TrOCR loaded on {self.device}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load local TrOCR: {e}")
            return False
    
    def extract_text(
        self,
        image: np.ndarray,
        preprocess: bool = True
    ) -> OCRResult:
        """
        Extract text from an image using AI
        
        Args:
            image: BGR image array
            preprocess: Whether to apply preprocessing
        
        Returns:
            OCRResult with extracted text and confidence
        """
        start_time = time.time()
        
        # Preprocess
        if preprocess:
            image = self._preprocess(image)
        
        result = None
        
        # Try Hugging Face API first (fastest)
        if self.api_available:
            result = self._extract_with_hf_api(image)
            if result and result.text.strip():
                result.processing_time = time.time() - start_time
                return result
        
        # Fallback to local TrOCR
        if self.fallback_to_local:
            if self._load_local_model():
                result = self._extract_with_local_trocr(image)
                if result and result.text.strip():
                    result.processing_time = time.time() - start_time
                    return result
        
        # Final fallback: Tesseract
        if self.tesseract_available:
            result = self._extract_with_tesseract(image)
            result.processing_time = time.time() - start_time
            return result
        
        # No OCR available
        return OCRResult(
            text="[OCR unavailable - set HUGGINGFACE_API_KEY or install Tesseract]",
            confidence=0.0,
            lines=[],
            processing_time=time.time() - start_time,
            model_used="none"
        )
    
    def _extract_with_hf_api(self, image: np.ndarray) -> Optional[OCRResult]:
        """Extract text using Hugging Face Inference API"""
        try:
            import requests
            from PIL import Image
            
            # Detect lines for better accuracy
            line_images = self._detect_lines(image)
            if not line_images:
                line_images = [image]
            
            extracted_lines = []
            
            for line_img in line_images:
                # Convert to PNG bytes
                pil_img = Image.fromarray(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
                buffer = io.BytesIO()
                pil_img.save(buffer, format="PNG")
                img_bytes = buffer.getvalue()
                
                # Call API
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.post(
                    f"{self.HF_API_URL}/{self.HF_OCR_MODEL}",
                    headers=headers,
                    data=img_bytes,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        text = result[0].get("generated_text", "")
                        if text.strip():
                            extracted_lines.append(text.strip())
                elif response.status_code == 503:
                    # Model loading, wait and retry once
                    logger.info("HF model loading, waiting...")
                    import time
                    time.sleep(20)
                    response = requests.post(
                        f"{self.HF_API_URL}/{self.HF_OCR_MODEL}",
                        headers=headers,
                        data=img_bytes,
                        timeout=60
                    )
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            text = result[0].get("generated_text", "")
                            if text.strip():
                                extracted_lines.append(text.strip())
                else:
                    logger.warning(f"HF API error: {response.status_code} - {response.text}")
                    return None
            
            full_text = "\n".join(extracted_lines)
            
            return OCRResult(
                text=full_text,
                confidence=0.90,  # API doesn't provide confidence, estimate high
                lines=extracted_lines,
                processing_time=0.0,
                model_used="HuggingFace API (TrOCR)"
            )
            
        except Exception as e:
            logger.warning(f"HF API error: {e}")
            return None
    
    def _extract_with_local_trocr(self, image: np.ndarray) -> Optional[OCRResult]:
        """Extract text using local TrOCR model"""
        try:
            import torch
            from PIL import Image
            
            line_images = self._detect_lines(image)
            if not line_images:
                line_images = [image]
            
            extracted_lines = []
            
            with torch.no_grad():
                for line_img in line_images:
                    pil_image = Image.fromarray(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
                    
                    pixel_values = self.processor(
                        images=pil_image,
                        return_tensors="pt"
                    ).pixel_values.to(self.device)
                    
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_length=128,
                        num_beams=4,
                        early_stopping=True
                    )
                    
                    text = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )[0]
                    
                    if text.strip():
                        extracted_lines.append(text.strip())
            
            full_text = "\n".join(extracted_lines)
            
            return OCRResult(
                text=full_text,
                confidence=0.85,
                lines=extracted_lines,
                processing_time=0.0,
                model_used="Local TrOCR"
            )
            
        except Exception as e:
            logger.warning(f"Local TrOCR error: {e}")
            return None
    
    def _extract_with_tesseract(self, image: np.ndarray) -> OCRResult:
        """Extract text using Tesseract (fallback)"""
        import pytesseract
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Get OCR data
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        
        lines = []
        confidences = []
        current_line = []
        
        for i, text in enumerate(data['text']):
            if text.strip():
                current_line.append(text)
                conf = data['conf'][i]
                if conf > 0:
                    confidences.append(conf / 100.0)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = []
        
        if current_line:
            lines.append(" ".join(current_line))
        
        full_text = "\n".join(lines)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            lines=lines,
            processing_time=0.0,
            model_used="Tesseract"
        )
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if too large
        max_dim = 1200
        h, w = gray.shape
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Light denoising
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Convert back to BGR
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    def _detect_lines(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect individual text lines in image with improved filtering"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        height, width = gray.shape
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection
        horizontal_proj = np.sum(binary, axis=1)
        
        # IMPROVED: Use 5% threshold (was 15%) to capture lighter text
        threshold = np.max(horizontal_proj) * 0.05
        in_line = False
        line_regions = []
        start = 0
        
        for i, val in enumerate(horizontal_proj):
            if val > threshold and not in_line:
                in_line = True
                start = i
            elif val <= threshold and in_line:
                in_line = False
                if i - start > self.line_height_threshold // 2:
                    line_regions.append((max(0, start - 10), min(len(horizontal_proj), i + 10)))
        
        if in_line and len(horizontal_proj) - start > self.line_height_threshold // 2:
            line_regions.append((max(0, start - 10), len(horizontal_proj)))
        
        # Merge nearby lines
        merged_regions = []
        for region in line_regions:
            if merged_regions and region[0] - merged_regions[-1][1] < self.line_height_threshold:
                merged_regions[-1] = (merged_regions[-1][0], region[1])
            else:
                merged_regions.append(region)
        
        # IMPROVED: Filter by aspect ratio and horizontal extent
        filtered_regions = []
        for start, end in merged_regions:
            line_binary = binary[start:end, :]
            horiz_extent = np.sum(line_binary, axis=0) > 0
            extent_ratio = np.sum(horiz_extent) / width
            
            # Skip lines that don't span enough width (likely decorations)
            # Relaxed to 5% (was 20%) to catch short text
            if extent_ratio < 0.05:
                continue
            
            # Skip lines with poor aspect ratio (width should be > 1.5x height)
            # Relaxed from 3x to catch blocky text
            line_height = end - start
            line_width = np.sum(horiz_extent)
            aspect_ratio = line_width / line_height if line_height > 0 else 0
            if aspect_ratio < 1.5:
                continue
            
            filtered_regions.append((start, end))
        
        # Extract line images
        line_images = []
        for start, end in filtered_regions:
            line_img = image[start:end, :]
            if line_img.size > 0:
                line_images.append(line_img)
        
        return line_images


# Convenience functions
def extract_text_from_image(image_path: str) -> OCRResult:
    """Extract text from an image file"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    ocr = OCRService()
    return ocr.extract_text(image)


def batch_extract_text(
    image_paths: List[str],
    progress_callback: Optional[callable] = None
) -> List[Tuple[str, OCRResult]]:
    """Extract text from multiple images"""
    ocr = OCRService()
    results = []
    
    for i, path in enumerate(image_paths):
        if progress_callback:
            progress_callback(i / len(image_paths), f"OCR on image {i+1}/{len(image_paths)}")
        
        try:
            image = cv2.imread(path)
            if image is not None:
                result = ocr.extract_text(image)
                results.append((path, result))
            else:
                logger.warning(f"Could not read: {path}")
        except Exception as e:
            logger.error(f"OCR error on {path}: {e}")
    
    return results
