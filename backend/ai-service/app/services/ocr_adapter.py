"""
OCR Adapter - Config-Driven OCR Model Selection

Provides an abstract interface for different OCR backends:
- TrOCRAdapter: Local TrOCR model (default)
- SageMakerAdapter: SageMaker Nanonets-OCR2-3B 
- EasyOCRAdapter: EasyOCR multi-language fallback

Each adapter returns standardized OCRLine objects with bounding boxes.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
import time
import os

logger = logging.getLogger(__name__)


@dataclass
class OCRLine:
    """Single OCR-detected text line with bounding box"""
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float


@dataclass
class OCRPageResult:
    """Full page OCR result with all detected lines"""
    lines: List[OCRLine]
    full_text: str
    avg_confidence: float
    model_used: str
    processing_time_ms: int


class OCRAdapter(ABC):
    """Abstract base class for OCR adapters"""
    
    @abstractmethod
    def extract_lines(self, image_bytes: bytes) -> OCRPageResult:
        """
        Extract text lines with bounding boxes from an image.
        
        Args:
            image_bytes: Raw image bytes (PNG/JPEG)
            
        Returns:
            OCRPageResult with lines, text, and metadata
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier string"""
        pass


class TrOCRAdapter(OCRAdapter):
    """
    Local TrOCR model adapter.
    
    Uses microsoft/trocr-base-handwritten or trocr-large-handwritten
    for handwritten text recognition.
    """
    
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model_name = f"microsoft/trocr-{model_size}-handwritten"
        self.processor = None
        self.model = None
        self.device = "cpu"
        self._loaded = False
    
    def _load_model(self):
        """Lazy load the TrOCR model"""
        if self._loaded:
            return
        
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            logger.info(f"Loading TrOCR model: {self.model_name}")
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            
            # Select best available device
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
            
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"TrOCR loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load TrOCR: {e}")
            raise
    
    def _detect_text_lines(self, image_np) -> List[Tuple[int, int, int, int]]:
        """
        Detect text line bounding boxes using horizontal projection.
        
        Returns list of (y_start, y_end, x_start, x_end) tuples.
        """
        import cv2
        import numpy as np
        
        # Convert to grayscale
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        height, width = gray.shape
        
        # Binary threshold (inverse for text detection)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection profile
        horizontal_proj = np.sum(binary, axis=1)
        
        # IMPROVED: Use 15% threshold instead of 5% to reduce noise
        threshold = np.max(horizontal_proj) * 0.15
        
        # Find line boundaries
        in_line = False
        line_regions = []
        start = 0
        min_line_height = 20  # Minimum pixels for valid line
        
        for y_idx, val in enumerate(horizontal_proj):
            if val > threshold and not in_line:
                in_line = True
                start = y_idx
            elif val <= threshold and in_line:
                in_line = False
                if y_idx - start > min_line_height:
                    line_regions.append((start, y_idx))
        
        # Handle last line
        if in_line and height - start > min_line_height:
            line_regions.append((start, height))
        
        # Merge nearby lines (within 15px)
        merged_regions = []
        for region in line_regions:
            if merged_regions and region[0] - merged_regions[-1][1] < 15:
                merged_regions[-1] = (merged_regions[-1][0], region[1])
            else:
                merged_regions.append(region)
        
        # IMPROVED: Filter lines by horizontal extent (must span >20% of width)
        # and aspect ratio (text lines are wide, not square)
        filtered_regions = []
        for line_start, line_end in merged_regions:
            line_binary = binary[line_start:line_end, :]
            
            # Check horizontal extent
            horiz_extent = np.sum(line_binary, axis=0) > 0
            extent_ratio = np.sum(horiz_extent) / width
            
            if extent_ratio < 0.20:
                # Line doesn't span enough width - likely decoration
                logger.debug(f"Skipping line y={line_start}-{line_end}: extent={extent_ratio:.2f}")
                continue
            
            # Check aspect ratio (width should be > 3x height for text)
            line_height = line_end - line_start
            line_width = np.sum(horiz_extent)
            aspect_ratio = line_width / line_height if line_height > 0 else 0
            
            if aspect_ratio < 3:
                # Likely a decoration (circle, box, etc.) not text
                logger.debug(f"Skipping line y={line_start}-{line_end}: aspect={aspect_ratio:.2f}")
                continue
            
            # Find actual x bounds of content
            x_indices = np.where(horiz_extent)[0]
            if len(x_indices) > 0:
                x_start = max(0, x_indices[0] - 5)
                x_end = min(width, x_indices[-1] + 5)
            else:
                x_start, x_end = 0, width
            
            # Add padding
            y_start_padded = max(0, line_start - 5)
            y_end_padded = min(height, line_end + 5)
            
            filtered_regions.append((y_start_padded, y_end_padded, x_start, x_end))
        
        return filtered_regions
    
    def _compute_confidence(self, outputs, scores) -> float:
        """
        Compute actual confidence from TrOCR decoder scores.
        
        Returns average token probability as confidence score.
        """
        import torch
        
        if not scores or len(scores) == 0:
            return 0.5  # Default if no scores available
        
        try:
            probs = []
            for score in scores:
                # Get softmax probabilities and take the max (selected token prob)
                token_probs = torch.softmax(score, dim=-1)
                max_prob = token_probs.max().item()
                probs.append(max_prob)
            
            if probs:
                return sum(probs) / len(probs)
            return 0.5
            
        except Exception as e:
            logger.warning(f"Error computing confidence: {e}")
            return 0.5
    
    def extract_lines(self, image_bytes: bytes) -> OCRPageResult:
        """Extract text with bounding boxes from image bytes"""
        import torch
        import numpy as np
        from PIL import Image
        import io
        
        start_time = time.time()
        self._load_model()
        
        # Decode image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(pil_image)
        width, height = pil_image.size
        
        # Detect text line regions
        line_regions = self._detect_text_lines(image_np)
        logger.info(f"Detected {len(line_regions)} text lines")
        
        ocr_lines = []
        all_confidences = []
        
        with torch.no_grad():
            for y_start, y_end, x_start, x_end in line_regions:
                # Crop line region
                line_img = pil_image.crop((x_start, y_start, x_end, y_end))
                
                # Check if line has content (not blank)
                line_np = np.array(line_img.convert('L'))
                if np.mean(line_np) > 250:  # Nearly white = blank
                    continue
                
                # Process with TrOCR
                pixel_values = self.processor(
                    images=line_img, 
                    return_tensors="pt"
                ).pixel_values.to(self.device)
                
                # Generate with score output for confidence calculation
                outputs = self.model.generate(
                    pixel_values,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # Decode text
                text = self.processor.batch_decode(
                    outputs.sequences, 
                    skip_special_tokens=True
                )[0].strip()
                
                if not text:
                    continue
                
                # Compute real confidence from decoder scores
                confidence = self._compute_confidence(outputs, outputs.scores)
                all_confidences.append(confidence)
                
                # Create OCRLine with bbox
                bbox = (x_start, y_start, x_end - x_start, y_end - y_start)
                ocr_lines.append(OCRLine(
                    text=text,
                    bbox=bbox,
                    confidence=confidence
                ))
        
        # Build full text
        full_text = '\n'.join(line.text for line in ocr_lines)
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        processing_time = int((time.time() - start_time) * 1000)
        
        return OCRPageResult(
            lines=ocr_lines,
            full_text=full_text,
            avg_confidence=avg_confidence,
            model_used=self.model_name,
            processing_time_ms=processing_time
        )
    
    def get_model_name(self) -> str:
        return self.model_name


class SageMakerAdapter(OCRAdapter):
    """
    SageMaker Nanonets-OCR2-3B adapter.
    
    Uses AWS SageMaker serverless endpoint for high-quality OCR.
    Requires SAGEMAKER_OCR_ENABLED=true environment variable.
    """
    
    def __init__(self):
        self.enabled = os.getenv("SAGEMAKER_OCR_ENABLED", "false").lower() == "true"
        self.endpoint = os.getenv("SAGEMAKER_OCR_ENDPOINT", "ensurestudy-ocr-serverless")
    
    def extract_lines(self, image_bytes: bytes) -> OCRPageResult:
        """Extract text using SageMaker endpoint"""
        from app.services.sagemaker_ocr import get_sagemaker_ocr
        import asyncio
        
        start_time = time.time()
        
        # Get existing SageMaker service
        service = get_sagemaker_ocr()
        
        # Run async extraction
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(service.extract_text(image_bytes))
        finally:
            loop.close()
        
        # Convert SageMakerOCRResult to our format
        ocr_lines = []
        for layout_elem in result.layout:
            if layout_elem.bbox and len(layout_elem.bbox) == 4:
                bbox = tuple(layout_elem.bbox)
            else:
                bbox = (0, 0, 0, 0)
            
            ocr_lines.append(OCRLine(
                text=layout_elem.text,
                bbox=bbox,
                confidence=layout_elem.confidence
            ))
        
        # If no layout elements, create single line from full text
        if not ocr_lines and result.text:
            ocr_lines.append(OCRLine(
                text=result.text,
                bbox=(0, 0, 0, 0),
                confidence=result.confidence_overall
            ))
        
        return OCRPageResult(
            lines=ocr_lines,
            full_text=result.text,
            avg_confidence=result.confidence_overall,
            model_used=result.model_used,
            processing_time_ms=result.processing_time_ms
        )
    
    def get_model_name(self) -> str:
        return "nanonets-ocr2-3b-sagemaker"


class EasyOCRAdapter(OCRAdapter):
    """
    EasyOCR multi-language fallback adapter.
    
    Useful for mixed language content or when TrOCR struggles.
    """
    
    def __init__(self, languages: List[str] = None):
        self.languages = languages or ['en']
        self.reader = None
    
    def _load_reader(self):
        """Lazy load EasyOCR reader"""
        if self.reader is not None:
            return
        
        try:
            import easyocr
            logger.info(f"Loading EasyOCR for languages: {self.languages}")
            self.reader = easyocr.Reader(self.languages, gpu=True)
        except Exception as e:
            logger.warning(f"EasyOCR GPU not available, using CPU: {e}")
            import easyocr
            self.reader = easyocr.Reader(self.languages, gpu=False)
    
    def extract_lines(self, image_bytes: bytes) -> OCRPageResult:
        """Extract text using EasyOCR"""
        import numpy as np
        from PIL import Image
        import io
        
        start_time = time.time()
        self._load_reader()
        
        # Decode image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(pil_image)
        
        # Run EasyOCR
        results = self.reader.readtext(image_np)
        
        ocr_lines = []
        all_confidences = []
        
        for (bbox_points, text, confidence) in results:
            if not text.strip():
                continue
            
            # Convert polygon to bounding box
            x_coords = [p[0] for p in bbox_points]
            y_coords = [p[1] for p in bbox_points]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            ocr_lines.append(OCRLine(
                text=text.strip(),
                bbox=bbox,
                confidence=confidence
            ))
            all_confidences.append(confidence)
        
        full_text = '\n'.join(line.text for line in ocr_lines)
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        processing_time = int((time.time() - start_time) * 1000)
        
        return OCRPageResult(
            lines=ocr_lines,
            full_text=full_text,
            avg_confidence=avg_confidence,
            model_used="easyocr",
            processing_time_ms=processing_time
        )
    
    def get_model_name(self) -> str:
        return f"easyocr-{'-'.join(self.languages)}"


def get_ocr_adapter(config: Optional[dict] = None) -> OCRAdapter:
    """
    Factory function for config-driven OCR adapter selection.
    
    Config options:
        OCR_ADAPTER: "trocr" (default), "sagemaker", or "easyocr"
        TROCR_MODEL_SIZE: "base" (default) or "large"
        EASYOCR_LANGUAGES: comma-separated language codes
    
    Environment variables can also be used:
        OCR_ADAPTER_TYPE, TROCR_MODEL_SIZE, EASYOCR_LANGUAGES
    """
    if config is None:
        config = {}
    
    adapter_type = config.get("OCR_ADAPTER") or os.getenv("OCR_ADAPTER_TYPE", "trocr")
    
    if adapter_type == "sagemaker":
        logger.info("Using SageMaker OCR adapter")
        return SageMakerAdapter()
    
    elif adapter_type == "easyocr":
        lang_str = config.get("EASYOCR_LANGUAGES") or os.getenv("EASYOCR_LANGUAGES", "en")
        languages = [l.strip() for l in lang_str.split(",")]
        logger.info(f"Using EasyOCR adapter with languages: {languages}")
        return EasyOCRAdapter(languages=languages)
    
    else:  # Default: trocr
        model_size = config.get("TROCR_MODEL_SIZE") or os.getenv("TROCR_MODEL_SIZE", "base")
        logger.info(f"Using TrOCR adapter with model size: {model_size}")
        return TrOCRAdapter(model_size=model_size)
