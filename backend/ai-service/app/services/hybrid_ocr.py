"""
Hybrid OCR Service - Tesseract Layout + TrOCR Line Recognition

This approach combines:
1. Tesseract for layout analysis (detecting text lines/regions)
2. TrOCR for line-by-line handwritten text recognition

This works much better than whole-page TrOCR because:
- TrOCR is designed for single-line text recognition
- Tesseract is good at detecting text regions even in handwriting
"""
import os
import logging
import asyncio
from typing import List, Tuple, Optional, Dict, Any
from io import BytesIO
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Configuration
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
TROCR_MODEL = os.getenv("OCR_MODEL", "microsoft/trocr-base-handwritten")
TIMEOUT_SECONDS = 30
MIN_LINE_HEIGHT = 20  # Minimum pixels for a text line
LINE_PADDING = 10  # Padding around detected lines


@dataclass
class TextLine:
    """Detected text line with bounding box."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    text: str = ""
    confidence: float = 0.0


class HybridOCRService:
    """
    Hybrid OCR: Tesseract for layout + TrOCR for recognition.
    
    Much better for handwritten notes than full-page recognition.
    """
    
    def __init__(
        self,
        api_key: str = HF_API_KEY,
        model_id: str = TROCR_MODEL
    ):
        self.api_key = api_key
        self.model_id = model_id
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
        
        # Check Tesseract availability
        self.tesseract_available = False
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("[HybridOCR] Tesseract available")
        except Exception:
            logger.warning("[HybridOCR] Tesseract not available")
    
    async def extract_text(
        self,
        image,
        use_hybrid: bool = True
    ) -> Tuple[str, float, List[TextLine]]:
        """
        Extract text from an image using hybrid approach.
        
        Args:
            image: PIL Image, path, or numpy array
            use_hybrid: If True, use Tesseract layout + TrOCR lines
            
        Returns:
            (full_text, avg_confidence, lines)
        """
        from PIL import Image as PILImage
        
        # Normalize image input
        try:
            if isinstance(image, str):
                img = PILImage.open(image).convert("RGB")
            elif hasattr(image, 'convert'):
                img = image.convert("RGB")
            else:
                img = PILImage.fromarray(image).convert("RGB")
        except Exception as e:
            logger.error(f"[HybridOCR] Failed to load image: {e}")
            return "", 0.0, []
        
        if not use_hybrid or not self.tesseract_available:
            # Fallback to simple Tesseract
            text, confidence = self._tesseract_full_page(img)
            return text, confidence, []
        
        # Step 1: Detect text lines using Tesseract
        logger.info("[HybridOCR] Step 1: Detecting text lines with Tesseract...")
        lines = self._detect_lines(img)
        logger.info(f"[HybridOCR] Detected {len(lines)} text lines")
        
        if not lines:
            # No lines detected, fall back to full page
            logger.warning("[HybridOCR] No lines detected, using full page")
            text, confidence = self._tesseract_full_page(img)
            return text, confidence, []
        
        # Step 2: Recognize each line using TrOCR
        if self.api_key:
            logger.info("[HybridOCR] Step 2: Recognizing lines with TrOCR...")
            lines = await self._recognize_lines_trocr(img, lines)
        else:
            logger.info("[HybridOCR] Step 2: Recognizing lines with Tesseract...")
            lines = self._recognize_lines_tesseract(img, lines)
        
        # Step 3: Assemble full text
        full_text = "\n".join([line.text for line in lines if line.text.strip()])
        
        # Calculate average confidence
        if lines:
            avg_conf = sum(l.confidence for l in lines) / len(lines)
        else:
            avg_conf = 0.0
        
        logger.info(f"[HybridOCR] Extracted {len(full_text)} chars, avg confidence: {avg_conf:.2f}")
        
        return full_text, avg_conf, lines
    
    def _detect_lines(self, img) -> List[TextLine]:
        """
        Detect text lines using Tesseract's layout analysis.
        
        Returns list of TextLine with bounding boxes.
        """
        try:
            import pytesseract
            
            # Get word-level bounding boxes
            data = pytesseract.image_to_data(
                img, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6'  # Assume single uniform block
            )
            
            # Group words into lines based on line_num
            lines_dict: Dict[int, List[Tuple[int, int, int, int]]] = {}
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) < 0:  # Skip empty/invalid
                    continue
                
                line_num = data['line_num'][i]
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                if w <= 0 or h <= 0:
                    continue
                
                if line_num not in lines_dict:
                    lines_dict[line_num] = []
                lines_dict[line_num].append((x, y, x + w, y + h))
            
            # Convert to TextLine objects with merged bounding boxes
            lines = []
            for line_num in sorted(lines_dict.keys()):
                boxes = lines_dict[line_num]
                if not boxes:
                    continue
                
                # Merge all word boxes into line box
                x1 = min(b[0] for b in boxes)
                y1 = min(b[1] for b in boxes)
                x2 = max(b[2] for b in boxes)
                y2 = max(b[3] for b in boxes)
                
                # Add padding
                x1 = max(0, x1 - LINE_PADDING)
                y1 = max(0, y1 - LINE_PADDING)
                x2 = min(img.width, x2 + LINE_PADDING)
                y2 = min(img.height, y2 + LINE_PADDING)
                
                # Filter too small lines
                if y2 - y1 < MIN_LINE_HEIGHT:
                    continue
                
                lines.append(TextLine(
                    bbox=(x1, y1, x2, y2),
                    text="",
                    confidence=0.0
                ))
            
            return lines
            
        except Exception as e:
            logger.error(f"[HybridOCR] Line detection failed: {e}")
            return []
    
    async def _recognize_lines_trocr(
        self,
        img,
        lines: List[TextLine]
    ) -> List[TextLine]:
        """
        Recognize each line using TrOCR via HuggingFace API.
        
        Processes lines in parallel for speed.
        """
        import httpx
        
        async def recognize_single_line(line: TextLine) -> TextLine:
            """Recognize a single line."""
            try:
                # Crop line from image
                x1, y1, x2, y2 = line.bbox
                line_img = img.crop((x1, y1, x2, y2))
                
                # Convert to bytes
                buffer = BytesIO()
                line_img.save(buffer, format="PNG")
                img_bytes = buffer.getvalue()
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "image/png"
                }
                
                async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        content=img_bytes
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    text = self._parse_response(result)
                    line.text = text
                    line.confidence = 0.85 if text else 0.0
                else:
                    # Fallback to Tesseract for this line
                    line_text = self._tesseract_line(line_img)
                    line.text = line_text
                    line.confidence = 0.6
                    
            except Exception as e:
                logger.warning(f"[HybridOCR] Line recognition failed: {e}")
                # Fallback to Tesseract
                try:
                    x1, y1, x2, y2 = line.bbox
                    line_img = img.crop((x1, y1, x2, y2))
                    line.text = self._tesseract_line(line_img)
                    line.confidence = 0.5
                except:
                    pass
            
            return line
        
        # Process all lines in parallel (batch of 5 to avoid rate limits)
        batch_size = 5
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            tasks = [recognize_single_line(line) for line in batch]
            await asyncio.gather(*tasks)
            
            # Small delay between batches to avoid rate limits
            if i + batch_size < len(lines):
                await asyncio.sleep(0.5)
        
        return lines
    
    def _recognize_lines_tesseract(
        self,
        img,
        lines: List[TextLine]
    ) -> List[TextLine]:
        """Recognize each line using Tesseract."""
        for line in lines:
            try:
                x1, y1, x2, y2 = line.bbox
                line_img = img.crop((x1, y1, x2, y2))
                line.text = self._tesseract_line(line_img)
                line.confidence = 0.7
            except Exception as e:
                logger.warning(f"[HybridOCR] Tesseract line failed: {e}")
        
        return lines
    
    def _tesseract_line(self, line_img) -> str:
        """Extract text from a single line using Tesseract."""
        try:
            import pytesseract
            # PSM 7: Treat image as a single text line
            text = pytesseract.image_to_string(
                line_img,
                config='--psm 7 --oem 3'
            )
            return text.strip()
        except Exception as e:
            logger.error(f"[HybridOCR] Tesseract line error: {e}")
            return ""
    
    def _tesseract_full_page(self, img) -> Tuple[str, float]:
        """Fallback full-page Tesseract OCR."""
        try:
            import pytesseract
            
            text = pytesseract.image_to_string(
                img,
                config='--psm 6 --oem 3 -l eng'
            )
            
            # Get confidence
            try:
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                confidences = [int(c) for c in data['conf'] if str(c) != '-1' and int(c) > 0]
                confidence = sum(confidences) / len(confidences) / 100 if confidences else 0.5
            except:
                confidence = 0.5
            
            return text.strip(), confidence
            
        except Exception as e:
            logger.error(f"[HybridOCR] Full page OCR failed: {e}")
            return "", 0.0
    
    def _parse_response(self, result) -> str:
        """Parse HuggingFace API response."""
        if isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, dict):
                return item.get("generated_text", "") or item.get("text", "")
            return str(item)
        
        if isinstance(result, dict):
            return result.get("generated_text", "") or result.get("text", "")
        
        return str(result) if result else ""


# Global service instance
_hybrid_ocr_service: Optional[HybridOCRService] = None


def get_hybrid_ocr_service() -> HybridOCRService:
    """Get or create the global Hybrid OCR service instance."""
    global _hybrid_ocr_service
    if _hybrid_ocr_service is None:
        _hybrid_ocr_service = HybridOCRService()
    return _hybrid_ocr_service
