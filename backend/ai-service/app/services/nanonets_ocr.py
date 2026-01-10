"""
OCR Service - Multi-backend OCR with HuggingFace API + Tesseract fallback

This service provides reliable OCR for handwritten notes by:
1. First trying HuggingFace Inference API with TrOCR
2. Falling back to Tesseract if API fails or is unavailable
"""
import os
import logging
import base64
from typing import Tuple, Optional
from io import BytesIO
import time

logger = logging.getLogger(__name__)

# Configuration
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
# TrOCR model for handwritten text
DEFAULT_MODEL = os.getenv("OCR_MODEL", "microsoft/trocr-base-handwritten")
MAX_RETRIES = 3
TIMEOUT_SECONDS = 60


class OCRService:
    """
    Multi-backend OCR service with HuggingFace API + Tesseract fallback.
    """
    
    def __init__(
        self,
        api_key: str = HF_API_KEY,
        model_id: str = DEFAULT_MODEL
    ):
        self.api_key = api_key
        self.model_id = model_id
        # Use the new router.huggingface.co endpoint
        # For image-to-text models, the format is:
        # POST https://router.huggingface.co/hf-inference/models/{model_id}
        # with raw image bytes and Authorization header
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
        
        # Check for Tesseract availability
        self.tesseract_available = False
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("Tesseract OCR available as fallback")
        except Exception:
            logger.warning("Tesseract not available")
    
    async def extract_text(
        self,
        image,
        prompt: str = None  # Ignored for TrOCR, kept for API compatibility
    ) -> Tuple[str, float]:
        """
        Extract text from an image.
        
        First tries HuggingFace API, then falls back to Tesseract.
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
            logger.error(f"Failed to load image: {e}")
            return "", 0.0
        
        text = ""
        confidence = 0.0
        
        # Try HuggingFace API first (if key is provided)
        if self.api_key:
            logger.info(f"Trying HuggingFace API with model: {self.model_id}")
            text, confidence = await self._extract_with_hf_api(img)
            if text:
                logger.info(f"HF API success: extracted {len(text)} chars")
                return text, confidence
            else:
                logger.warning("HF API failed, falling back to Tesseract")
        
        # Fallback to Tesseract
        if self.tesseract_available:
            logger.info("Using Tesseract OCR fallback")
            text, confidence = self._extract_with_tesseract(img)
            if text:
                logger.info(f"Tesseract extracted {len(text)} chars, confidence: {confidence:.2f}")
                return text, confidence
        
        logger.warning("Both OCR backends failed or returned empty")
        return "", 0.0
    
    async def _extract_with_hf_api(self, img) -> Tuple[str, float]:
        """Extract text using HuggingFace Inference API."""
        import httpx
        
        try:
            # Convert image to bytes
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "image/png"  # Important: specify image content type
            }
            
            start_time = time.time()
            
            for attempt in range(MAX_RETRIES):
                try:
                    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                        response = await client.post(
                            self.api_url,
                            headers=headers,
                            content=img_bytes  # Send raw image bytes
                        )
                    
                    elapsed = time.time() - start_time
                    
                    if response.status_code == 200:
                        result = response.json()
                        text = self._parse_response(result)
                        logger.info(f"HF API completed in {elapsed:.1f}s")
                        # TrOCR confidence estimation
                        confidence = 0.85 if text and len(text) > 5 else 0.4
                        return text, confidence
                    
                    elif response.status_code == 503:
                        # Model loading
                        wait_time = 30
                        try:
                            data = response.json()
                            wait_time = data.get("estimated_time", 30)
                        except:
                            pass
                        logger.info(f"Model loading, waiting {wait_time}s...")
                        import asyncio
                        await asyncio.sleep(min(wait_time, 60))
                        continue
                    
                    elif response.status_code == 404:
                        # Model not found or endpoint issue
                        logger.error(f"API error 404: Model or endpoint not found")
                        return "", 0.0
                    
                    else:
                        error_text = response.text[:200] if response.text else "No response"
                        logger.error(f"API error {response.status_code}: {error_text}")
                        return "", 0.0
                        
                except httpx.TimeoutException:
                    logger.warning(f"Timeout (attempt {attempt + 1}/{MAX_RETRIES})")
                    if attempt < MAX_RETRIES - 1:
                        import asyncio
                        await asyncio.sleep(3)
                        continue
                    return "", 0.0
                    
        except Exception as e:
            logger.error(f"HF API error: {e}")
            return "", 0.0
        
        return "", 0.0
    
    def _extract_with_tesseract(self, img) -> Tuple[str, float]:
        """Extract text using Tesseract OCR."""
        try:
            import pytesseract
            
            # Use PSM 6 for uniform text blocks (good for handwritten notes)
            # Add OEM 3 for best OCR engine mode
            custom_config = r'--psm 6 --oem 3 -l eng'
            
            # Get text
            text = pytesseract.image_to_string(img, config=custom_config)
            
            # Get confidence from word-level data
            try:
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                confidences = []
                for conf in data['conf']:
                    if conf != '-1':
                        try:
                            c = int(conf)
                            if c > 0:
                                confidences.append(c)
                        except:
                            pass
                if confidences:
                    confidence = sum(confidences) / len(confidences) / 100
                else:
                    confidence = 0.5
            except Exception as e:
                logger.warning(f"Could not get Tesseract confidence: {e}")
                confidence = 0.5
            
            return text.strip(), confidence
            
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return "", 0.0
    
    def _parse_response(self, result) -> str:
        """Parse HuggingFace API response."""
        if isinstance(result, list):
            if len(result) > 0:
                item = result[0]
                if isinstance(item, dict):
                    return item.get("generated_text", "") or item.get("text", "")
                return str(item)
            return ""
        
        if isinstance(result, dict):
            if "generated_text" in result:
                return result["generated_text"]
            if "text" in result:
                return result["text"]
        
        return str(result) if result else ""


# Global service instance
_ocr_service: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    """Get or create the global OCR service instance."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service
