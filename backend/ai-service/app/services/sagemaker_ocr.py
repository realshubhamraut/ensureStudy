"""
SageMaker OCR Service - Nanonets-OCR2-3B Integration

Provides high-quality OCR via SageMaker Serverless with fallback to local TrOCR.

Features:
- SageMaker Serverless endpoint invocation
- Redis caching by image hash
- Automatic fallback to TrOCR
- Cold start handling with retries
- Formula extraction (LaTeX)
"""
import os
import base64
import hashlib
import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class OCRFormula:
    """Extracted mathematical formula"""
    latex: str
    position: Dict[str, int]  # x, y, width, height
    confidence: float


@dataclass
class OCRLayoutElement:
    """Layout element from document"""
    element_type: str  # heading, paragraph, formula, table, list
    text: str
    bbox: List[int]  # [x, y, width, height]
    confidence: float


@dataclass
class SageMakerOCRResult:
    """Result from SageMaker OCR endpoint"""
    text: str
    formulas: List[OCRFormula]
    layout: List[OCRLayoutElement]
    confidence_overall: float
    processing_time_ms: int
    cache_hit: bool = False
    model_used: str = "nanonets-ocr"


class SageMakerOCRService:
    """
    SageMaker Nanonets-OCR integration with fallback support.
    
    Environment Variables:
    - SAGEMAKER_OCR_ENABLED: Enable SageMaker (default: false for cost saving)
    - SAGEMAKER_OCR_ENDPOINT: Endpoint name (default: ensurestudy-ocr-serverless)
    - AWS_REGION: AWS region (default: us-east-1)
    - OCR_CACHE_TTL: Cache TTL in seconds (default: 604800 = 7 days)
    """
    
    def __init__(self):
        self.enabled = os.getenv("SAGEMAKER_OCR_ENABLED", "false").lower() == "true"
        self.endpoint_name = os.getenv("SAGEMAKER_OCR_ENDPOINT", "ensurestudy-ocr-serverless")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.cache_ttl = int(os.getenv("OCR_CACHE_TTL", "604800"))  # 7 days
        
        self.sagemaker_client = None
        self.cache = None
        
        if self.enabled:
            self._initialize_clients()
        
        logger.info(f"[SageMaker OCR] Enabled: {self.enabled}, Endpoint: {self.endpoint_name}")
    
    def _initialize_clients(self):
        """Initialize AWS and Redis clients"""
        try:
            import boto3
            self.sagemaker_client = boto3.client(
                'sagemaker-runtime',
                region_name=self.region
            )
            logger.info("[SageMaker OCR] AWS client initialized")
        except Exception as e:
            logger.warning(f"[SageMaker OCR] Failed to initialize AWS client: {e}")
            self.enabled = False
        
        try:
            from app.services.response_cache import get_response_cache
            self.cache = get_response_cache()
        except Exception as e:
            logger.warning(f"[SageMaker OCR] Cache not available: {e}")
    
    def _generate_cache_key(self, image_bytes: bytes, options: Dict) -> str:
        """Generate cache key from image hash and options"""
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        options_hash = hashlib.md5(json.dumps(options, sort_keys=True).encode()).hexdigest()[:8]
        return f"ocr:{image_hash}:{options_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[SageMakerOCRResult]:
        """Get cached OCR result"""
        if not self.cache:
            return None
        
        try:
            # Try to get from Redis
            if hasattr(self.cache, 'redis_client') and self.cache.redis_client:
                data = self.cache.redis_client.get(cache_key)
                if data:
                    parsed = json.loads(data)
                    return SageMakerOCRResult(
                        text=parsed["text"],
                        formulas=[OCRFormula(**f) for f in parsed.get("formulas", [])],
                        layout=[OCRLayoutElement(**e) for e in parsed.get("layout", [])],
                        confidence_overall=parsed.get("confidence_overall", 0.9),
                        processing_time_ms=parsed.get("processing_time_ms", 0),
                        cache_hit=True,
                        model_used=parsed.get("model_used", "nanonets-ocr")
                    )
        except Exception as e:
            logger.warning(f"[SageMaker OCR] Cache read error: {e}")
        
        return None
    
    def _set_cache(self, cache_key: str, result: SageMakerOCRResult):
        """Cache OCR result"""
        if not self.cache:
            return
        
        try:
            data = {
                "text": result.text,
                "formulas": [asdict(f) for f in result.formulas],
                "layout": [asdict(e) for e in result.layout],
                "confidence_overall": result.confidence_overall,
                "processing_time_ms": result.processing_time_ms,
                "model_used": result.model_used
            }
            
            if hasattr(self.cache, 'redis_client') and self.cache.redis_client:
                self.cache.redis_client.setex(
                    cache_key, 
                    self.cache_ttl, 
                    json.dumps(data)
                )
        except Exception as e:
            logger.warning(f"[SageMaker OCR] Cache write error: {e}")
    
    async def extract_text(
        self,
        image_bytes: bytes,
        options: Optional[Dict] = None
    ) -> SageMakerOCRResult:
        """
        Extract text from image using SageMaker or fallback.
        
        Args:
            image_bytes: Raw image bytes
            options: OCR options (max_tokens, extract_formulas, etc.)
        
        Returns:
            SageMakerOCRResult with text, formulas, and layout
        """
        options = options or {
            "max_tokens": 2048,
            "return_confidence": True,
            "extract_formulas": True
        }
        
        # Check cache
        cache_key = self._generate_cache_key(image_bytes, options)
        cached = self._get_from_cache(cache_key)
        if cached:
            logger.info("[SageMaker OCR] Cache hit")
            return cached
        
        # Try SageMaker if enabled
        if self.enabled and self.sagemaker_client:
            try:
                result = await self._invoke_sagemaker(image_bytes, options)
                self._set_cache(cache_key, result)
                return result
            except Exception as e:
                logger.warning(f"[SageMaker OCR] SageMaker failed, falling back: {e}")
        
        # Fallback to local TrOCR
        result = await self._fallback_trocr(image_bytes)
        self._set_cache(cache_key, result)
        return result
    
    async def _invoke_sagemaker(
        self,
        image_bytes: bytes,
        options: Dict
    ) -> SageMakerOCRResult:
        """Invoke SageMaker endpoint"""
        import asyncio
        
        start_time = time.time()
        
        # Prepare payload
        payload = {
            "image": base64.b64encode(image_bytes).decode('utf-8'),
            "max_tokens": options.get("max_tokens", 2048),
            "return_confidence": options.get("return_confidence", True),
            "extract_formulas": options.get("extract_formulas", True)
        }
        
        # Invoke endpoint (run in thread pool for async)
        loop = asyncio.get_event_loop()
        
        def invoke():
            return self.sagemaker_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload),
                # Increased timeout for cold starts
                CustomAttributes='timeout=45'
            )
        
        response = await loop.run_in_executor(None, invoke)
        
        # Parse response
        result_body = json.loads(response['Body'].read().decode())
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Parse formulas
        formulas = [
            OCRFormula(
                latex=f.get("latex", ""),
                position=f.get("position", {}),
                confidence=f.get("confidence", 0.9)
            )
            for f in result_body.get("formulas", [])
        ]
        
        # Parse layout
        layout = [
            OCRLayoutElement(
                element_type=e.get("type", "paragraph"),
                text=e.get("text", ""),
                bbox=e.get("bbox", []),
                confidence=e.get("confidence", 0.9)
            )
            for e in result_body.get("layout", [])
        ]
        
        return SageMakerOCRResult(
            text=result_body.get("text", ""),
            formulas=formulas,
            layout=layout,
            confidence_overall=result_body.get("confidence_overall", 0.9),
            processing_time_ms=processing_time,
            model_used="nanonets-ocr-sagemaker"
        )
    
    async def _fallback_trocr(self, image_bytes: bytes) -> SageMakerOCRResult:
        """Fallback to local TrOCR"""
        import cv2
        import numpy as np
        
        start_time = time.time()
        
        try:
            from app.services.ocr_service import OCRService
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image")
            
            # Use existing TrOCR service
            ocr = OCRService()
            result = ocr.extract_text(image)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return SageMakerOCRResult(
                text=result.text,
                formulas=[],  # TrOCR doesn't extract formulas
                layout=[],    # No layout from TrOCR
                confidence_overall=result.confidence,
                processing_time_ms=processing_time,
                model_used=f"trocr-fallback ({result.model_used})"
            )
            
        except Exception as e:
            logger.error(f"[SageMaker OCR] TrOCR fallback failed: {e}")
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return SageMakerOCRResult(
                text="",
                formulas=[],
                layout=[],
                confidence_overall=0.0,
                processing_time_ms=processing_time,
                model_used="error"
            )
    
    async def warm_up(self) -> bool:
        """Warm up the SageMaker endpoint to reduce cold start latency"""
        if not self.enabled or not self.sagemaker_client:
            return False
        
        try:
            # Create a small test image
            import numpy as np
            import cv2
            
            # 100x100 white image with "test" text
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            cv2.putText(img, "test", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            _, buffer = cv2.imencode('.png', img)
            image_bytes = buffer.tobytes()
            
            # Invoke with minimal options
            await self._invoke_sagemaker(image_bytes, {"max_tokens": 100})
            
            logger.info("[SageMaker OCR] Warm-up successful")
            return True
            
        except Exception as e:
            logger.warning(f"[SageMaker OCR] Warm-up failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "enabled": self.enabled,
            "endpoint_name": self.endpoint_name,
            "region": self.region,
            "cache_available": self.cache is not None,
            "sagemaker_client_ready": self.sagemaker_client is not None
        }


# Singleton instance
_sagemaker_ocr: Optional[SageMakerOCRService] = None


def get_sagemaker_ocr() -> SageMakerOCRService:
    """Get or create SageMaker OCR service singleton"""
    global _sagemaker_ocr
    if _sagemaker_ocr is None:
        _sagemaker_ocr = SageMakerOCRService()
    return _sagemaker_ocr


# Convenience function
async def extract_text_with_sagemaker(
    image_bytes: bytes,
    options: Optional[Dict] = None
) -> SageMakerOCRResult:
    """
    Extract text using SageMaker OCR (or fallback)
    
    Args:
        image_bytes: Raw image bytes
        options: OCR options
    
    Returns:
        SageMakerOCRResult
    """
    service = get_sagemaker_ocr()
    return await service.extract_text(image_bytes, options)
