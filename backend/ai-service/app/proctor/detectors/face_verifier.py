"""
Face Verifier - Verifies student identity using face recognition

Refactored from: AutoOEP/Proctor/feature_extractor.py (DeepFace integration)

Features:
- Face verification against registered photo
- Multiple backend support (VGG-Face, ArcFace, etc.)
- Confidence scoring for identity matching
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional
import os
import tempfile

logger = logging.getLogger(__name__)


class FaceVerifier:
    """
    Verifies student identity by comparing live webcam face to registered photo.
    
    Uses DeepFace library supporting multiple backends:
    - VGG-Face (default)
    - ArcFace
    - Facenet
    - OpenFace
    
    Note: DeepFace is lazily loaded to avoid startup overhead.
    """
    
    # Available backends
    BACKENDS = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
    MODELS = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "ArcFace"]
    
    DEFAULT_MODEL = "VGG-Face"
    DEFAULT_BACKEND = "opencv"
    DEFAULT_DISTANCE_METRIC = "cosine"
    DEFAULT_THRESHOLD = 0.4  # Cosine distance threshold for match
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        detector_backend: str = DEFAULT_BACKEND,
        distance_metric: str = DEFAULT_DISTANCE_METRIC,
        threshold: float = DEFAULT_THRESHOLD
    ):
        """
        Initialize face verifier.
        
        Args:
            model_name: Face recognition model to use
            detector_backend: Face detection backend
            distance_metric: Distance metric for comparison (cosine, euclidean, euclidean_l2)
            threshold: Distance threshold for verification match
        """
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.threshold = threshold
        
        # Registered face for comparison
        self._registered_face: Optional[np.ndarray] = None
        self._registered_face_path: Optional[str] = None
        
        # DeepFace lazy loading
        self._deepface = None
        self._deepface_available: Optional[bool] = None
        
        # Verification stats
        self._verification_count = 0
        self._verified_count = 0
    
    def _check_deepface(self) -> bool:
        """Check if DeepFace is available"""
        if self._deepface_available is None:
            try:
                from deepface import DeepFace
                self._deepface = DeepFace
                self._deepface_available = True
                logger.info("DeepFace loaded successfully")
            except ImportError:
                self._deepface_available = False
                logger.warning("DeepFace not installed. Face verification disabled.")
        return self._deepface_available
    
    def register_face(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Register a reference face for verification.
        
        Args:
            face_image: BGR image containing the reference face
            
        Returns:
            Dict with registration status
        """
        if face_image is None or face_image.size == 0:
            return {"registered": False, "message": "Invalid image"}
        
        # Save to temp file for DeepFace
        try:
            fd, path = tempfile.mkstemp(suffix='.jpg')
            cv2.imwrite(path, face_image)
            os.close(fd)
            
            self._registered_face = face_image.copy()
            self._registered_face_path = path
            
            logger.info(f"Reference face registered: {path}")
            
            return {
                "registered": True,
                "message": "Face registered successfully",
                "path": path
            }
            
        except Exception as e:
            logger.error(f"Error registering face: {e}")
            return {"registered": False, "message": str(e)}
    
    def register_face_base64(self, image_base64: str) -> Dict[str, Any]:
        """
        Register face from base64-encoded image.
        
        Args:
            image_base64: Base64 encoded JPEG image
            
        Returns:
            Registration result
        """
        import base64
        
        try:
            image_bytes = base64.b64decode(image_base64)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            return self.register_face(image)
            
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return {"registered": False, "message": str(e)}
    
    def verify(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Verify if the face in frame matches registered face.
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            Dict with:
                - verified: bool
                - confidence: float (1 - distance)
                - distance: float
                - threshold: float
                - message: str
        """
        self._verification_count += 1
        
        if self._registered_face_path is None:
            return self._default_result("No reference face registered")
        
        if not self._check_deepface():
            return self._default_result("DeepFace not available")
        
        if frame is None or frame.size == 0:
            return self._default_result("Invalid input frame")
        
        try:
            # Save current frame to temp file
            fd, current_path = tempfile.mkstemp(suffix='.jpg')
            cv2.imwrite(current_path, frame)
            os.close(fd)
            
            # Verify using DeepFace
            result = self._deepface.verify(
                img1_path=self._registered_face_path,
                img2_path=current_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=False  # Don't fail if face not detected
            )
            
            # Cleanup temp file
            os.unlink(current_path)
            
            verified = result.get("verified", False)
            distance = result.get("distance", 1.0)
            confidence = 1.0 - min(distance, 1.0)
            
            if verified:
                self._verified_count += 1
            
            return {
                "verified": verified,
                "confidence": confidence,
                "distance": distance,
                "threshold": result.get("threshold", self.threshold),
                "message": "Match" if verified else "No match",
                "model": self.model_name,
                "verification_rate": self._verified_count / max(1, self._verification_count)
            }
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return self._default_result(f"Error: {str(e)}")
    
    def _default_result(self, message: str = "") -> Dict[str, Any]:
        """Return default result when verification fails"""
        return {
            "verified": False,
            "confidence": 0.0,
            "distance": 1.0,
            "threshold": self.threshold,
            "message": message,
            "model": self.model_name,
            "verification_rate": self._verified_count / max(1, self._verification_count)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get verification metrics"""
        return {
            "verification_count": self._verification_count,
            "verified_count": self._verified_count,
            "verification_rate": self._verified_count / max(1, self._verification_count),
            "has_registered_face": self._registered_face_path is not None,
            "model": self.model_name
        }
    
    def reset(self):
        """Reset verification state"""
        self._verification_count = 0
        self._verified_count = 0
    
    def clear_registration(self):
        """Clear registered face"""
        if self._registered_face_path and os.path.exists(self._registered_face_path):
            try:
                os.unlink(self._registered_face_path)
            except:
                pass
        
        self._registered_face = None
        self._registered_face_path = None
        self.reset()
    
    def is_available(self) -> bool:
        """Check if face verification is available"""
        return self._check_deepface()
