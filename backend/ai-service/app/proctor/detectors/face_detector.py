"""
Face Detector - Detects faces using dlib's HOG detector

Refactored from: Artificial-Intelligence-based-Online-Exam-Proctoring-System/facial_detections.py
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Detects faces in video frames using dlib's HOG-based face detector.
    
    Provides:
    - Face count (for multi-person detection)
    - Face presence (for absence detection)
    - Face bounding boxes
    - Facial landmarks (68-point)
    """
    
    def __init__(self, predictor_path: str = None):
        """
        Initialize face detector.
        
        Args:
            predictor_path: Path to dlib shape predictor model.
                           If None, uses default from model_loader.
        """
        try:
            import dlib
            self.dlib = dlib
            self.detector = dlib.get_frontal_face_detector()
            
            if predictor_path:
                self.predictor = dlib.shape_predictor(predictor_path)
            else:
                # Lazy load from model_loader
                self.predictor = None
                self._predictor_loaded = False
        except ImportError:
            logger.error("dlib not installed. Run: pip install dlib")
            raise
    
    def _ensure_predictor(self):
        """Lazy load predictor if not already loaded"""
        if self.predictor is None and not self._predictor_loaded:
            try:
                from ..models import get_dlib_predictor
                self.predictor = get_dlib_predictor()
                self._predictor_loaded = True
            except Exception as e:
                logger.warning(f"Could not load dlib predictor: {e}")
                self._predictor_loaded = True  # Don't retry
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect faces in a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            dict with:
                - num_faces: int (0, 1, 2+)
                - face_present: bool
                - faces: List of dlib rectangles
                - landmarks: List of 68-point landmarks (if predictor loaded)
        """
        if frame is None or frame.size == 0:
            return {
                "num_faces": 0,
                "face_present": False,
                "faces": [],
                "landmarks": []
            }
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray, 0)
        num_faces = len(faces)
        
        # Get landmarks if predictor available
        landmarks = []
        self._ensure_predictor()
        if self.predictor is not None and num_faces > 0:
            for face in faces:
                try:
                    marks = self.predictor(gray, face)
                    # Convert to numpy array of (x, y) points
                    points = np.array([
                        (marks.part(i).x, marks.part(i).y) 
                        for i in range(68)
                    ])
                    landmarks.append(points)
                except Exception as e:
                    logger.warning(f"Error getting landmarks: {e}")
        
        return {
            "num_faces": num_faces,
            "face_present": num_faces > 0,
            "faces": faces,
            "landmarks": landmarks
        }
    
    def get_face_bbox(self, face) -> Tuple[int, int, int, int]:
        """
        Get bounding box from dlib face rectangle.
        
        Returns:
            (x, y, width, height)
        """
        return (face.left(), face.top(), face.width(), face.height())
    
    def draw_faces(self, frame: np.ndarray, faces, landmarks=None) -> np.ndarray:
        """
        Draw face bounding boxes and landmarks on frame (for debugging).
        
        Args:
            frame: BGR image
            faces: List of dlib rectangles
            landmarks: Optional list of 68-point landmarks
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for i, face in enumerate(faces):
            x, y, w, h = self.get_face_bbox(face)
            
            # Draw corners (like original code)
            color = (0, 255, 255)  # Yellow
            
            # Top left
            cv2.line(annotated, (x, y), (x + 20, y), color, 2)
            cv2.line(annotated, (x, y), (x, y + 20), color, 2)
            
            # Top right
            cv2.line(annotated, (x + w, y), (x + w - 20, y), color, 2)
            cv2.line(annotated, (x + w, y), (x + w, y + 20), color, 2)
            
            # Bottom left
            cv2.line(annotated, (x, y + h), (x + 20, y + h), color, 2)
            cv2.line(annotated, (x, y + h), (x, y + h - 20), color, 2)
            
            # Bottom right
            cv2.line(annotated, (x + w, y + h), (x + w - 20, y + h), color, 2)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - 20), color, 2)
            
            # Draw landmarks if available
            if landmarks and i < len(landmarks):
                for (lx, ly) in landmarks[i]:
                    cv2.circle(annotated, (lx, ly), 2, (255, 255, 0), -1)
        
        return annotated
