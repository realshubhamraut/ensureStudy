"""
Blink Detector - Detects eye blinks using Eye Aspect Ratio (EAR)

Refactored from: Artificial-Intelligence-based-Online-Exam-Proctoring-System/blink_detection.py

Features:
- Eye Aspect Ratio (EAR) calculation
- Blink counting for identity verification
- Works with dlib 68-point facial landmarks
"""

import cv2
import numpy as np
import logging
from math import hypot
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class BlinkDetector:
    """
    Detects eye blinks using Eye Aspect Ratio (EAR) algorithm.
    
    Uses the 68-point dlib facial landmarks:
    - Left eye: points 36-41
    - Right eye: points 42-47
    
    EAR drops significantly when eyes are closed, enabling blink detection.
    """
    
    # Landmark indices for eyes (68-point model)
    LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
    
    # Default EAR threshold for blink detection
    # When EAR falls below this, eye is considered closed
    DEFAULT_EAR_THRESHOLD = 0.25
    
    # Consecutive frames threshold for confirmed blink
    DEFAULT_CONSEC_FRAMES = 2
    
    def __init__(
        self,
        ear_threshold: float = DEFAULT_EAR_THRESHOLD,
        consec_frames: int = DEFAULT_CONSEC_FRAMES
    ):
        """
        Initialize blink detector.
        
        Args:
            ear_threshold: EAR value below which eye is considered closed
            consec_frames: Number of consecutive frames for confirmed blink
        """
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        
        # State tracking
        self._blink_counter = 0  # Consecutive frames with closed eyes
        self._total_blinks = 0
        self._frame_count = 0
    
    def _midpoint(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[float, float]:
        """Calculate midpoint between two points"""
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    
    def _distance(self, p1: Tuple, p2: Tuple) -> float:
        """Calculate Euclidean distance between two points"""
        return hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    def _calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR).
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Where p1-p6 are the 6 landmark points of one eye.
        
        Args:
            eye_points: Array of 6 (x, y) points for one eye
            
        Returns:
            EAR value (typically 0.2-0.4 for open eye)
        """
        if len(eye_points) != 6:
            return 0.0
        
        # Vertical distances
        v1 = self._distance(eye_points[1], eye_points[5])
        v2 = self._distance(eye_points[2], eye_points[4])
        
        # Horizontal distance
        h = self._distance(eye_points[0], eye_points[3])
        
        if h == 0:
            return 0.0
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def detect(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Detect blink from facial landmarks.
        
        Args:
            landmarks: Array of 68 (x, y) facial landmark points
            
        Returns:
            Dict with:
                - is_blinking: bool
                - left_ear: float (left eye aspect ratio)
                - right_ear: float (right eye aspect ratio)
                - avg_ear: float (average EAR)
                - total_blinks: int
                - blink_rate: float (blinks per frame)
        """
        self._frame_count += 1
        
        if landmarks is None or len(landmarks) < 48:
            return self._default_result()
        
        try:
            # Extract eye landmarks
            left_eye = landmarks[self.LEFT_EYE_INDICES]
            right_eye = landmarks[self.RIGHT_EYE_INDICES]
            
            # Calculate EAR for each eye
            left_ear = self._calculate_ear(left_eye)
            right_ear = self._calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Check for blink
            is_blinking = avg_ear < self.ear_threshold
            
            if is_blinking:
                self._blink_counter += 1
            else:
                # Check if we just finished a blink
                if self._blink_counter >= self.consec_frames:
                    self._total_blinks += 1
                self._blink_counter = 0
            
            return {
                "is_blinking": is_blinking,
                "left_ear": left_ear,
                "right_ear": right_ear,
                "avg_ear": avg_ear,
                "total_blinks": self._total_blinks,
                "blink_rate": self._total_blinks / max(1, self._frame_count),
                "blink_counter": self._blink_counter
            }
            
        except Exception as e:
            logger.error(f"Blink detection error: {e}")
            return self._default_result()
    
    def detect_from_face_result(self, face_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect blink from FaceDetector result.
        
        Args:
            face_result: Result dict from FaceDetector.detect()
            
        Returns:
            Blink detection result
        """
        landmarks_list = face_result.get("landmarks", [])
        
        if not landmarks_list:
            return self._default_result()
        
        # Use first face's landmarks
        landmarks = landmarks_list[0]
        return self.detect(landmarks)
    
    def _default_result(self) -> Dict[str, Any]:
        """Return default result when detection fails"""
        return {
            "is_blinking": False,
            "left_ear": 0.0,
            "right_ear": 0.0,
            "avg_ear": 0.0,
            "total_blinks": self._total_blinks,
            "blink_rate": self._total_blinks / max(1, self._frame_count),
            "blink_counter": 0
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get blink detection metrics"""
        return {
            "total_blinks": self._total_blinks,
            "frame_count": self._frame_count,
            "blink_rate": self._total_blinks / max(1, self._frame_count),
            "ear_threshold": self.ear_threshold
        }
    
    def reset(self):
        """Reset counters"""
        self._blink_counter = 0
        self._total_blinks = 0
        self._frame_count = 0
    
    def draw_eyes(
        self, 
        frame: np.ndarray, 
        landmarks: np.ndarray,
        result: Dict[str, Any]
    ) -> np.ndarray:
        """
        Draw eye landmarks and EAR on frame (for debugging).
        
        Args:
            frame: BGR image
            landmarks: 68-point facial landmarks
            result: Result from detect()
            
        Returns:
            Annotated frame
        """
        if landmarks is None or len(landmarks) < 48:
            return frame
        
        annotated = frame.copy()
        
        # Draw eye landmarks
        for idx in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES:
            x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
            color = (0, 0, 255) if result.get("is_blinking") else (0, 255, 0)
            cv2.circle(annotated, (x, y), 2, color, -1)
        
        # Draw eye contours
        left_eye = landmarks[self.LEFT_EYE_INDICES].astype(int)
        right_eye = landmarks[self.RIGHT_EYE_INDICES].astype(int)
        
        cv2.polylines(annotated, [left_eye], True, (0, 255, 255), 1)
        cv2.polylines(annotated, [right_eye], True, (0, 255, 255), 1)
        
        # Display EAR and blink count
        text = f"EAR: {result.get('avg_ear', 0):.2f} | Blinks: {result.get('total_blinks', 0)}"
        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if result.get("is_blinking"):
            cv2.putText(annotated, "BLINK", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return annotated
