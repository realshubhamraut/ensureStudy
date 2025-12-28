"""
Hand Detector - Detects hands using MediaPipe

Refactored from: AutoOEP/VisionUtils/handpose.py
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class HandDetector:
    """
    Detects hands in frames using MediaPipe Hands.
    
    Provides:
    - Hand count
    - Hand visibility status
    - Hand landmarks (21 points per hand)
    """
    
    def __init__(self, max_hands: int = 2, min_confidence: float = 0.5):
        """
        Initialize hand detector.
        
        Args:
            max_hands: Maximum number of hands to detect
            min_confidence: Minimum detection confidence
        """
        self.max_hands = max_hands
        self.min_confidence = min_confidence
        self.hands = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialize MediaPipe hands"""
        if self._initialized:
            return
        
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence
            )
            self.mp_draw = mp.solutions.drawing_utils
            self._initialized = True
            logger.info("MediaPipe Hands initialized successfully")
        except ImportError:
            logger.error("MediaPipe not installed. Run: pip install mediapipe")
            self._initialized = True  # Don't retry
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            self._initialized = True
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect hands in a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            dict with:
                - num_hands: int (0, 1, 2)
                - hands_visible: bool
                - landmarks: List of hand landmarks (21 points each)
                - handedness: List of 'Left' or 'Right'
        """
        if frame is None or frame.size == 0:
            return self._default_result()
        
        self._ensure_initialized()
        
        if self.hands is None:
            return self._default_result()
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks is None:
                return {
                    "num_hands": 0,
                    "hands_visible": False,
                    "landmarks": [],
                    "handedness": []
                }
            
            # Extract landmarks
            landmarks_list: List[np.ndarray] = []
            handedness_list: List[str] = []
            
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Convert landmarks to numpy array
                points = np.array([
                    [lm.x, lm.y, lm.z] 
                    for lm in hand_landmarks.landmark
                ])
                landmarks_list.append(points)
                
                # Get handedness (Left/Right)
                if results.multi_handedness and i < len(results.multi_handedness):
                    handedness = results.multi_handedness[i].classification[0].label
                    handedness_list.append(handedness)
                else:
                    handedness_list.append("Unknown")
            
            return {
                "num_hands": len(landmarks_list),
                "hands_visible": len(landmarks_list) > 0,
                "landmarks": landmarks_list,
                "handedness": handedness_list
            }
            
        except Exception as e:
            logger.warning(f"Hand detection error: {e}")
            return self._default_result()
    
    def _default_result(self) -> Dict[str, Any]:
        """Return default result when detection fails"""
        return {
            "num_hands": 0,
            "hands_visible": False,
            "landmarks": [],
            "handedness": []
        }
    
    def draw_hands(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        Draw hand landmarks on frame (for debugging).
        
        Args:
            frame: BGR image
            result: Result from detect()
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        if not self._initialized or self.hands is None:
            return annotated
        
        landmarks_list = result.get("landmarks", [])
        handedness_list = result.get("handedness", [])
        
        height, width = frame.shape[:2]
        
        for i, landmarks in enumerate(landmarks_list):
            # Draw landmarks as circles
            for point in landmarks:
                x = int(point[0] * width)
                y = int(point[1] * height)
                cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)
            
            # Draw connections (simplified)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
            ]
            
            for start, end in connections:
                if start < len(landmarks) and end < len(landmarks):
                    x1 = int(landmarks[start][0] * width)
                    y1 = int(landmarks[start][1] * height)
                    x2 = int(landmarks[end][0] * width)
                    y2 = int(landmarks[end][1] * height)
                    cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Label handedness
            if i < len(handedness_list):
                label = handedness_list[i]
                first_point = landmarks[0]
                x = int(first_point[0] * width)
                y = int(first_point[1] * height) - 10
                cv2.putText(
                    annotated,
                    label,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
        
        # Status text
        num_hands = result.get("num_hands", 0)
        cv2.putText(
            annotated,
            f"Hands: {num_hands}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if num_hands > 0 else (0, 0, 255),
            2
        )
        
        return annotated
    
    def close(self):
        """Release resources"""
        if self.hands is not None:
            self.hands.close()
            self.hands = None
