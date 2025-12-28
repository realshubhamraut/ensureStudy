"""
Gaze Tracker - Tracks eye gaze direction using facial landmarks

Refactored from: Artificial-Intelligence-based-Online-Exam-Proctoring-System/eye_tracker.py
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class GazeTracker:
    """
    Tracks eye gaze direction by analyzing iris position within eye regions.
    
    Uses eye landmarks (36-47) to extract eye regions, then applies
    adaptive thresholding to segment the iris/pupil and determine
    gaze direction based on white pixel distribution.
    """
    
    # Landmark indices for eyes (68-point model)
    LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
    
    # Ratio threshold for gaze detection
    GAZE_RATIO_THRESHOLD = 1.2
    
    def __init__(self):
        """Initialize gaze tracker"""
        pass
    
    def track(self, frame: np.ndarray, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Track gaze direction from eye landmarks.
        
        Args:
            frame: BGR image
            landmarks: 68-point facial landmarks as numpy array (68, 2)
            
        Returns:
            dict with:
                - gaze_direction: 'left', 'right', or 'center'
                - iris_pos: same as gaze_direction (compatibility)
                - iris_ratio: float ratio indicating gaze strength
                - left_eye_region: bounding box of left eye
                - right_eye_region: bounding box of right eye
        """
        if landmarks is None or len(landmarks) < 48:
            return self._default_result()
        
        try:
            # Extract eye regions
            left_eye_region = landmarks[self.LEFT_EYE_INDICES]
            right_eye_region = landmarks[self.RIGHT_EYE_INDICES]
            
            # Create mask and extract eyes
            mask = self._create_mask(frame)
            eyes = self._extract_eyes(mask, left_eye_region, right_eye_region, frame)
            
            # Get eye bounding boxes
            left_bbox = self._get_eye_bbox(left_eye_region)
            right_bbox = self._get_eye_bbox(right_eye_region)
            
            # Extract individual eye images
            left_eye_img = eyes[left_bbox[1]:left_bbox[3], left_bbox[0]:left_bbox[2]]
            right_eye_img = eyes[right_bbox[1]:right_bbox[3], right_bbox[0]:right_bbox[2]]
            
            if left_eye_img.size == 0 or right_eye_img.size == 0:
                return self._default_result()
            
            # Convert to grayscale
            left_gray = cv2.cvtColor(left_eye_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_eye_img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive threshold
            left_thresh = cv2.adaptiveThreshold(
                left_gray, 255, 
                cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            right_thresh = cv2.adaptiveThreshold(
                right_gray, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Analyze gaze
            direction, ratio = self._analyze_gaze(left_thresh, right_thresh)
            
            return {
                "gaze_direction": direction,
                "iris_pos": direction,  # Compatibility
                "iris_ratio": ratio,
                "left_eye_region": left_bbox,
                "right_eye_region": right_bbox
            }
            
        except Exception as e:
            logger.warning(f"Gaze tracking error: {e}")
            return self._default_result()
    
    def _create_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create black mask with frame dimensions"""
        height, width = frame.shape[:2]
        return np.zeros((height, width), dtype=np.uint8)
    
    def _extract_eyes(self, mask: np.ndarray, left_region: np.ndarray, 
                      right_region: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Extract eye regions from frame using mask"""
        # Draw and fill eye polygons on mask
        cv2.polylines(mask, [left_region.astype(np.int32)], True, 255, 2)
        cv2.fillPoly(mask, [left_region.astype(np.int32)], 255)
        cv2.polylines(mask, [right_region.astype(np.int32)], True, 255, 2)
        cv2.fillPoly(mask, [right_region.astype(np.int32)], 255)
        
        # Apply mask
        eyes = cv2.bitwise_and(frame, frame, mask=mask)
        return eyes
    
    def _get_eye_bbox(self, eye_region: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box for eye region (x_min, y_min, x_max, y_max)"""
        x_min = int(np.min(eye_region[:, 0]))
        x_max = int(np.max(eye_region[:, 0]))
        y_min = int(np.min(eye_region[:, 1]))
        y_max = int(np.max(eye_region[:, 1]))
        return (x_min, y_min, x_max, y_max)
    
    def _segment_eye_side(self, thresh_img: np.ndarray, side: str) -> int:
        """
        Count white pixels in left or right half of thresholded eye image.
        
        Args:
            thresh_img: Binary thresholded eye image
            side: 'left' or 'right'
            
        Returns:
            Count of white pixels
        """
        height, width = thresh_img.shape
        
        if side == 'left':
            half = thresh_img[0:height, 0:width // 2]
        else:
            half = thresh_img[0:height, width // 2:width]
        
        return cv2.countNonZero(half)
    
    def _analyze_gaze(self, left_eye_thresh: np.ndarray, 
                      right_eye_thresh: np.ndarray) -> Tuple[str, float]:
        """
        Analyze gaze direction from thresholded eye images.
        
        Logic:
        - If right side of right eye has more white pixels → looking left
        - If left side of left eye has more white pixels → looking right
        - Otherwise → center
        
        Returns:
            (direction, ratio)
        """
        # Left eye analysis (person's left eye)
        left_eye_left_side = self._segment_eye_side(left_eye_thresh, 'right')
        left_eye_right_side = self._segment_eye_side(left_eye_thresh, 'left')
        
        # Right eye analysis (person's right eye)
        right_eye_left_side = self._segment_eye_side(right_eye_thresh, 'right')
        right_eye_right_side = self._segment_eye_side(right_eye_thresh, 'left')
        
        # Calculate ratios
        try:
            # Looking left: more white on right side of right eye
            left_ratio = right_eye_right_side / max(1, right_eye_left_side)
            # Looking right: more white on left side of left eye  
            right_ratio = left_eye_left_side / max(1, left_eye_right_side)
            
            if left_ratio >= self.GAZE_RATIO_THRESHOLD:
                return ("left", left_ratio)
            elif right_ratio >= self.GAZE_RATIO_THRESHOLD:
                return ("right", right_ratio)
            else:
                return ("center", 1.0)
                
        except Exception:
            return ("center", 1.0)
    
    def _default_result(self) -> Dict[str, Any]:
        """Return default result when tracking fails"""
        return {
            "gaze_direction": "unknown",
            "iris_pos": "unknown",
            "iris_ratio": 1.0,
            "left_eye_region": None,
            "right_eye_region": None
        }
    
    def draw_gaze(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        Draw gaze visualization on frame (for debugging).
        
        Args:
            frame: BGR image
            result: Result from track()
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        direction = result.get("gaze_direction", "unknown")
        color = (0, 255, 0) if direction == "center" else (0, 0, 255)
        
        cv2.putText(
            annotated,
            f"Gaze: {direction}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        return annotated
