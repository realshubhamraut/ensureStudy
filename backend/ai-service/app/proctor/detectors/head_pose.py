"""
Head Pose Estimator - Estimates head orientation using facial landmarks

Refactored from: Artificial-Intelligence-based-Online-Exam-Proctoring-System/head_pose_estimation.py
"""

import cv2
import numpy as np
import math
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class HeadPoseEstimator:
    """
    Estimates head pose (pitch, yaw, roll) using 68-point facial landmarks
    and PnP (Perspective-n-Point) algorithm.
    
    Uses 6 key facial points:
    - Nose tip (30)
    - Chin (8)
    - Left eye corner (36)
    - Right eye corner (45)
    - Left mouth corner (48)
    - Right mouth corner (54)
    """
    
    # 3D model points (generic face model)
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),            # Nose tip
        (0.0, -330.0, -65.0),       # Chin
        (-255.0, 170.0, -135.0),    # Left eye left corner
        (225.0, 170.0, -135.0),     # Right eye right corner
        (-150.0, -150.0, -125.0),   # Left mouth corner
        (150.0, -150.0, -125.0)     # Right mouth corner
    ], dtype=np.float64)
    
    # Landmark indices for the 6 key points
    LANDMARK_INDICES = [30, 8, 36, 45, 48, 54]
    
    # Thresholds for deviation classification (in degrees)
    PITCH_THRESHOLD = 25  # Up/down
    YAW_THRESHOLD = 25    # Left/right
    
    def __init__(self, frame_size: Tuple[int, int] = (480, 640)):
        """
        Initialize head pose estimator.
        
        Args:
            frame_size: (height, width) of expected frames
        """
        self.frame_size = frame_size
        self._init_camera_matrix(frame_size)
    
    def _init_camera_matrix(self, frame_size: Tuple[int, int]):
        """Initialize camera matrix based on frame size"""
        height, width = frame_size
        focal_length = width
        center = (width / 2, height / 2)
        
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.zeros((4, 1))
    
    def estimate(self, frame: np.ndarray, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Estimate head pose from facial landmarks.
        
        Args:
            frame: BGR image
            landmarks: 68-point facial landmarks as numpy array of shape (68, 2)
            
        Returns:
            dict with:
                - x_rotation: pitch angle (up/down)
                - y_rotation: yaw angle (left/right)
                - z_rotation: roll angle (tilt)
                - deviation: str ('normal', 'up', 'down', 'left', 'right')
                - rotation_vector: raw rotation vector
                - translation_vector: raw translation vector
        """
        if landmarks is None or len(landmarks) < 68:
            return self._default_result()
        
        # Update camera matrix if frame size changed
        h, w = frame.shape[:2]
        if (h, w) != self.frame_size:
            self._init_camera_matrix((h, w))
            self.frame_size = (h, w)
        
        try:
            # Extract the 6 key points
            image_points = np.array([
                landmarks[30],  # Nose tip
                landmarks[8],   # Chin
                landmarks[36],  # Left eye
                landmarks[45],  # Right eye
                landmarks[48],  # Left mouth
                landmarks[54]   # Right mouth
            ], dtype=np.float64)
            
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.MODEL_POINTS,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return self._default_result()
            
            # Convert rotation vector to Euler angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            angles = self._rotation_matrix_to_euler(rotation_matrix)
            
            pitch, yaw, roll = angles
            
            # Calculate deviation direction
            deviation = self._classify_deviation(pitch, yaw)
            
            return {
                "x_rotation": float(pitch),
                "y_rotation": float(yaw),
                "z_rotation": float(roll),
                "deviation": deviation,
                "rotation_vector": rotation_vector,
                "translation_vector": translation_vector
            }
            
        except Exception as e:
            logger.warning(f"Head pose estimation error: {e}")
            return self._default_result()
    
    def _rotation_matrix_to_euler(self, rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (pitch, yaw, roll).
        
        Returns angles in degrees.
        """
        sy = math.sqrt(
            rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2
        )
        
        singular = sy < 1e-6
        
        if not singular:
            pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = 0
        
        # Convert to degrees
        return (
            math.degrees(pitch),
            math.degrees(yaw),
            math.degrees(roll)
        )
    
    def _classify_deviation(self, pitch: float, yaw: float) -> str:
        """
        Classify head deviation based on pitch and yaw angles.
        
        Args:
            pitch: Up/down angle in degrees
            yaw: Left/right angle in degrees
            
        Returns:
            One of: 'normal', 'up', 'down', 'left', 'right'
        """
        # Check pitch first (up/down)
        if pitch >= self.PITCH_THRESHOLD:
            return "up"
        elif pitch <= -self.PITCH_THRESHOLD:
            return "down"
        
        # Then check yaw (left/right)
        if yaw >= self.YAW_THRESHOLD:
            return "right"
        elif yaw <= -self.YAW_THRESHOLD:
            return "left"
        
        return "normal"
    
    def _default_result(self) -> Dict[str, Any]:
        """Return default result when estimation fails"""
        return {
            "x_rotation": 0.0,
            "y_rotation": 0.0,
            "z_rotation": 0.0,
            "deviation": "unknown",
            "rotation_vector": None,
            "translation_vector": None
        }
    
    def draw_pose(self, frame: np.ndarray, landmarks: np.ndarray, 
                  result: Dict[str, Any]) -> np.ndarray:
        """
        Draw head pose visualization on frame (for debugging).
        
        Args:
            frame: BGR image
            landmarks: 68-point landmarks
            result: Result from estimate()
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        if result.get("rotation_vector") is None:
            return annotated
        
        try:
            # Draw nose direction line
            nose_tip = tuple(landmarks[30].astype(int))
            
            # Project a point 1000 units in front of nose
            nose_end_3d = np.array([[0.0, 0.0, 1000.0]])
            nose_end_2d, _ = cv2.projectPoints(
                nose_end_3d,
                result["rotation_vector"],
                result["translation_vector"],
                self.camera_matrix,
                self.dist_coeffs
            )
            nose_end = tuple(nose_end_2d[0][0].astype(int))
            
            # Draw arrow
            cv2.arrowedLine(annotated, nose_tip, nose_end, (0, 255, 255), 2)
            
            # Add text
            deviation = result.get("deviation", "unknown")
            color = (0, 255, 0) if deviation == "normal" else (0, 0, 255)
            cv2.putText(
                annotated, 
                f"Head: {deviation}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
            
        except Exception as e:
            logger.warning(f"Error drawing pose: {e}")
        
        return annotated
