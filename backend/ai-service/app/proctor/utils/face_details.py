"""
Face Details Calculator - Extracts face features for proctoring

Ported from AutoOEP/VisionUtils/FaceDetailsCalculator.py
Uses MediaPipe face landmarks to extract:
- Iris position and ratio
- Mouth area and zone (open/closed)
- Head pose (x, y, z rotation)
- Gaze direction and zone
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


class FaceDetails:
    """
    Calculate face details from MediaPipe FaceLandmarker results.
    
    Extracts features for proctoring analysis:
    - Iris position (left/center/right)
    - Mouth zone (GREEN/YELLOW/ORANGE/RED based on openness)
    - Head pose angles (pitch, yaw, roll)
    - Gaze direction and radial zone
    """
    
    # MediaPipe face mesh landmark indices
    RIGHT_IRIS = [474, 475, 476, 477]
    LEFT_IRIS = [469, 470, 471, 472]
    L_H_LEFT = [33]   # Right eye rightmost mark
    L_H_RIGHT = [133]  # Right eye leftmost mark
    R_H_LEFT = [362]   # Left eye rightmost mark
    R_H_RIGHT = [263]  # Left eye leftmost mark
    INNER_LIPS = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 
                  14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
    
    # Head pose landmark indices
    POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]
    
    def __init__(self, result, image: np.ndarray):
        """
        Initialize FaceDetails from MediaPipe result.
        
        Args:
            result: MediaPipe FaceLandmarker result object
            image: BGR image used for detection
        """
        self.result = result
        self.image = image
        self.image_h, self.image_w = image.shape[:2]
        
        # Get face landmarks
        self.num_faces = len(result.face_landmarks) if hasattr(result, 'face_landmarks') else 0
        self.multi_face_landmarks = result.face_landmarks if self.num_faces > 0 else []
        self.face_landmarks = self.multi_face_landmarks[0] if self.num_faces > 0 else None
        
        # Initialize features
        self.iris_pos = ""
        self.iris_ratio = 0.0
        self.mouth_zone = ""
        self.mouth_area = 0.0
        self.x_rotation = 0.0  # pitch
        self.y_rotation = 0.0  # yaw
        self.z_rotation = 0.0  # roll
        self.radial_distance = 0.0
        self.gaze_direction = ""
        self.gaze_zone = ""
        
        # Calculate features if exactly one face detected
        if self.num_faces == 1 and self.face_landmarks is not None:
            try:
                self.iris_pos, self.iris_ratio = self._get_iris()
                self.mouth_area, self.mouth_zone = self._get_mouth()
                (self.x_rotation, self.y_rotation, self.z_rotation, 
                 self.radial_distance, self.gaze_direction, self.gaze_zone) = self._calculate_zone()
            except Exception as e:
                logger.warning(f"FaceDetails calculation error: {e}")
    
    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(point1 - point2)
    
    def _iris_position(self, iris_center: np.ndarray, right_point: np.ndarray, 
                       left_point: np.ndarray) -> Tuple[str, float]:
        """Determine iris position (left/center/right) based on ratio."""
        center_to_right_dist = self._euclidean_distance(iris_center, right_point)
        total_distance = self._euclidean_distance(right_point, left_point)
        
        if total_distance == 0:
            return "center", 0.5
            
        ratio = center_to_right_dist / total_distance
        
        if ratio <= 0.42:
            iris_position = "right"
        elif 0.42 < ratio <= 0.57:
            iris_position = "center"
        else:
            iris_position = "left"
            
        return iris_position, ratio
    
    def _get_mesh_points(self) -> np.ndarray:
        """Convert face landmarks to pixel coordinates."""
        return np.array([
            np.multiply([p.x, p.y], [self.image_w, self.image_h]).astype(int)
            for p in self.face_landmarks
        ])
    
    def _get_iris(self) -> Tuple[str, float]:
        """Calculate iris position and ratio."""
        mesh_points = self._get_mesh_points()
        
        # Get iris points
        l_iris_points = [mesh_points[idx] for idx in self.LEFT_IRIS]
        r_iris_points = [mesh_points[idx] for idx in self.RIGHT_IRIS]
        
        # Calculate iris centers
        (l_cx, l_cy), _ = cv2.minEnclosingCircle(np.array(l_iris_points))
        (r_cx, r_cy), _ = cv2.minEnclosingCircle(np.array(r_iris_points))
        
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        
        # Get eye corner points
        llm = mesh_points[self.R_H_RIGHT[0]]
        lrm = mesh_points[self.R_H_LEFT[0]]
        
        return self._iris_position(center_right, llm, lrm)
    
    def _get_mouth(self) -> Tuple[float, str]:
        """Calculate mouth area and zone (openness level)."""
        mesh_points = self._get_mesh_points()
        
        # Calculate mouth area using Shoelace formula
        points = np.array([mesh_points[idx] for idx in self.INNER_LIPS])
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        # Determine zone based on area
        if 0 <= area <= 160:
            zone = "GREEN"  # Closed
        elif 160 < area <= 500:
            zone = "YELLOW"  # Slightly open
        elif 500 < area <= 1000:
            zone = "ORANGE"  # Open
        else:
            zone = "RED"  # Wide open (talking/yawning)
            
        return area, zone
    
    def _initialize_camera_matrix(self) -> np.ndarray:
        """Initialize camera matrix for pose estimation."""
        focal_length = 1 * self.image_w
        return np.array([
            [focal_length, 0, self.image_h / 2],
            [0, focal_length, self.image_w / 2],
            [0, 0, 1]
        ])
    
    def _calculate_gaze_radius(self, x: float, y: float, z: float,
                               threshold_white: float = 10.0,
                               threshold_yellow: float = 15.0) -> Tuple[float, str]:
        """
        Calculate radial distance and color zone from head pose angles.
        
        Returns:
            (radial_distance, color_zone) where zone is 'white', 'yellow', or 'red'
        """
        radial_distance = np.sqrt(x**2 + y**2 + z**2)
        
        if radial_distance <= threshold_white:
            color_zone = "white"  # Looking at screen
        elif radial_distance <= threshold_yellow:
            color_zone = "yellow"  # Slight deviation
        else:
            color_zone = "red"  # Looking away
            
        return radial_distance, color_zone
    
    def _calculate_zone(self) -> Tuple[float, float, float, float, str, str]:
        """
        Calculate head pose and gaze zone.
        
        Returns:
            (x_rotation, y_rotation, z_rotation, radial_distance, gaze_direction, gaze_zone)
        """
        # Get mesh points as floats for solvePnP
        mesh_points = np.array([
            np.multiply([p.x, p.y], [self.image_w, self.image_h]).astype(float)
            for p in self.face_landmarks
        ])
        
        face_2d = []
        face_3d = []
        
        for idx in self.POSE_LANDMARKS:
            x, y = mesh_points[idx]
            z = self.face_landmarks[idx].z
            face_2d.append([x, y])
            face_3d.append([x, y, z])
        
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        
        # Camera matrix and distortion
        cam_matrix = self._initialize_camera_matrix()
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        # Solve PnP for pose
        success, rot_vec, trans_vec = cv2.solvePnP(
            face_3d, face_2d, cam_matrix, dist_matrix
        )
        
        if not success:
            return 0.0, 0.0, 0.0, 0.0, "forward", "white"
        
        # Get rotation matrix and Euler angles
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        x = angles[0] * 360  # pitch
        y = angles[1] * 360  # yaw
        z = angles[2] * 360  # roll
        
        # Determine gaze direction
        if y < -10:
            text = "left"
        elif y > 10:
            text = "right"
        elif x < -8:
            text = "down"
        elif x > 10:
            text = "up"
        else:
            text = "forward"
        
        radius, colour_zone = self._calculate_gaze_radius(x, y, z)
        
        return x, y, z, radius, text, colour_zone
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert face details to dictionary for API response."""
        return {
            "num_faces": self.num_faces,
            "iris_pos": self.iris_pos,
            "iris_ratio": round(self.iris_ratio, 4),
            "mouth_zone": self.mouth_zone,
            "mouth_area": round(self.mouth_area, 2),
            "x_rotation": round(self.x_rotation, 2),
            "y_rotation": round(self.y_rotation, 2),
            "z_rotation": round(self.z_rotation, 2),
            "radial_distance": round(self.radial_distance, 2),
            "gaze_direction": self.gaze_direction,
            "gaze_zone": self.gaze_zone
        }


def get_face_details(result, image: np.ndarray) -> Optional[FaceDetails]:
    """
    Create FaceDetails from MediaPipe result.
    
    Args:
        result: MediaPipe FaceLandmarker result
        image: BGR image
        
    Returns:
        FaceDetails instance or None on error
    """
    try:
        return FaceDetails(result, image)
    except Exception as e:
        logger.error(f"Failed to create FaceDetails: {e}")
        return None


def extract_face_features(result, image: np.ndarray) -> Dict[str, Any]:
    """
    Extract face features as dictionary.
    
    Args:
        result: MediaPipe FaceLandmarker result
        image: BGR image
        
    Returns:
        Dictionary of face features
    """
    details = get_face_details(result, image)
    if details:
        return details.to_dict()
    return {
        "num_faces": 0,
        "iris_pos": "",
        "iris_ratio": 0.0,
        "mouth_zone": "",
        "mouth_area": 0.0,
        "x_rotation": 0.0,
        "y_rotation": 0.0,
        "z_rotation": 0.0,
        "radial_distance": 0.0,
        "gaze_direction": "",
        "gaze_zone": ""
    }
