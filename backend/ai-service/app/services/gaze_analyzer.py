"""
Gaze Analyzer Service

Provides eye contact and gaze direction analysis using MediaPipe Face Mesh.
Detects:
- Gaze direction (center, left, right)
- Whether user is looking at camera
- Head pose (yaw, pitch, roll)

Uses MediaPipe for landmark detection.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

# Try to import mediapipe (optional dependency)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

# MediaPipe Face Mesh landmark indices
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468
LEFT_EYE_INNER_CORNER = 362
LEFT_EYE_OUTER_CORNER = 263
RIGHT_EYE_INNER_CORNER = 133
RIGHT_EYE_OUTER_CORNER = 33

# Thresholds
GAZE_CENTER_THRESHOLD = 0.15  # How far from 0.5 is still "center"
HEAD_YAW_THRESHOLD = 20.0  # degrees
HEAD_PITCH_THRESHOLD = 15.0  # degrees


@dataclass
class GazeResult:
    """Result from gaze analysis."""
    score: float
    gaze_direction: str  # center, left, right
    gaze_ratio: float
    is_looking_at_camera: bool
    head_yaw: float
    head_pitch: float
    head_roll: float
    face_detected: bool
    
    def to_dict(self) -> Dict:
        return {
            "score": round(self.score, 1),
            "gaze_direction": self.gaze_direction,
            "gaze_ratio": round(self.gaze_ratio, 3),
            "is_looking_at_camera": self.is_looking_at_camera,
            "head_yaw": round(self.head_yaw, 1),
            "head_pitch": round(self.head_pitch, 1),
            "head_roll": round(self.head_roll, 1),
            "face_detected": self.face_detected,
        }


def get_single_landmark(landmarks, idx: int, img_width: int, img_height: int) -> Tuple[int, int]:
    """Get single landmark as (x, y) tuple."""
    lm = landmarks[idx]
    return (int(lm.x * img_width), int(lm.y * img_height))


def calculate_gaze_ratio(landmarks, img_width: int, img_height: int) -> Tuple[float, float, float]:
    """
    Calculate gaze ratio for each eye.
    
    Returns:
        (left_ratio, right_ratio, average_ratio)
    """
    # Get iris centers
    left_iris = get_single_landmark(landmarks, LEFT_IRIS_CENTER, img_width, img_height)
    right_iris = get_single_landmark(landmarks, RIGHT_IRIS_CENTER, img_width, img_height)
    
    # Get eye corners
    left_inner = get_single_landmark(landmarks, LEFT_EYE_INNER_CORNER, img_width, img_height)
    left_outer = get_single_landmark(landmarks, LEFT_EYE_OUTER_CORNER, img_width, img_height)
    right_inner = get_single_landmark(landmarks, RIGHT_EYE_INNER_CORNER, img_width, img_height)
    right_outer = get_single_landmark(landmarks, RIGHT_EYE_OUTER_CORNER, img_width, img_height)
    
    # Calculate horizontal ratio for left eye
    left_eye_width = abs(left_outer[0] - left_inner[0])
    if left_eye_width > 0:
        left_ratio = (left_iris[0] - min(left_inner[0], left_outer[0])) / left_eye_width
    else:
        left_ratio = 0.5
    
    # Calculate horizontal ratio for right eye
    right_eye_width = abs(right_outer[0] - right_inner[0])
    if right_eye_width > 0:
        right_ratio = (right_iris[0] - min(right_inner[0], right_outer[0])) / right_eye_width
    else:
        right_ratio = 0.5
    
    # Average both eyes
    avg_ratio = (left_ratio + right_ratio) / 2
    
    return left_ratio, right_ratio, avg_ratio


def get_gaze_direction(gaze_ratio: float, center_threshold: float = GAZE_CENTER_THRESHOLD) -> str:
    """Determine gaze direction from ratio."""
    if abs(gaze_ratio - 0.5) <= center_threshold:
        return "center"
    return "left" if gaze_ratio < 0.5 else "right"


def estimate_head_pose(landmarks, img_width: int, img_height: int) -> Dict[str, float]:
    """
    Estimate head pose (pitch, yaw, roll) using facial landmarks.
    """
    # 3D model points (generic face model)
    model_points = np.array([
        [0.0, 0.0, 0.0],          # Nose tip
        [0.0, -330.0, -65.0],     # Chin
        [-225.0, 170.0, -135.0],  # Left eye corner
        [225.0, 170.0, -135.0],   # Right eye corner
        [-150.0, -150.0, -125.0], # Left mouth corner
        [150.0, -150.0, -125.0]   # Right mouth corner
    ], dtype=np.float64)
    
    # 2D image points
    landmark_indices = [1, 152, 33, 263, 61, 291]
    image_points = np.array([
        get_single_landmark(landmarks, idx, img_width, img_height)
        for idx in landmark_indices
    ], dtype=np.float64)
    
    # Camera matrix (approximate)
    focal_length = img_width
    center = (img_width / 2, img_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Assume no lens distortion
    dist_coeffs = np.zeros((4, 1))
    
    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return {"pitch": 0, "yaw": 0, "roll": 0}
    
    # Convert rotation vector to Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Extract Euler angles
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    
    if sy > 1e-6:
        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0
    
    return {
        "pitch": np.degrees(pitch),
        "yaw": np.degrees(yaw),
        "roll": np.degrees(roll)
    }


def calculate_eye_contact_score(
    gaze_ratio: float,
    head_yaw: float,
    head_pitch: float,
    gaze_center_threshold: float = GAZE_CENTER_THRESHOLD,
    head_yaw_threshold: float = HEAD_YAW_THRESHOLD,
    head_pitch_threshold: float = HEAD_PITCH_THRESHOLD
) -> Dict:
    """
    Calculate eye contact score (0-100).
    
    Weights:
    - Gaze direction: 60%
    - Head yaw: 25%
    - Head pitch: 15%
    """
    # Gaze score (higher when closer to 0.5)
    gaze_deviation = abs(gaze_ratio - 0.5)
    gaze_score = max(0, 100 - (gaze_deviation / 0.5) * 100)
    
    # Head yaw score
    yaw_deviation = abs(head_yaw)
    if yaw_deviation <= head_yaw_threshold:
        yaw_score = 100 - (yaw_deviation / head_yaw_threshold) * 30
    else:
        yaw_score = max(0, 70 - (yaw_deviation - head_yaw_threshold) * 2)
    
    # Head pitch score
    pitch_deviation = abs(head_pitch)
    if pitch_deviation <= head_pitch_threshold:
        pitch_score = 100 - (pitch_deviation / head_pitch_threshold) * 30
    else:
        pitch_score = max(0, 70 - (pitch_deviation - head_pitch_threshold) * 2)
    
    # Determine if looking at camera
    is_looking_at_camera = (
        gaze_deviation <= gaze_center_threshold and
        yaw_deviation <= head_yaw_threshold and
        pitch_deviation <= head_pitch_threshold
    )
    
    # Weighted average
    overall_score = gaze_score * 0.6 + yaw_score * 0.25 + pitch_score * 0.15
    
    return {
        "overall_score": round(overall_score, 1),
        "gaze_score": round(gaze_score, 1),
        "yaw_score": round(yaw_score, 1),
        "pitch_score": round(pitch_score, 1),
        "is_looking_at_camera": is_looking_at_camera,
        "gaze_direction": get_gaze_direction(gaze_ratio, gaze_center_threshold)
    }


class GazeAnalyzer:
    """
    Gaze analyzer using MediaPipe Face Mesh.
    """
    
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not installed. Run: pip install mediapipe")
        
        # Initialize Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enables iris tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # History for smoothing
        self.gaze_history = deque(maxlen=30)
    
    def analyze_frame(self, frame: np.ndarray) -> GazeResult:
        """
        Analyze a single frame for gaze direction.
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            GazeResult with analysis
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return GazeResult(
                score=0,
                gaze_direction="unknown",
                gaze_ratio=0.5,
                is_looking_at_camera=False,
                head_yaw=0,
                head_pitch=0,
                head_roll=0,
                face_detected=False
            )
        
        # Get first face landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate gaze ratio
        _, _, avg_ratio = calculate_gaze_ratio(landmarks, w, h)
        
        # Estimate head pose
        head_pose = estimate_head_pose(landmarks, w, h)
        
        # Calculate score
        score_result = calculate_eye_contact_score(
            avg_ratio,
            head_pose["yaw"],
            head_pose["pitch"]
        )
        
        # Update history for smoothing
        self.gaze_history.append(avg_ratio)
        
        return GazeResult(
            score=score_result["overall_score"],
            gaze_direction=score_result["gaze_direction"],
            gaze_ratio=avg_ratio,
            is_looking_at_camera=score_result["is_looking_at_camera"],
            head_yaw=head_pose["yaw"],
            head_pitch=head_pose["pitch"],
            head_roll=head_pose["roll"],
            face_detected=True
        )
    
    def get_smoothed_ratio(self) -> float:
        """Get smoothed gaze ratio from history."""
        if not self.gaze_history:
            return 0.5
        return sum(self.gaze_history) / len(self.gaze_history)
    
    def close(self):
        """Release resources."""
        self.face_mesh.close()


# Singleton instance
_gaze_analyzer = None


def get_gaze_analyzer() -> GazeAnalyzer:
    """Get singleton GazeAnalyzer instance."""
    global _gaze_analyzer
    if _gaze_analyzer is None:
        _gaze_analyzer = GazeAnalyzer()
    return _gaze_analyzer
