"""
Gaze Analyzer Service - Updated for MediaPipe Tasks API

Provides eye contact and gaze direction analysis using MediaPipe FaceLandmarker.
Detects:
- Gaze direction (center, left, right)
- Whether user is looking at camera
- Head pose (yaw, pitch, roll)
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import os

# Try to import new MediaPipe Tasks API
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
    print(f"[GazeAnalyzer] MediaPipe {mp.__version__} loaded (Tasks API)")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"[GazeAnalyzer] MediaPipe not available: {e}")


# Face mesh landmark indices (same as before)
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468
LEFT_EYE_INNER_CORNER = 362
LEFT_EYE_OUTER_CORNER = 263
RIGHT_EYE_INNER_CORNER = 133
RIGHT_EYE_OUTER_CORNER = 33

# Thresholds
GAZE_CENTER_THRESHOLD = 0.15
HEAD_YAW_THRESHOLD = 20.0
HEAD_PITCH_THRESHOLD = 15.0


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


def get_gaze_direction(gaze_ratio: float, center_threshold: float = GAZE_CENTER_THRESHOLD) -> str:
    """Determine gaze direction from ratio."""
    if abs(gaze_ratio - 0.5) <= center_threshold:
        return "center"
    return "left" if gaze_ratio < 0.5 else "right"


def calculate_eye_contact_score(
    gaze_ratio: float,
    head_yaw: float = 0.0,
    head_pitch: float = 0.0,
) -> Dict:
    """Calculate eye contact score (0-100)."""
    # Gaze score (higher when closer to 0.5)
    gaze_deviation = abs(gaze_ratio - 0.5)
    gaze_score = max(0, 100 - (gaze_deviation / 0.5) * 100)
    
    # Head yaw score
    yaw_deviation = abs(head_yaw)
    if yaw_deviation <= HEAD_YAW_THRESHOLD:
        yaw_score = 100 - (yaw_deviation / HEAD_YAW_THRESHOLD) * 30
    else:
        yaw_score = max(0, 70 - (yaw_deviation - HEAD_YAW_THRESHOLD) * 2)
    
    # Head pitch score
    pitch_deviation = abs(head_pitch)
    if pitch_deviation <= HEAD_PITCH_THRESHOLD:
        pitch_score = 100 - (pitch_deviation / HEAD_PITCH_THRESHOLD) * 30
    else:
        pitch_score = max(0, 70 - (pitch_deviation - HEAD_PITCH_THRESHOLD) * 2)
    
    # Determine if looking at camera
    is_looking_at_camera = (
        gaze_deviation <= GAZE_CENTER_THRESHOLD and
        yaw_deviation <= HEAD_YAW_THRESHOLD and
        pitch_deviation <= HEAD_PITCH_THRESHOLD
    )
    
    # Weighted average
    overall_score = gaze_score * 0.6 + yaw_score * 0.25 + pitch_score * 0.15
    
    return {
        "overall_score": round(overall_score, 1),
        "is_looking_at_camera": is_looking_at_camera,
        "gaze_direction": get_gaze_direction(gaze_ratio)
    }


class GazeAnalyzer:
    """
    Gaze analyzer using MediaPipe Tasks API (FaceLandmarker).
    """
    
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not installed. Run: pip install mediapipe")
        
        # Download/find model file
        model_path = self._get_model_path()
        
        # Configure FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True
        )
        
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # History for smoothing
        self.gaze_history = deque(maxlen=30)
        print("[GazeAnalyzer] Initialized successfully")
    
    def _get_model_path(self) -> str:
        """Get path to face landmarker model, download if needed."""
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "face_landmarker.task")
        
        if not os.path.exists(model_path):
            print("[GazeAnalyzer] Downloading face_landmarker.task model...")
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("[GazeAnalyzer] Model downloaded successfully")
        
        return model_path
    
    def analyze_frame(self, frame: np.ndarray) -> GazeResult:
        """
        Analyze a single frame for gaze direction.
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            GazeResult with analysis
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect faces
            result = self.face_landmarker.detect(mp_image)
            
            if not result.face_landmarks:
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
            landmarks = result.face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Calculate gaze ratio from iris position
            gaze_ratio = self._calculate_gaze_ratio(landmarks, w, h)
            
            # Get head pose from transformation matrix
            head_yaw, head_pitch, head_roll = 0.0, 0.0, 0.0
            if result.facial_transformation_matrixes:
                head_yaw, head_pitch, head_roll = self._extract_head_pose(
                    result.facial_transformation_matrixes[0]
                )
            
            # Calculate score
            score_result = calculate_eye_contact_score(gaze_ratio, head_yaw, head_pitch)
            
            # Update history
            self.gaze_history.append(gaze_ratio)
            
            return GazeResult(
                score=score_result["overall_score"],
                gaze_direction=score_result["gaze_direction"],
                gaze_ratio=gaze_ratio,
                is_looking_at_camera=score_result["is_looking_at_camera"],
                head_yaw=head_yaw,
                head_pitch=head_pitch,
                head_roll=head_roll,
                face_detected=True
            )
            
        except Exception as e:
            print(f"[GazeAnalyzer] Error analyzing frame: {e}")
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
    
    def _calculate_gaze_ratio(self, landmarks, width: int, height: int) -> float:
        """Calculate gaze ratio from iris landmarks."""
        try:
            # Get iris centers
            left_iris = landmarks[LEFT_IRIS_CENTER]
            right_iris = landmarks[RIGHT_IRIS_CENTER]
            
            # Get eye corners
            left_inner = landmarks[LEFT_EYE_INNER_CORNER]
            left_outer = landmarks[LEFT_EYE_OUTER_CORNER]
            right_inner = landmarks[RIGHT_EYE_INNER_CORNER]
            right_outer = landmarks[RIGHT_EYE_OUTER_CORNER]
            
            # Calculate horizontal ratio for left eye
            left_eye_width = abs(left_outer.x - left_inner.x)
            if left_eye_width > 0:
                left_ratio = (left_iris.x - min(left_inner.x, left_outer.x)) / left_eye_width
            else:
                left_ratio = 0.5
            
            # Calculate horizontal ratio for right eye
            right_eye_width = abs(right_outer.x - right_inner.x)
            if right_eye_width > 0:
                right_ratio = (right_iris.x - min(right_inner.x, right_outer.x)) / right_eye_width
            else:
                right_ratio = 0.5
            
            # Average both eyes
            return (left_ratio + right_ratio) / 2
            
        except (IndexError, AttributeError):
            return 0.5
    
    def _extract_head_pose(self, transformation_matrix) -> Tuple[float, float, float]:
        """Extract yaw, pitch, roll from transformation matrix."""
        try:
            # Convert to numpy array if needed
            if hasattr(transformation_matrix, 'data'):
                matrix = np.array(transformation_matrix.data).reshape(4, 4)
            else:
                matrix = np.array(transformation_matrix).reshape(4, 4)
            
            # Extract rotation matrix
            rotation = matrix[:3, :3]
            
            # Calculate Euler angles
            sy = np.sqrt(rotation[0, 0]**2 + rotation[1, 0]**2)
            
            if sy > 1e-6:
                pitch = np.arctan2(rotation[2, 1], rotation[2, 2])
                yaw = np.arctan2(-rotation[2, 0], sy)
                roll = np.arctan2(rotation[1, 0], rotation[0, 0])
            else:
                pitch = np.arctan2(-rotation[1, 2], rotation[1, 1])
                yaw = np.arctan2(-rotation[2, 0], sy)
                roll = 0
            
            return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)
            
        except Exception:
            return 0.0, 0.0, 0.0
    
    def get_smoothed_ratio(self) -> float:
        """Get smoothed gaze ratio from history."""
        if not self.gaze_history:
            return 0.5
        return sum(self.gaze_history) / len(self.gaze_history)
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'face_landmarker'):
            self.face_landmarker.close()


# Singleton instance
_gaze_analyzer = None


def get_gaze_analyzer() -> GazeAnalyzer:
    """Get singleton GazeAnalyzer instance."""
    global _gaze_analyzer
    if _gaze_analyzer is None:
        _gaze_analyzer = GazeAnalyzer()
    return _gaze_analyzer
