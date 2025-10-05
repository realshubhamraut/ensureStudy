"""
Posture Analyzer Service - Updated for MediaPipe Tasks API

Provides body posture analysis using MediaPipe PoseLandmarker.
Detects:
- Shoulder alignment (level vs tilted)
- Body lean (upright vs leaning)
- Stability (fidgeting detection)
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import os

# Try to import new MediaPipe Tasks API
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
    print(f"[PostureAnalyzer] MediaPipe {mp.__version__} loaded (Tasks API)")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"[PostureAnalyzer] MediaPipe not available: {e}")


# Posture thresholds
MAX_ACCEPTABLE_TILT = 8.0
MAX_ACCEPTABLE_LEAN = 10.0

# Score weights
TILT_WEIGHT = 0.25
LEAN_WEIGHT = 0.25
STABILITY_WEIGHT = 0.50


@dataclass
class PostureResult:
    """Result from posture analysis."""
    score: float
    shoulder_tilt: float
    lean_angle: float
    is_upright: bool
    shoulders_level: bool
    stability_score: float
    tilt_score: float
    lean_score: float
    body_detected: bool
    shoulder_center: Optional[Tuple[float, float]] = None
    lean_direction: str = "upright"
    assessment: str = "unknown"
    
    def to_dict(self) -> Dict:
        return {
            "score": round(self.score, 1),
            "shoulder_tilt": round(self.shoulder_tilt, 1),
            "lean_angle": round(self.lean_angle, 1),
            "is_upright": self.is_upright,
            "shoulders_level": self.shoulders_level,
            "stability_score": round(self.stability_score, 1),
            "tilt_score": round(self.tilt_score, 1),
            "lean_score": round(self.lean_score, 1),
            "body_detected": self.body_detected,
            "shoulder_center": self.shoulder_center,
            "lean_direction": self.lean_direction,
            "assessment": self.assessment,
        }


class PostureTracker:
    """Tracks posture over time for stability calculation."""
    
    def __init__(self, history_size: int = 30):
        self.history_size = history_size
        self.shoulder_center_history = deque(maxlen=history_size)
        self.shoulder_tilt_history = deque(maxlen=history_size)
        self.lean_angle_history = deque(maxlen=history_size)
    
    def update(self, shoulder_center, shoulder_tilt, lean_angle):
        self.shoulder_center_history.append(shoulder_center)
        self.shoulder_tilt_history.append(shoulder_tilt)
        self.lean_angle_history.append(lean_angle)
    
    def calculate_stability(self) -> Dict:
        valid_centers = [p for p in self.shoulder_center_history if p is not None]
        valid_tilts = [t for t in self.shoulder_tilt_history if t is not None]
        valid_leans = [l for l in self.lean_angle_history if l is not None]
        
        if len(valid_centers) >= 2:
            x_coords = [c[0] for c in valid_centers]
            y_coords = [c[1] for c in valid_centers]
            position_variance = np.std(x_coords) + np.std(y_coords)
            position_stability = max(0, 100 - position_variance * 2)
        else:
            position_stability = 100
        
        if len(valid_tilts) >= 2:
            tilt_variance = np.std(valid_tilts)
            tilt_stability = max(0, 100 - tilt_variance * 10)
        else:
            tilt_stability = 100
        
        if len(valid_leans) >= 2:
            lean_variance = np.std(valid_leans)
            lean_stability = max(0, 100 - lean_variance * 5)
        else:
            lean_stability = 100
        
        overall_stability = (position_stability + tilt_stability + lean_stability) / 3
        
        return {
            "position_stability": position_stability,
            "tilt_stability": tilt_stability,
            "lean_stability": lean_stability,
            "overall_stability": overall_stability
        }
    
    def reset(self):
        self.shoulder_center_history.clear()
        self.shoulder_tilt_history.clear()
        self.lean_angle_history.clear()


def calculate_posture_score(
    shoulder_tilt: float,
    lean_angle: float,
    stability: float,
    body_detected: bool = True
) -> Dict:
    if not body_detected:
        return {
            "overall_score": 0,
            "tilt_score": 0,
            "lean_score": 0,
            "stability_score": 0,
            "assessment": "no_body_detected"
        }
    
    tilt_deviation = abs(shoulder_tilt)
    if tilt_deviation <= 3:
        tilt_score = 100
    elif tilt_deviation <= MAX_ACCEPTABLE_TILT:
        tilt_score = 100 - ((tilt_deviation - 3) / (MAX_ACCEPTABLE_TILT - 3)) * 30
    else:
        tilt_score = max(30, 70 - (tilt_deviation - MAX_ACCEPTABLE_TILT) * 5)
    
    lean_deviation = abs(lean_angle)
    if lean_deviation <= 5:
        lean_score = 100
    elif lean_deviation <= MAX_ACCEPTABLE_LEAN:
        lean_score = 100 - ((lean_deviation - 5) / (MAX_ACCEPTABLE_LEAN - 5)) * 30
    else:
        lean_score = max(30, 70 - (lean_deviation - MAX_ACCEPTABLE_LEAN) * 3)
    
    stability_score = stability
    
    overall_score = (
        tilt_score * TILT_WEIGHT +
        lean_score * LEAN_WEIGHT +
        stability_score * STABILITY_WEIGHT
    )
    
    if overall_score >= 85:
        assessment = "excellent"
    elif overall_score >= 70:
        assessment = "good"
    elif overall_score >= 50:
        assessment = "fair"
    else:
        assessment = "needs_improvement"
    
    return {
        "overall_score": overall_score,
        "tilt_score": tilt_score,
        "lean_score": lean_score,
        "stability_score": stability_score,
        "assessment": assessment
    }


class PostureAnalyzer:
    """Posture analyzer using MediaPipe Tasks API (PoseLandmarker)."""
    
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not installed. Run: pip install mediapipe")
        
        model_path = self._get_model_path()
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        self.tracker = PostureTracker()
        print("[PostureAnalyzer] Initialized successfully")
    
    def _get_model_path(self) -> str:
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "pose_landmarker.task")
        
        if not os.path.exists(model_path):
            print("[PostureAnalyzer] Downloading pose_landmarker.task model...")
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, model_path)
            print("[PostureAnalyzer] Model downloaded successfully")
        
        return model_path
    
    def analyze_frame(self, frame: np.ndarray) -> PostureResult:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.pose_landmarker.detect(mp_image)
            
            if not result.pose_landmarks:
                return PostureResult(
                    score=0,
                    shoulder_tilt=0,
                    lean_angle=0,
                    is_upright=False,
                    shoulders_level=False,
                    stability_score=0,
                    tilt_score=0,
                    lean_score=0,
                    body_detected=False,
                    shoulder_center=None,
                    lean_direction="unknown",
                    assessment="no_body_detected"
                )
            
            # Get first pose landmarks
            landmarks = result.pose_landmarks[0]
            
            # Calculate shoulder metrics
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Shoulder positions
            left_s = (left_shoulder.x * w, left_shoulder.y * h)
            right_s = (right_shoulder.x * w, right_shoulder.y * h)
            
            # Shoulder metrics
            level_diff = abs(left_s[1] - right_s[1])
            shoulder_width = abs(left_s[0] - right_s[0])
            shoulder_tilt = np.degrees(np.arctan(level_diff / shoulder_width)) if shoulder_width > 0 else 0
            shoulder_center = ((left_s[0] + right_s[0]) / 2, (left_s[1] + right_s[1]) / 2)
            
            # Hip positions
            left_h = (left_hip.x * w, left_hip.y * h)
            right_h = (right_hip.x * w, right_hip.y * h)
            hip_center = ((left_h[0] + right_h[0]) / 2, (left_h[1] + right_h[1]) / 2)
            
            # Lean calculation
            horizontal_lean = shoulder_center[0] - hip_center[0]
            vertical_distance = abs(hip_center[1] - shoulder_center[1])
            lean_angle = np.degrees(np.arctan(horizontal_lean / vertical_distance)) if vertical_distance > 0 else 0
            
            if abs(lean_angle) < 3:
                lean_direction = "upright"
            elif lean_angle > 0:
                lean_direction = "leaning_right"
            else:
                lean_direction = "leaning_left"
            
            # Update tracker
            self.tracker.update(shoulder_center, shoulder_tilt, lean_angle)
            stability_metrics = self.tracker.calculate_stability()
            
            # Calculate score
            score_result = calculate_posture_score(
                shoulder_tilt,
                lean_angle,
                stability_metrics["overall_stability"]
            )
            
            return PostureResult(
                score=score_result["overall_score"],
                shoulder_tilt=shoulder_tilt,
                lean_angle=lean_angle,
                is_upright=abs(lean_angle) < 5,
                shoulders_level=shoulder_tilt < 5,
                stability_score=score_result["stability_score"],
                tilt_score=score_result["tilt_score"],
                lean_score=score_result["lean_score"],
                body_detected=True,
                shoulder_center=shoulder_center,
                lean_direction=lean_direction,
                assessment=score_result["assessment"]
            )
            
        except Exception as e:
            print(f"[PostureAnalyzer] Error: {e}")
            return PostureResult(
                score=0,
                shoulder_tilt=0,
                lean_angle=0,
                is_upright=False,
                shoulders_level=False,
                stability_score=0,
                tilt_score=0,
                lean_score=0,
                body_detected=False,
                shoulder_center=None,
                lean_direction="unknown",
                assessment="error"
            )
    
    def reset(self):
        self.tracker.reset()
    
    def close(self):
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()


# Singleton instance
_posture_analyzer = None


def get_posture_analyzer() -> PostureAnalyzer:
    global _posture_analyzer
    if _posture_analyzer is None:
        _posture_analyzer = PostureAnalyzer()
    return _posture_analyzer
