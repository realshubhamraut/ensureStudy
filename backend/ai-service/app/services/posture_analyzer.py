"""
Posture Analyzer Service

Provides body posture analysis using MediaPipe Pose.
Detects:
- Shoulder alignment (level vs tilted)
- Body lean (upright vs leaning)
- Stability (fidgeting detection)

Uses MediaPipe Pose for upper body detection.
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

# Posture thresholds
MAX_ACCEPTABLE_TILT = 8.0    # degrees
MAX_ACCEPTABLE_LEAN = 10.0   # degrees

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


def get_landmark(landmarks, idx: int, img_width: int, img_height: int) -> Tuple:
    """Get landmark coordinates."""
    lm = landmarks.landmark[idx]
    return (lm.x * img_width, lm.y * img_height, lm.z, lm.visibility)


def calculate_shoulder_metrics(landmarks, img_width: int, img_height: int) -> Dict:
    """Calculate shoulder alignment metrics."""
    left = get_landmark(landmarks, 11, img_width, img_height)  # Left shoulder
    right = get_landmark(landmarks, 12, img_width, img_height)  # Right shoulder
    
    # Shoulder level (y difference)
    level_diff = abs(left[1] - right[1])
    
    # Shoulder width (x difference)
    width = abs(left[0] - right[0])
    
    # Shoulder tilt angle (degrees)
    tilt_angle = np.degrees(np.arctan(level_diff / width)) if width > 0 else 0
    
    # Shoulder center
    center = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
    
    return {
        "left_shoulder": left[:2],
        "right_shoulder": right[:2],
        "center": center,
        "width": width,
        "level_diff": level_diff,
        "tilt_angle": tilt_angle,
        "is_level": tilt_angle < 5
    }


def calculate_body_lean(landmarks, img_width: int, img_height: int) -> Dict:
    """Calculate body lean from shoulder and hip alignment."""
    left_shoulder = get_landmark(landmarks, 11, img_width, img_height)
    right_shoulder = get_landmark(landmarks, 12, img_width, img_height)
    left_hip = get_landmark(landmarks, 23, img_width, img_height)
    right_hip = get_landmark(landmarks, 24, img_width, img_height)
    
    # Shoulder center
    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
    
    # Hip center
    hip_center_x = (left_hip[0] + right_hip[0]) / 2
    hip_center_y = (left_hip[1] + right_hip[1]) / 2
    
    # Horizontal lean (shoulder relative to hip)
    horizontal_lean = shoulder_center_x - hip_center_x
    
    # Vertical distance (for normalization)
    vertical_distance = abs(hip_center_y - shoulder_center_y)
    
    # Lean angle
    if vertical_distance > 0:
        lean_angle = np.degrees(np.arctan(horizontal_lean / vertical_distance))
    else:
        lean_angle = 0
    
    # Determine lean direction
    if abs(lean_angle) < 3:
        lean_direction = "upright"
    elif lean_angle > 0:
        lean_direction = "leaning_right"
    else:
        lean_direction = "leaning_left"
    
    return {
        "lean_angle": lean_angle,
        "lean_direction": lean_direction,
        "is_upright": abs(lean_angle) < 5
    }


class PostureTracker:
    """Tracks posture over time for stability calculation."""
    
    def __init__(self, history_size: int = 30):
        self.history_size = history_size
        self.shoulder_center_history = deque(maxlen=history_size)
        self.shoulder_tilt_history = deque(maxlen=history_size)
        self.lean_angle_history = deque(maxlen=history_size)
    
    def update(self, shoulder_center, shoulder_tilt, lean_angle):
        """Update tracking history."""
        self.shoulder_center_history.append(shoulder_center)
        self.shoulder_tilt_history.append(shoulder_tilt)
        self.lean_angle_history.append(lean_angle)
    
    def calculate_stability(self) -> Dict:
        """Calculate stability from historical data."""
        valid_centers = [p for p in self.shoulder_center_history if p is not None]
        valid_tilts = [t for t in self.shoulder_tilt_history if t is not None]
        valid_leans = [l for l in self.lean_angle_history if l is not None]
        
        # Position stability
        if len(valid_centers) >= 2:
            x_coords = [c[0] for c in valid_centers]
            y_coords = [c[1] for c in valid_centers]
            position_variance = np.std(x_coords) + np.std(y_coords)
            position_stability = max(0, 100 - position_variance * 2)
        else:
            position_stability = 100
        
        # Tilt stability
        if len(valid_tilts) >= 2:
            tilt_variance = np.std(valid_tilts)
            tilt_stability = max(0, 100 - tilt_variance * 10)
        else:
            tilt_stability = 100
        
        # Lean stability
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
        """Clear history."""
        self.shoulder_center_history.clear()
        self.shoulder_tilt_history.clear()
        self.lean_angle_history.clear()


def calculate_posture_score(
    shoulder_tilt: float,
    lean_angle: float,
    stability: float,
    body_detected: bool = True
) -> Dict:
    """
    Calculate posture score for interview context.
    
    Good posture:
    - Level shoulders
    - Upright body
    - Stable (not fidgeting)
    """
    if not body_detected:
        return {
            "overall_score": 0,
            "tilt_score": 0,
            "lean_score": 0,
            "stability_score": 0,
            "assessment": "no_body_detected"
        }
    
    # Shoulder level score
    tilt_deviation = abs(shoulder_tilt)
    if tilt_deviation <= 3:
        tilt_score = 100
    elif tilt_deviation <= MAX_ACCEPTABLE_TILT:
        tilt_score = 100 - ((tilt_deviation - 3) / (MAX_ACCEPTABLE_TILT - 3)) * 30
    else:
        tilt_score = max(30, 70 - (tilt_deviation - MAX_ACCEPTABLE_TILT) * 5)
    
    # Body lean score
    lean_deviation = abs(lean_angle)
    if lean_deviation <= 5:
        lean_score = 100
    elif lean_deviation <= MAX_ACCEPTABLE_LEAN:
        lean_score = 100 - ((lean_deviation - 5) / (MAX_ACCEPTABLE_LEAN - 5)) * 30
    else:
        lean_score = max(30, 70 - (lean_deviation - MAX_ACCEPTABLE_LEAN) * 3)
    
    # Stability score (already 0-100)
    stability_score = stability
    
    # Weighted average
    overall_score = (
        tilt_score * TILT_WEIGHT +
        lean_score * LEAN_WEIGHT +
        stability_score * STABILITY_WEIGHT
    )
    
    # Determine assessment
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
    """
    Posture analyzer using MediaPipe Pose.
    """
    
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not installed. Run: pip install mediapipe")
        
        # Initialize Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Stability tracker
        self.tracker = PostureTracker()
    
    def analyze_frame(self, frame: np.ndarray) -> PostureResult:
        """
        Analyze a single frame for posture.
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            PostureResult with analysis
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
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
        
        # Calculate metrics
        shoulder_metrics = calculate_shoulder_metrics(results.pose_landmarks, w, h)
        lean_metrics = calculate_body_lean(results.pose_landmarks, w, h)
        
        # Update tracker
        self.tracker.update(
            shoulder_metrics["center"],
            shoulder_metrics["tilt_angle"],
            lean_metrics["lean_angle"]
        )
        stability_metrics = self.tracker.calculate_stability()
        
        # Calculate score
        score_result = calculate_posture_score(
            shoulder_metrics["tilt_angle"],
            lean_metrics["lean_angle"],
            stability_metrics["overall_stability"]
        )
        
        return PostureResult(
            score=score_result["overall_score"],
            shoulder_tilt=shoulder_metrics["tilt_angle"],
            lean_angle=lean_metrics["lean_angle"],
            is_upright=lean_metrics["is_upright"],
            shoulders_level=shoulder_metrics["is_level"],
            stability_score=score_result["stability_score"],
            tilt_score=score_result["tilt_score"],
            lean_score=score_result["lean_score"],
            body_detected=True,
            shoulder_center=shoulder_metrics["center"],
            lean_direction=lean_metrics["lean_direction"],
            assessment=score_result["assessment"]
        )
    
    def reset(self):
        """Reset tracker for new session."""
        self.tracker.reset()
    
    def close(self):
        """Release resources."""
        self.pose.close()


# Singleton instance
_posture_analyzer = None


def get_posture_analyzer() -> PostureAnalyzer:
    """Get singleton PostureAnalyzer instance."""
    global _posture_analyzer
    if _posture_analyzer is None:
        _posture_analyzer = PostureAnalyzer()
    return _posture_analyzer
