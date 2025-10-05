"""
Gesture Analyzer Service - Updated for MediaPipe Tasks API

Provides hand gesture analysis using MediaPipe HandLandmarker.
Detects:
- Hand visibility
- Movement frequency
- Gesture stability
- Natural vs fidgety gestures
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
    print(f"[GestureAnalyzer] MediaPipe {mp.__version__} loaded (Tasks API)")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"[GestureAnalyzer] MediaPipe not available: {e}")


# Movement thresholds
OPTIMAL_VELOCITY_MIN = 20.0
OPTIMAL_VELOCITY_MAX = 100.0

# Score weights
STABILITY_WEIGHT = 0.40
MOVEMENT_WEIGHT = 0.30
VISIBILITY_WEIGHT = 0.30


@dataclass
class GestureResult:
    """Result from gesture analysis."""
    score: float
    hands_visible: bool
    num_hands: int
    stability_score: float
    movement_score: float
    visibility_score: float
    left_hand_center: Optional[Tuple[float, float]] = None
    right_hand_center: Optional[Tuple[float, float]] = None
    assessment: str = "unknown"
    
    def to_dict(self) -> Dict:
        return {
            "score": round(self.score, 1),
            "hands_visible": self.hands_visible,
            "num_hands": self.num_hands,
            "stability_score": round(self.stability_score, 1),
            "movement_score": round(self.movement_score, 1),
            "visibility_score": round(self.visibility_score, 1),
            "left_hand_center": self.left_hand_center,
            "right_hand_center": self.right_hand_center,
            "assessment": self.assessment,
        }


class HandMovementTracker:
    """Tracks hand movements over time for stability analysis."""
    
    def __init__(self, history_size: int = 30, fps: int = 30):
        self.history_size = history_size
        self.fps = fps
        self.left_hand_history = deque(maxlen=history_size)
        self.right_hand_history = deque(maxlen=history_size)
    
    def update(self, left_hand_center=None, right_hand_center=None):
        self.left_hand_history.append(left_hand_center)
        self.right_hand_history.append(right_hand_center)
    
    def _calculate_movement(self, history) -> Dict:
        valid_positions = [p for p in history if p is not None]
        
        if len(valid_positions) < 2:
            return {
                "visible": False,
                "movement_distance": 0,
                "velocity": 0,
                "stability": 100
            }
        
        distances = []
        for i in range(1, len(valid_positions)):
            prev = valid_positions[i-1]
            curr = valid_positions[i]
            dist = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            distances.append(dist)
        
        total_distance = sum(distances)
        avg_velocity = np.mean(distances) * self.fps
        stability = max(0, 100 - avg_velocity * 0.5)
        
        return {
            "visible": True,
            "visibility_ratio": len(valid_positions) / len(history) if history else 0,
            "movement_distance": total_distance,
            "velocity": avg_velocity,
            "stability": stability
        }
    
    def get_metrics(self) -> Dict:
        left = self._calculate_movement(self.left_hand_history)
        right = self._calculate_movement(self.right_hand_history)
        
        hands_visible = left["visible"] or right["visible"]
        num_hands = int(left["visible"]) + int(right["visible"])
        
        if not hands_visible:
            avg_stability = 0
            avg_velocity = 0
        else:
            stabilities = []
            velocities = []
            if left["visible"]:
                stabilities.append(left["stability"])
                velocities.append(left["velocity"])
            if right["visible"]:
                stabilities.append(right["stability"])
                velocities.append(right["velocity"])
            avg_stability = np.mean(stabilities) if stabilities else 0
            avg_velocity = max(velocities) if velocities else 0
        
        return {
            "hands_visible": hands_visible,
            "num_hands": num_hands,
            "left_hand": left,
            "right_hand": right,
            "avg_stability": avg_stability,
            "avg_velocity": avg_velocity
        }
    
    def reset(self):
        self.left_hand_history.clear()
        self.right_hand_history.clear()


def calculate_gesture_score(
    hands_visible: bool,
    num_hands: int,
    stability: float,
    movement_velocity: float,
    visibility_ratio: float = 1.0
) -> Dict:
    # Visibility score
    if hands_visible:
        visibility_score = min(100, visibility_ratio * 100)
    else:
        visibility_score = 30
    
    # Movement score
    if OPTIMAL_VELOCITY_MIN <= movement_velocity <= OPTIMAL_VELOCITY_MAX:
        movement_score = 100
    elif movement_velocity < OPTIMAL_VELOCITY_MIN:
        movement_score = 70 + (movement_velocity / OPTIMAL_VELOCITY_MIN) * 30
    else:
        excess = movement_velocity - OPTIMAL_VELOCITY_MAX
        movement_score = max(30, 100 - excess * 0.5)
    
    stability_score = stability
    
    overall_score = (
        stability_score * STABILITY_WEIGHT +
        movement_score * MOVEMENT_WEIGHT +
        visibility_score * VISIBILITY_WEIGHT
    )
    
    if overall_score >= 80:
        assessment = "excellent"
    elif overall_score >= 60:
        assessment = "good"
    elif overall_score >= 40:
        assessment = "needs_improvement"
    else:
        assessment = "poor"
    
    return {
        "overall_score": overall_score,
        "visibility_score": visibility_score,
        "movement_score": movement_score,
        "stability_score": stability_score,
        "assessment": assessment
    }


class GestureAnalyzer:
    """Gesture analyzer using MediaPipe Tasks API (HandLandmarker)."""
    
    def __init__(self, fps: int = 30):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not installed. Run: pip install mediapipe")
        
        self.fps = fps
        
        # Download/find model
        model_path = self._get_model_path()
        
        # Configure HandLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        self.tracker = HandMovementTracker(fps=fps)
        print("[GestureAnalyzer] Initialized successfully")
    
    def _get_model_path(self) -> str:
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "hand_landmarker.task")
        
        if not os.path.exists(model_path):
            print("[GestureAnalyzer] Downloading hand_landmarker.task model...")
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("[GestureAnalyzer] Model downloaded successfully")
        
        return model_path
    
    def analyze_frame(self, frame: np.ndarray) -> GestureResult:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.hand_landmarker.detect(mp_image)
            
            left_center = None
            right_center = None
            num_hands = 0
            hands_visible = False
            
            if result.hand_landmarks:
                for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    label = handedness[0].category_name
                    
                    # Calculate center from all landmarks
                    x_coords = [lm.x * w for lm in hand_landmarks]
                    y_coords = [lm.y * h for lm in hand_landmarks]
                    center = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
                    
                    if label == "Left":
                        left_center = center
                    else:
                        right_center = center
                
                hands_visible = True
                num_hands = len(result.hand_landmarks)
            
            self.tracker.update(left_center, right_center)
            metrics = self.tracker.get_metrics()
            
            score_result = calculate_gesture_score(
                hands_visible=metrics["hands_visible"],
                num_hands=metrics["num_hands"],
                stability=metrics["avg_stability"],
                movement_velocity=metrics["avg_velocity"]
            )
            
            return GestureResult(
                score=score_result["overall_score"],
                hands_visible=hands_visible,
                num_hands=num_hands,
                stability_score=score_result["stability_score"],
                movement_score=score_result["movement_score"],
                visibility_score=score_result["visibility_score"],
                left_hand_center=left_center,
                right_hand_center=right_center,
                assessment=score_result["assessment"]
            )
            
        except Exception as e:
            print(f"[GestureAnalyzer] Error: {e}")
            return GestureResult(
                score=0,
                hands_visible=False,
                num_hands=0,
                stability_score=0,
                movement_score=0,
                visibility_score=0,
                assessment="error"
            )
    
    def reset(self):
        self.tracker.reset()
    
    def close(self):
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()


# Singleton instance
_gesture_analyzer = None


def get_gesture_analyzer() -> GestureAnalyzer:
    global _gesture_analyzer
    if _gesture_analyzer is None:
        _gesture_analyzer = GestureAnalyzer()
    return _gesture_analyzer
