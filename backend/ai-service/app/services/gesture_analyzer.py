"""
Gesture Analyzer Service

Provides hand gesture analysis using MediaPipe Hands.
Detects:
- Hand visibility
- Movement frequency
- Gesture stability
- Natural vs fidgety gestures

Uses MediaPipe for hand detection and tracking.
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

# Movement thresholds
OPTIMAL_VELOCITY_MIN = 20.0   # Some movement is good
OPTIMAL_VELOCITY_MAX = 100.0  # Too much is fidgety

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


def get_hand_center(landmarks, img_width: int, img_height: int) -> Tuple[float, float]:
    """Calculate center of hand from landmarks."""
    x_coords = [lm.x * img_width for lm in landmarks.landmark]
    y_coords = [lm.y * img_height for lm in landmarks.landmark]
    return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))


def get_hand_spread(landmarks) -> float:
    """Calculate hand spread (openness)."""
    fingertips = [4, 8, 12, 16, 20]
    wrist = landmarks.landmark[0]
    distances = []
    for tip_idx in fingertips:
        tip = landmarks.landmark[tip_idx]
        dist = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2 + (tip.z - wrist.z)**2)
        distances.append(dist)
    return sum(distances) / len(distances)


class HandMovementTracker:
    """Tracks hand movements over time for stability analysis."""
    
    def __init__(self, history_size: int = 30, fps: int = 30):
        self.history_size = history_size
        self.fps = fps
        self.left_hand_history = deque(maxlen=history_size)
        self.right_hand_history = deque(maxlen=history_size)
    
    def update(self, left_hand_center=None, right_hand_center=None):
        """Update position history."""
        self.left_hand_history.append(left_hand_center)
        self.right_hand_history.append(right_hand_center)
    
    def _calculate_movement(self, history) -> Dict:
        """Calculate movement metrics from position history."""
        valid_positions = [p for p in history if p is not None]
        
        if len(valid_positions) < 2:
            return {
                "visible": False,
                "movement_distance": 0,
                "velocity": 0,
                "stability": 100
            }
        
        # Calculate frame-to-frame distances
        distances = []
        for i in range(1, len(valid_positions)):
            prev = valid_positions[i-1]
            curr = valid_positions[i]
            dist = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            distances.append(dist)
        
        total_distance = sum(distances)
        avg_velocity = np.mean(distances) * self.fps  # pixels per second
        
        # Stability: lower movement = higher stability
        stability = max(0, 100 - avg_velocity * 0.5)
        
        return {
            "visible": True,
            "visibility_ratio": len(valid_positions) / len(history),
            "movement_distance": total_distance,
            "velocity": avg_velocity,
            "stability": stability
        }
    
    def get_metrics(self) -> Dict:
        """Get movement metrics for both hands."""
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
        """Clear history."""
        self.left_hand_history.clear()
        self.right_hand_history.clear()


def calculate_gesture_score(
    hands_visible: bool,
    num_hands: int,
    stability: float,
    movement_velocity: float,
    visibility_ratio: float = 1.0
) -> Dict:
    """
    Calculate gesture score for interview context.
    
    Good gestures:
    - Visible hands (builds trust)
    - Moderate, natural movement
    - Stable, controlled gestures
    """
    # Visibility score
    if hands_visible:
        visibility_score = min(100, visibility_ratio * 100)
    else:
        visibility_score = 30  # Penalty for hidden hands
    
    # Movement score (optimal range)
    if OPTIMAL_VELOCITY_MIN <= movement_velocity <= OPTIMAL_VELOCITY_MAX:
        movement_score = 100
    elif movement_velocity < OPTIMAL_VELOCITY_MIN:
        # Too static
        movement_score = 70 + (movement_velocity / OPTIMAL_VELOCITY_MIN) * 30
    else:
        # Too fidgety
        excess = movement_velocity - OPTIMAL_VELOCITY_MAX
        movement_score = max(30, 100 - excess * 0.5)
    
    # Stability score (already 0-100)
    stability_score = stability
    
    # Weighted average
    overall_score = (
        stability_score * STABILITY_WEIGHT +
        movement_score * MOVEMENT_WEIGHT +
        visibility_score * VISIBILITY_WEIGHT
    )
    
    # Determine assessment
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
    """
    Gesture analyzer using MediaPipe Hands.
    """
    
    def __init__(self, fps: int = 30):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe is not installed. Run: pip install mediapipe")
        
        self.fps = fps
        
        # Initialize Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Movement tracker
        self.tracker = HandMovementTracker(fps=fps)
    
    def analyze_frame(self, frame: np.ndarray) -> GestureResult:
        """
        Analyze a single frame for hand gestures.
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            GestureResult with analysis
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        left_center = None
        right_center = None
        num_hands = 0
        hands_visible = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                label = handedness.classification[0].label
                center = get_hand_center(hand_landmarks, w, h)
                
                if label == "Left":
                    left_center = center
                else:
                    right_center = center
            
            hands_visible = True
            num_hands = len(results.multi_hand_landmarks)
        
        # Update tracker
        self.tracker.update(left_center, right_center)
        metrics = self.tracker.get_metrics()
        
        # Calculate score
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
    
    def reset(self):
        """Reset tracker for new session."""
        self.tracker.reset()
    
    def close(self):
        """Release resources."""
        self.hands.close()


# Singleton instance
_gesture_analyzer = None


def get_gesture_analyzer() -> GestureAnalyzer:
    """Get singleton GestureAnalyzer instance."""
    global _gesture_analyzer
    if _gesture_analyzer is None:
        _gesture_analyzer = GestureAnalyzer()
    return _gesture_analyzer
