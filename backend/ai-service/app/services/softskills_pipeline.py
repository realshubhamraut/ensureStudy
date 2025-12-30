"""
Soft Skills Pipeline Service

Unified pipeline for real-time soft skills evaluation.
Combines all analyzers:
- Fluency analysis (speech/text)
- Gaze/eye contact (video)
- Hand gestures (video)
- Posture (video)

Provides:
- Frame-by-frame video analysis
- Transcript analysis
- Aggregated scoring
- Real-time WebSocket support
"""

import cv2
import base64
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import asyncio

from .fluency_analyzer import get_fluency_analyzer, FluencyResult
from .gaze_analyzer import get_gaze_analyzer, GazeResult, MEDIAPIPE_AVAILABLE
from .gesture_analyzer import get_gesture_analyzer, GestureResult
from .posture_analyzer import get_posture_analyzer, PostureResult


# Score weights for overall calculation
SCORE_WEIGHTS = {
    "fluency": 0.30,
    "grammar": 0.20,
    "eye_contact": 0.15,
    "gesture": 0.10,
    "posture": 0.10,
    "expression": 0.10,  # Future
}


@dataclass
class VisualMetrics:
    """Aggregated visual metrics from video analysis."""
    frames_analyzed: int = 0
    
    # Eye contact
    eye_contact_score: float = 0.0
    gaze_center_ratio: float = 0.0
    is_looking_at_camera_ratio: float = 0.0
    
    # Head pose
    avg_head_yaw: float = 0.0
    avg_head_pitch: float = 0.0
    
    # Gestures
    gesture_score: float = 0.0
    hands_visible_ratio: float = 0.0
    hand_stability: float = 0.0
    
    # Posture
    posture_score: float = 0.0
    is_upright_ratio: float = 0.0
    shoulders_level_ratio: float = 0.0
    posture_stability: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "frames_analyzed": self.frames_analyzed,
            "eye_contact_score": round(self.eye_contact_score, 1),
            "gaze_center_ratio": round(self.gaze_center_ratio, 3),
            "is_looking_at_camera_ratio": round(self.is_looking_at_camera_ratio, 3),
            "avg_head_yaw": round(self.avg_head_yaw, 1),
            "avg_head_pitch": round(self.avg_head_pitch, 1),
            "gesture_score": round(self.gesture_score, 1),
            "hands_visible_ratio": round(self.hands_visible_ratio, 3),
            "hand_stability": round(self.hand_stability, 1),
            "posture_score": round(self.posture_score, 1),
            "is_upright_ratio": round(self.is_upright_ratio, 3),
            "shoulders_level_ratio": round(self.shoulders_level_ratio, 3),
            "posture_stability": round(self.posture_stability, 1),
        }


@dataclass
class SoftSkillsScore:
    """Final combined soft skills score."""
    overall_score: float = 0.0
    
    # Individual scores
    fluency_score: float = 0.0
    grammar_score: float = 0.0
    eye_contact_score: float = 0.0
    gesture_score: float = 0.0
    posture_score: float = 0.0
    expression_score: float = 0.0
    
    # Feedback
    strengths: List[str] = field(default_factory=list)
    areas_for_improvement: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "overall_score": round(self.overall_score, 1),
            "fluency_score": round(self.fluency_score, 1),
            "grammar_score": round(self.grammar_score, 1),
            "eye_contact_score": round(self.eye_contact_score, 1),
            "gesture_score": round(self.gesture_score, 1),
            "posture_score": round(self.posture_score, 1),
            "expression_score": round(self.expression_score, 1),
            "strengths": self.strengths,
            "areas_for_improvement": self.areas_for_improvement,
        }


@dataclass
class FrameResult:
    """Result from analyzing a single frame."""
    timestamp_ms: int
    
    # Gaze
    face_detected: bool = False
    gaze_direction: str = "unknown"
    gaze_score: float = 0.0
    is_looking_at_camera: bool = False
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    
    # Gestures
    hands_visible: bool = False
    num_hands: int = 0
    gesture_score: float = 0.0
    
    # Posture
    body_detected: bool = False
    posture_score: float = 0.0
    is_upright: bool = False
    shoulders_level: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "timestamp_ms": self.timestamp_ms,
            "face_detected": self.face_detected,
            "gaze_direction": self.gaze_direction,
            "gaze_score": round(self.gaze_score, 1),
            "is_looking_at_camera": self.is_looking_at_camera,
            "head_yaw": round(self.head_yaw, 1),
            "head_pitch": round(self.head_pitch, 1),
            "hands_visible": self.hands_visible,
            "num_hands": self.num_hands,
            "gesture_score": round(self.gesture_score, 1),
            "body_detected": self.body_detected,
            "posture_score": round(self.posture_score, 1),
            "is_upright": self.is_upright,
            "shoulders_level": self.shoulders_level,
        }


class SoftSkillsPipeline:
    """
    Unified pipeline for soft skills analysis.
    
    Combines:
    - Fluency analysis (text-based)
    - Gaze detection (MediaPipe Face Mesh)
    - Gesture analysis (MediaPipe Hands)
    - Posture estimation (MediaPipe Pose)
    """
    
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.session_start_time = None
        
        # Initialize analyzers
        self.fluency_analyzer = get_fluency_analyzer()
        
        # Visual analyzers (lazy init to avoid loading if not needed)
        self._gaze_analyzer = None
        self._gesture_analyzer = None
        self._posture_analyzer = None
        
        # Frame history for aggregation
        self.frame_results: List[FrameResult] = []
        self.max_frame_history = fps * 60 * 5  # 5 minutes max
    
    @property
    def gaze_analyzer(self):
        """Lazy initialization of gaze analyzer."""
        if self._gaze_analyzer is None:
            if MEDIAPIPE_AVAILABLE:
                self._gaze_analyzer = get_gaze_analyzer()
        return self._gaze_analyzer
    
    @property
    def gesture_analyzer(self):
        """Lazy initialization of gesture analyzer."""
        if self._gesture_analyzer is None:
            if MEDIAPIPE_AVAILABLE:
                self._gesture_analyzer = get_gesture_analyzer()
        return self._gesture_analyzer
    
    @property
    def posture_analyzer(self):
        """Lazy initialization of posture analyzer."""
        if self._posture_analyzer is None:
            if MEDIAPIPE_AVAILABLE:
                self._posture_analyzer = get_posture_analyzer()
        return self._posture_analyzer
    
    def start_session(self):
        """Start a new analysis session."""
        self.session_start_time = time.time()
        self.frame_results = []
        
        # Reset trackers
        if self._gesture_analyzer:
            self._gesture_analyzer.reset()
        if self._posture_analyzer:
            self._posture_analyzer.reset()
    
    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process a single video frame.
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            FrameResult with all detection results
        """
        timestamp_ms = int((time.time() - (self.session_start_time or time.time())) * 1000)
        
        result = FrameResult(timestamp_ms=timestamp_ms)
        
        # Gaze analysis
        if self.gaze_analyzer:
            gaze_result = self.gaze_analyzer.analyze_frame(frame)
            result.face_detected = gaze_result.face_detected
            result.gaze_direction = gaze_result.gaze_direction
            result.gaze_score = gaze_result.score
            result.is_looking_at_camera = gaze_result.is_looking_at_camera
            result.head_yaw = gaze_result.head_yaw
            result.head_pitch = gaze_result.head_pitch
        
        # Gesture analysis
        if self.gesture_analyzer:
            gesture_result = self.gesture_analyzer.analyze_frame(frame)
            result.hands_visible = gesture_result.hands_visible
            result.num_hands = gesture_result.num_hands
            result.gesture_score = gesture_result.score
        
        # Posture analysis
        if self.posture_analyzer:
            posture_result = self.posture_analyzer.analyze_frame(frame)
            result.body_detected = posture_result.body_detected
            result.posture_score = posture_result.score
            result.is_upright = posture_result.is_upright
            result.shoulders_level = posture_result.shoulders_level
        
        # Store result
        self.frame_results.append(result)
        if len(self.frame_results) > self.max_frame_history:
            self.frame_results.pop(0)
        
        return result
    
    def process_frame_base64(self, frame_base64: str) -> FrameResult:
        """
        Process a base64-encoded JPEG frame.
        
        Args:
            frame_base64: Base64 encoded JPEG image
        
        Returns:
            FrameResult
        """
        # Decode base64
        img_bytes = base64.b64decode(frame_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return FrameResult(timestamp_ms=0)
        
        return self.process_frame(frame)
    
    def analyze_fluency(
        self,
        transcript: str,
        duration_seconds: float,
        pause_ratio: float = 0.0
    ) -> FluencyResult:
        """
        Analyze speech fluency from transcript.
        """
        return self.fluency_analyzer.analyze(transcript, duration_seconds, pause_ratio)
    
    def aggregate_visual_metrics(self) -> VisualMetrics:
        """
        Aggregate frame results into overall visual metrics.
        """
        if not self.frame_results:
            return VisualMetrics()
        
        metrics = VisualMetrics(frames_analyzed=len(self.frame_results))
        
        # Calculate ratios and averages
        face_detected_count = sum(1 for r in self.frame_results if r.face_detected)
        looking_at_camera_count = sum(1 for r in self.frame_results if r.is_looking_at_camera)
        center_gaze_count = sum(1 for r in self.frame_results if r.gaze_direction == "center")
        hands_visible_count = sum(1 for r in self.frame_results if r.hands_visible)
        upright_count = sum(1 for r in self.frame_results if r.is_upright)
        level_shoulders_count = sum(1 for r in self.frame_results if r.shoulders_level)
        
        total = len(self.frame_results)
        
        # Eye contact
        if face_detected_count > 0:
            metrics.gaze_center_ratio = center_gaze_count / face_detected_count
            metrics.is_looking_at_camera_ratio = looking_at_camera_count / face_detected_count
            gaze_scores = [r.gaze_score for r in self.frame_results if r.face_detected]
            metrics.eye_contact_score = sum(gaze_scores) / len(gaze_scores) if gaze_scores else 0
            
            head_yaws = [r.head_yaw for r in self.frame_results if r.face_detected]
            head_pitches = [r.head_pitch for r in self.frame_results if r.face_detected]
            metrics.avg_head_yaw = np.mean(head_yaws) if head_yaws else 0
            metrics.avg_head_pitch = np.mean(head_pitches) if head_pitches else 0
        
        # Gestures
        metrics.hands_visible_ratio = hands_visible_count / total
        gesture_scores = [r.gesture_score for r in self.frame_results]
        metrics.gesture_score = sum(gesture_scores) / len(gesture_scores) if gesture_scores else 0
        metrics.hand_stability = self.gesture_analyzer.tracker.get_metrics()["avg_stability"] if self.gesture_analyzer else 0
        
        # Posture
        body_detected_count = sum(1 for r in self.frame_results if r.body_detected)
        if body_detected_count > 0:
            metrics.is_upright_ratio = upright_count / body_detected_count
            metrics.shoulders_level_ratio = level_shoulders_count / body_detected_count
            posture_scores = [r.posture_score for r in self.frame_results if r.body_detected]
            metrics.posture_score = sum(posture_scores) / len(posture_scores) if posture_scores else 0
            metrics.posture_stability = self.posture_analyzer.tracker.calculate_stability()["overall_stability"] if self.posture_analyzer else 0
        
        return metrics
    
    def calculate_final_score(
        self,
        fluency_result: Optional[FluencyResult] = None,
        visual_metrics: Optional[VisualMetrics] = None,
        grammar_score: float = 75.0  # Default placeholder
    ) -> SoftSkillsScore:
        """
        Calculate final weighted soft skills score.
        """
        score = SoftSkillsScore()
        
        # Set individual scores
        if fluency_result:
            score.fluency_score = fluency_result.score
        
        score.grammar_score = grammar_score
        
        if visual_metrics:
            score.eye_contact_score = visual_metrics.eye_contact_score
            score.gesture_score = visual_metrics.gesture_score
            score.posture_score = visual_metrics.posture_score
        
        score.expression_score = 75.0  # Placeholder for future
        
        # Calculate weighted overall
        score.overall_score = (
            score.fluency_score * SCORE_WEIGHTS["fluency"] +
            score.grammar_score * SCORE_WEIGHTS["grammar"] +
            score.eye_contact_score * SCORE_WEIGHTS["eye_contact"] +
            score.gesture_score * SCORE_WEIGHTS["gesture"] +
            score.posture_score * SCORE_WEIGHTS["posture"] +
            score.expression_score * SCORE_WEIGHTS["expression"]
        )
        
        # Generate feedback
        score.strengths = self._generate_strengths(score)
        score.areas_for_improvement = self._generate_improvements(score)
        
        return score
    
    def _generate_strengths(self, score: SoftSkillsScore) -> List[str]:
        """Generate strengths feedback."""
        strengths = []
        
        if score.fluency_score >= 80:
            strengths.append("Clear and fluent speech")
        if score.eye_contact_score >= 80:
            strengths.append("Excellent eye contact")
        if score.gesture_score >= 80:
            strengths.append("Natural hand gestures")
        if score.posture_score >= 80:
            strengths.append("Good posture and body language")
        
        if not strengths:
            strengths.append("Keep practicing to improve your skills!")
        
        return strengths
    
    def _generate_improvements(self, score: SoftSkillsScore) -> List[str]:
        """Generate areas for improvement."""
        improvements = []
        
        if score.fluency_score < 60:
            improvements.append("Work on reducing filler words and pauses")
        elif score.fluency_score < 80:
            improvements.append("Practice speaking at a consistent pace")
        
        if score.eye_contact_score < 60:
            improvements.append("Try to look at the camera more often")
        elif score.eye_contact_score < 80:
            improvements.append("Maintain eye contact while answering")
        
        if score.gesture_score < 60:
            improvements.append("Keep your hands visible and use natural gestures")
        
        if score.posture_score < 60:
            improvements.append("Sit up straight and keep your shoulders level")
        
        return improvements
    
    def reset(self):
        """Reset for new session."""
        self.start_session()
    
    def close(self):
        """Release resources."""
        if self._gaze_analyzer:
            self._gaze_analyzer.close()
        if self._gesture_analyzer:
            self._gesture_analyzer.close()
        if self._posture_analyzer:
            self._posture_analyzer.close()


# Singleton
_pipeline = None


def get_softskills_pipeline() -> SoftSkillsPipeline:
    """Get singleton SoftSkillsPipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = SoftSkillsPipeline()
    return _pipeline
