"""
Enhanced Behavior Analyzer

Combines SoftSkills pipeline with Proctoring's temporal analysis
to provide comprehensive behavioral insights for interview/communication sessions.

Features:
- Real-time visual behavior tracking (from SoftSkills)
- Temporal pattern analysis (from Proctoring LSTM)
- Engagement scoring
- Attention consistency
- Behavioral report generation
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BehaviorMetrics:
    """Aggregated behavior metrics from analysis."""
    # Engagement metrics
    engagement_score: float = 0.0
    attention_score: float = 0.0
    consistency_score: float = 0.0
    
    # Visual behavior
    eye_contact_avg: float = 0.0
    gaze_stability: float = 0.0
    posture_consistency: float = 0.0
    gesture_expressiveness: float = 0.0
    
    # Temporal patterns
    focus_duration_ratio: float = 0.0
    distraction_events: int = 0
    recovery_speed: float = 0.0
    
    # Flags
    attention_warnings: List[str] = field(default_factory=list)
    behavior_flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "engagement_score": round(self.engagement_score, 1),
            "attention_score": round(self.attention_score, 1),
            "consistency_score": round(self.consistency_score, 1),
            "eye_contact_avg": round(self.eye_contact_avg, 1),
            "gaze_stability": round(self.gaze_stability, 1),
            "posture_consistency": round(self.posture_consistency, 1),
            "gesture_expressiveness": round(self.gesture_expressiveness, 1),
            "focus_duration_ratio": round(self.focus_duration_ratio, 2),
            "distraction_events": self.distraction_events,
            "recovery_speed": round(self.recovery_speed, 2),
            "attention_warnings": self.attention_warnings,
            "behavior_flags": self.behavior_flags
        }


@dataclass
class BehaviorReport:
    """Final behavior analysis report."""
    session_id: str
    duration_seconds: float
    total_frames: int
    metrics: BehaviorMetrics
    overall_behavior_score: float
    engagement_level: str  # 'high', 'medium', 'low'
    attention_pattern: str  # 'consistent', 'variable', 'declining'
    top_strengths: List[str] = field(default_factory=list)
    areas_for_improvement: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "duration_seconds": round(self.duration_seconds, 1),
            "total_frames": self.total_frames,
            "metrics": self.metrics.to_dict(),
            "overall_behavior_score": round(self.overall_behavior_score, 1),
            "engagement_level": self.engagement_level,
            "attention_pattern": self.attention_pattern,
            "top_strengths": self.top_strengths,
            "areas_for_improvement": self.areas_for_improvement,
            "timestamp": self.timestamp
        }


class EnhancedBehaviorAnalyzer:
    """
    Enhanced behavior analyzer combining softskills and proctoring analysis.
    
    Provides temporal pattern recognition for:
    - Engagement tracking over time
    - Attention consistency
    - Behavioral pattern detection
    - Distraction recovery analysis
    """
    
    # Thresholds
    ATTENTION_THRESHOLD = 70  # Score below this triggers attention warning
    DISTRACTION_THRESHOLD = 3  # Consecutive low scores = distraction
    WINDOW_SIZE = 15  # Frames for temporal analysis
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.started_at = datetime.utcnow()
        
        # History buffers
        self.gaze_scores: deque = deque(maxlen=100)
        self.posture_scores: deque = deque(maxlen=100)
        self.gesture_scores: deque = deque(maxlen=100)
        self.attention_scores: deque = deque(maxlen=100)
        
        # Temporal tracking
        self.consecutive_low_attention = 0
        self.distraction_events = 0
        self.recovery_times: List[float] = []
        self.focus_frames = 0
        self.total_frames = 0
        
        # Flags
        self.active_warnings: List[str] = []
        self.behavior_flags: List[str] = []
        
        logger.info(f"[BehaviorAnalyzer] Session {session_id} started")
    
    def analyze_frame(
        self, 
        gaze_score: float = 0.0,
        posture_score: float = 0.0,
        gesture_score: float = 0.0,
        is_looking_at_camera: bool = True,
        is_upright: bool = True,
        hands_visible: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze a single frame and update temporal metrics.
        
        Args:
            gaze_score: Eye contact score (0-100)
            posture_score: Posture score (0-100)
            gesture_score: Gesture score (0-100)
            is_looking_at_camera: Whether looking at camera
            is_upright: Whether posture is upright
            hands_visible: Whether hands are visible
            
        Returns:
            Dict with current frame analysis
        """
        self.total_frames += 1
        
        # Store scores
        self.gaze_scores.append(gaze_score)
        self.posture_scores.append(posture_score)
        self.gesture_scores.append(gesture_score)
        
        # Calculate frame attention score
        attention = self._calculate_attention(
            gaze_score, posture_score, is_looking_at_camera, is_upright
        )
        self.attention_scores.append(attention)
        
        # Update temporal tracking
        self._update_temporal_metrics(attention)
        
        # Generate current warnings
        current_warnings = self._generate_warnings(
            attention, gaze_score, posture_score, is_looking_at_camera
        )
        
        return {
            "frame_number": self.total_frames,
            "attention_score": round(attention, 1),
            "is_focused": attention >= self.ATTENTION_THRESHOLD,
            "current_warnings": current_warnings,
            "consecutive_low_attention": self.consecutive_low_attention,
            "distraction_events": self.distraction_events
        }
    
    def _calculate_attention(
        self, 
        gaze_score: float, 
        posture_score: float,
        is_looking: bool,
        is_upright: bool
    ) -> float:
        """Calculate overall attention score from individual metrics."""
        # Weighted combination
        base_score = (gaze_score * 0.5) + (posture_score * 0.3)
        
        # Bonuses for positive behaviors
        if is_looking:
            base_score += 10
        if is_upright:
            base_score += 10
        
        return min(100, max(0, base_score))
    
    def _update_temporal_metrics(self, attention: float) -> None:
        """Update temporal tracking based on attention score."""
        if attention >= self.ATTENTION_THRESHOLD:
            self.focus_frames += 1
            
            # Check for recovery from distraction
            if self.consecutive_low_attention >= self.DISTRACTION_THRESHOLD:
                recovery_time = self.consecutive_low_attention
                self.recovery_times.append(recovery_time)
            
            self.consecutive_low_attention = 0
        else:
            self.consecutive_low_attention += 1
            
            # Mark distraction event
            if self.consecutive_low_attention == self.DISTRACTION_THRESHOLD:
                self.distraction_events += 1
    
    def _generate_warnings(
        self, 
        attention: float,
        gaze_score: float,
        posture_score: float,
        is_looking: bool
    ) -> List[str]:
        """Generate warnings for current frame."""
        warnings = []
        
        if not is_looking and gaze_score < 50:
            warnings.append("Look at the camera")
        
        if posture_score < 40:
            warnings.append("Improve your posture")
        
        if attention < self.ATTENTION_THRESHOLD:
            if self.consecutive_low_attention >= 3:
                warnings.append("Stay focused")
        
        self.active_warnings = warnings
        return warnings
    
    def get_temporal_metrics(self) -> Dict[str, Any]:
        """Get current temporal analysis metrics."""
        if not self.attention_scores:
            return {}
        
        scores = list(self.attention_scores)
        recent_scores = scores[-self.WINDOW_SIZE:] if len(scores) >= self.WINDOW_SIZE else scores
        
        # Calculate stability (inverse of variance)
        variance = np.var(recent_scores) if len(recent_scores) > 1 else 0
        stability = max(0, 100 - (variance * 0.5))
        
        # Calculate trend
        if len(recent_scores) >= 5:
            first_half = np.mean(recent_scores[:len(recent_scores)//2])
            second_half = np.mean(recent_scores[len(recent_scores)//2:])
            trend = "improving" if second_half > first_half + 5 else \
                    "declining" if second_half < first_half - 5 else "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "current_attention": round(recent_scores[-1], 1) if recent_scores else 0,
            "avg_attention": round(np.mean(recent_scores), 1),
            "attention_stability": round(stability, 1),
            "attention_trend": trend,
            "focus_ratio": round(self.focus_frames / max(1, self.total_frames), 2),
            "distraction_events": self.distraction_events,
            "avg_recovery_time": round(np.mean(self.recovery_times), 2) if self.recovery_times else 0
        }
    
    def generate_report(self) -> BehaviorReport:
        """Generate final behavior analysis report."""
        duration = (datetime.utcnow() - self.started_at).total_seconds()
        
        # Calculate aggregated metrics
        metrics = BehaviorMetrics()
        
        if self.gaze_scores:
            metrics.eye_contact_avg = np.mean(self.gaze_scores)
            metrics.gaze_stability = 100 - min(100, np.std(self.gaze_scores) * 2)
        
        if self.posture_scores:
            metrics.posture_consistency = 100 - min(100, np.std(self.posture_scores) * 2)
        
        if self.gesture_scores:
            metrics.gesture_expressiveness = np.mean(self.gesture_scores)
        
        if self.attention_scores:
            metrics.attention_score = np.mean(self.attention_scores)
            metrics.consistency_score = 100 - min(100, np.std(self.attention_scores) * 2)
        
        # Engagement score (combination)
        metrics.engagement_score = (
            metrics.attention_score * 0.4 +
            metrics.eye_contact_avg * 0.3 +
            metrics.posture_consistency * 0.2 +
            metrics.gesture_expressiveness * 0.1
        )
        
        # Temporal metrics
        metrics.focus_duration_ratio = self.focus_frames / max(1, self.total_frames)
        metrics.distraction_events = self.distraction_events
        metrics.recovery_speed = np.mean(self.recovery_times) if self.recovery_times else 0
        
        # Collect flags
        if self.distraction_events > 3:
            metrics.behavior_flags.append("Frequent distractions")
        if metrics.gaze_stability < 50:
            metrics.behavior_flags.append("Inconsistent eye contact")
        if metrics.posture_consistency < 50:
            metrics.behavior_flags.append("Posture needs improvement")
        
        # Determine engagement level
        if metrics.engagement_score >= 75:
            engagement_level = "high"
        elif metrics.engagement_score >= 50:
            engagement_level = "medium"
        else:
            engagement_level = "low"
        
        # Determine attention pattern
        if metrics.consistency_score >= 70:
            attention_pattern = "consistent"
        elif self.distraction_events > 5:
            attention_pattern = "declining"
        else:
            attention_pattern = "variable"
        
        # Generate strengths and improvements
        strengths = []
        improvements = []
        
        if metrics.eye_contact_avg >= 70:
            strengths.append("Strong eye contact")
        else:
            improvements.append("Maintain more consistent eye contact")
        
        if metrics.posture_consistency >= 70:
            strengths.append("Good posture throughout")
        else:
            improvements.append("Work on maintaining upright posture")
        
        if metrics.focus_duration_ratio >= 0.8:
            strengths.append("High focus and engagement")
        elif metrics.focus_duration_ratio < 0.6:
            improvements.append("Improve focus and minimize distractions")
        
        if metrics.gesture_expressiveness >= 60:
            strengths.append("Effective use of hand gestures")
        
        return BehaviorReport(
            session_id=self.session_id,
            duration_seconds=duration,
            total_frames=self.total_frames,
            metrics=metrics,
            overall_behavior_score=metrics.engagement_score,
            engagement_level=engagement_level,
            attention_pattern=attention_pattern,
            top_strengths=strengths[:3],
            areas_for_improvement=improvements[:3]
        )
    
    def reset(self):
        """Reset analyzer for new session."""
        self.started_at = datetime.utcnow()
        self.gaze_scores.clear()
        self.posture_scores.clear()
        self.gesture_scores.clear()
        self.attention_scores.clear()
        self.consecutive_low_attention = 0
        self.distraction_events = 0
        self.recovery_times = []
        self.focus_frames = 0
        self.total_frames = 0
        self.active_warnings = []
        self.behavior_flags = []


# Session storage
_behavior_sessions: Dict[str, EnhancedBehaviorAnalyzer] = {}


def get_behavior_analyzer(session_id: str) -> EnhancedBehaviorAnalyzer:
    """Get or create behavior analyzer for session."""
    if session_id not in _behavior_sessions:
        _behavior_sessions[session_id] = EnhancedBehaviorAnalyzer(session_id)
    return _behavior_sessions[session_id]


def cleanup_behavior_session(session_id: str) -> None:
    """Clean up behavior analyzer session."""
    if session_id in _behavior_sessions:
        del _behavior_sessions[session_id]
        logger.info(f"[BehaviorAnalyzer] Session {session_id} cleaned up")
