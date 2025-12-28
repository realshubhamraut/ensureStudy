"""
Flag Generator - Generates review flags based on metrics
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class FlagGenerator:
    """
    Generates flags for human review based on proctoring metrics.
    
    Flags are raised when specific metrics exceed thresholds,
    indicating potential cheating behavior that requires review.
    """
    
    # Threshold configuration (percentage of frames or count)
    THRESHOLDS: Dict[str, float] = {
        "gaze_diversion": 30.0,    # 30% of frames
        "head_deviation": 25.0,    # 25% of frames
        "face_absence": 20.0,      # 20% of frames
        "hand_anomaly": 20.0,      # 20% of frames
        "multi_person": 5.0,       # 5% of frames (very suspicious)
        "tab_switches": 3.0,       # 3 or more switches
        "prohibited_items": 10.0   # 10% of frames
    }
    
    # Critical flags that always require review
    CRITICAL_FLAGS = ["multi_person", "prohibited_items"]
    
    # Score threshold below which review is required
    REVIEW_SCORE_THRESHOLD = 60
    
    # Minimum flags for review
    MIN_FLAGS_FOR_REVIEW = 2
    
    def __init__(self, thresholds: Dict[str, float] = None):
        """
        Initialize flag generator with optional custom thresholds.
        
        Args:
            thresholds: Optional dict overriding default thresholds
        """
        self.thresholds = self.THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)
    
    def generate(self, metrics: Dict[str, float]) -> List[str]:
        """
        Generate flags based on metrics.
        
        Args:
            metrics: Dict with metric names and values
            
        Returns:
            List of flag names that were triggered
        """
        flags = []
        
        for metric, threshold in self.thresholds.items():
            value = metrics.get(metric, 0.0)
            
            if value >= threshold:
                flags.append(metric)
                logger.info(f"Flag triggered: {metric} ({value:.2f} >= {threshold})")
        
        return flags
    
    def generate_detailed(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate flags with detailed information.
        
        Args:
            metrics: Dict with metric names and values
            
        Returns:
            Dict with flags list and details for each flag
        """
        flags = []
        details = {}
        
        for metric, threshold in self.thresholds.items():
            value = metrics.get(metric, 0.0)
            
            triggered = value >= threshold
            severity = self._calculate_severity(metric, value, threshold)
            
            details[metric] = {
                "value": round(value, 2),
                "threshold": threshold,
                "triggered": triggered,
                "severity": severity
            }
            
            if triggered:
                flags.append(metric)
        
        return {
            "flags": flags,
            "details": details,
            "critical_flags": [f for f in flags if f in self.CRITICAL_FLAGS]
        }
    
    def _calculate_severity(self, metric: str, value: float, threshold: float) -> str:
        """
        Calculate severity level for a metric.
        
        Returns:
            'low', 'medium', 'high', or 'critical'
        """
        if value < threshold:
            return "low"
        
        if metric in self.CRITICAL_FLAGS:
            return "critical"
        
        ratio = value / threshold if threshold > 0 else 0
        
        if ratio >= 2.0:
            return "critical"
        elif ratio >= 1.5:
            return "high"
        elif ratio >= 1.0:
            return "medium"
        else:
            return "low"
    
    def requires_review(self, flags: List[str], score: int) -> bool:
        """
        Determine if manual review is required.
        
        Args:
            flags: List of triggered flags
            score: Integrity score
            
        Returns:
            True if human review is required
        """
        # Always review if critical flag triggered
        if any(f in self.CRITICAL_FLAGS for f in flags):
            return True
        
        # Review if score is below threshold
        if score < self.REVIEW_SCORE_THRESHOLD:
            return True
        
        # Review if enough flags triggered
        if len(flags) >= self.MIN_FLAGS_FOR_REVIEW:
            return True
        
        return False
    
    def get_review_priority(self, flags: List[str], score: int) -> str:
        """
        Get review priority level.
        
        Args:
            flags: List of triggered flags
            score: Integrity score
            
        Returns:
            'urgent', 'high', 'normal', or 'low'
        """
        # Critical flags = urgent
        if any(f in self.CRITICAL_FLAGS for f in flags):
            return "urgent"
        
        # Very low score = urgent
        if score < 40:
            return "urgent"
        
        # Moderately low score = high
        if score < 60:
            return "high"
        
        # Multiple flags = high
        if len(flags) >= 3:
            return "high"
        
        # Some flags = normal
        if len(flags) >= 1:
            return "normal"
        
        return "low"
