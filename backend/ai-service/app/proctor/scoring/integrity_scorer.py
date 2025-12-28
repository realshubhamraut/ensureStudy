"""
Integrity Scorer - Computes integrity score from aggregated metrics
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class IntegrityScorer:
    """
    Computes integrity score from proctoring metrics.
    
    Formula:
        integrity_score = 100
            - (0.25 * gaze_diversion)
            - (0.20 * head_deviation)
            - (0.20 * face_absence)
            - (0.15 * hand_anomaly)
            - (0.10 * multi_person)
            - (0.10 * tab_switches_normalized)
    
    All input metrics are percentages (0-100) except tab_switches
    which is a raw count that gets normalized.
    """
    
    # Weight configuration
    WEIGHTS: Dict[str, float] = {
        "gaze_diversion": 0.25,
        "head_deviation": 0.20,
        "face_absence": 0.20,
        "hand_anomaly": 0.15,
        "multi_person": 0.10,
        "tab_switches": 0.10
    }
    
    # Maximum tab switches before maxing out the penalty
    MAX_TAB_SWITCHES = 10
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize scorer with optional custom weights.
        
        Args:
            weights: Optional dict overriding default weights
        """
        self.weights = self.WEIGHTS.copy()
        if weights:
            self.weights.update(weights)
        
        # Validate weights sum to ~1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, expected 1.0")
    
    def compute(self, metrics: Dict[str, float]) -> int:
        """
        Compute integrity score from metrics.
        
        Args:
            metrics: Dict with metric names and values (0-100 percentages)
            
        Returns:
            Integrity score (0-100, higher is better)
        """
        score = 100.0
        
        for metric, weight in self.weights.items():
            value = metrics.get(metric, 0.0)
            
            # Normalize tab_switches (raw count -> 0-100)
            if metric == "tab_switches":
                value = min(value, self.MAX_TAB_SWITCHES) * 10  # 10 switches = 100%
            
            # Subtract weighted penalty
            penalty = weight * value
            score -= penalty
            
            logger.debug(f"Metric {metric}: value={value:.2f}, weight={weight}, penalty={penalty:.2f}")
        
        # Clamp to 0-100
        final_score = max(0, min(100, int(round(score))))
        
        logger.info(f"Computed integrity score: {final_score}")
        return final_score
    
    def compute_breakdown(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute integrity score with detailed breakdown.
        
        Args:
            metrics: Dict with metric names and values
            
        Returns:
            Dict with score and breakdown of penalties
        """
        score = 100.0
        penalties = {}
        
        for metric, weight in self.weights.items():
            value = metrics.get(metric, 0.0)
            
            if metric == "tab_switches":
                value = min(value, self.MAX_TAB_SWITCHES) * 10
            
            penalty = weight * value
            penalties[metric] = {
                "value": round(value, 2),
                "weight": weight,
                "penalty": round(penalty, 2)
            }
            score -= penalty
        
        final_score = max(0, min(100, int(round(score))))
        
        return {
            "integrity_score": final_score,
            "raw_score": round(score, 2),
            "penalties": penalties,
            "total_penalty": round(100 - score, 2)
        }
    
    def get_grade(self, score: int) -> str:
        """
        Convert score to letter grade.
        
        Args:
            score: Integrity score (0-100)
            
        Returns:
            Grade: 'A' (excellent), 'B' (good), 'C' (warning), 'D' (concerning), 'F' (failed)
        """
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
