"""
Unified Cheat Score Calculator

Combines static (per-frame) and temporal (sequence) analysis
to produce a unified cheating probability score.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Weight for flags in final score calculation
FLAG_WEIGHTS = {
    'phone_detected': 0.25,
    'multiple_faces': 0.20,
    'no_face': 0.15,
    'book_detected': 0.15,
    'looking_away': 0.10,
    'suspicious_head_pose': 0.08,
    'suspicious_audio': 0.08,
    'tab_switch': 0.05,
    'mouth_open_talking': 0.05,
    'earpiece_detected': 0.20,
}


def calculate_cheat_score(
    static_prob: float = 0.0,
    temporal_prob: float = 0.0,
    active_flags: list = None,
    static_weight: float = 0.4,
    temporal_weight: float = 0.6
) -> Dict[str, Any]:
    """
    Calculate unified cheating score from multiple sources.
    
    Args:
        static_prob: Per-frame probability from LightGBM (0-1)
        temporal_prob: Temporal sequence probability from LSTM (0-1)
        active_flags: List of currently active warning flags
        static_weight: Weight for static score (default 0.4)
        temporal_weight: Weight for temporal score (default 0.6)
        
    Returns:
        Dict with unified score and breakdown
    """
    if active_flags is None:
        active_flags = []
    
    # Base score from models
    base_score = (static_weight * static_prob) + (temporal_weight * temporal_prob)
    
    # Add penalties for active flags
    flag_penalty = 0.0
    flag_breakdown = {}
    
    for flag in active_flags:
        flag_key = flag.lower().replace(' ', '_')
        weight = FLAG_WEIGHTS.get(flag_key, 0.03)  # Default small penalty
        flag_penalty += weight
        flag_breakdown[flag] = weight
    
    # Combined score (capped at 1.0)
    unified_score = min(1.0, base_score + flag_penalty)
    
    # Determine severity level
    if unified_score < 0.3:
        severity = 'low'
        integrity_score = 85 + int((1 - unified_score) * 15)  # 85-100
    elif unified_score < 0.5:
        severity = 'medium'
        integrity_score = 60 + int((0.5 - unified_score) * 50)  # 60-85
    elif unified_score < 0.7:
        severity = 'high'
        integrity_score = 40 + int((0.7 - unified_score) * 50)  # 40-60
    else:
        severity = 'critical'
        integrity_score = max(0, int((1.0 - unified_score) * 40))  # 0-40
    
    return {
        'unified_probability': round(unified_score, 4),
        'integrity_score': integrity_score,
        'severity': severity,
        'breakdown': {
            'static_contribution': round(static_prob * static_weight, 4),
            'temporal_contribution': round(temporal_prob * temporal_weight, 4),
            'flag_penalty': round(flag_penalty, 4),
            'flag_breakdown': flag_breakdown
        },
        'is_cheating': unified_score >= 0.5,
        'review_required': unified_score >= 0.4 or len(active_flags) >= 3
    }


def calculate_session_integrity(
    frame_scores: list,
    total_flags: Dict[str, int],
    tab_switch_count: int = 0,
    duration_seconds: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate final session integrity score.
    
    Args:
        frame_scores: List of per-frame unified scores
        total_flags: Dictionary of flag names to occurrence counts
        tab_switch_count: Number of tab switches detected
        duration_seconds: Session duration
        
    Returns:
        Final session integrity report
    """
    if not frame_scores:
        return {
            'final_integrity_score': 100,
            'average_cheat_probability': 0.0,
            'max_cheat_probability': 0.0,
            'suspicious_frame_percentage': 0.0,
            'severity': 'low',
            'review_required': False
        }
    
    # Calculate statistics
    avg_score = sum(frame_scores) / len(frame_scores)
    max_score = max(frame_scores)
    suspicious_frames = sum(1 for s in frame_scores if s >= 0.4)
    suspicious_percentage = (suspicious_frames / len(frame_scores)) * 100
    
    # Base integrity score
    base_integrity = 100 - int(avg_score * 60)  # Average contributes up to 60 points
    
    # Penalties
    penalties = 0
    
    # Penalty for high max score
    if max_score >= 0.8:
        penalties += 15
    elif max_score >= 0.6:
        penalties += 10
    
    # Penalty for suspicious frame percentage
    penalties += min(15, int(suspicious_percentage / 5))  # Up to 15 points
    
    # Penalty for tab switches
    penalties += min(10, tab_switch_count * 2)  # Up to 10 points
    
    # Calculate final score
    final_score = max(0, min(100, base_integrity - penalties))
    
    # Determine severity
    if final_score >= 80:
        severity = 'low'
    elif final_score >= 60:
        severity = 'medium'
    elif final_score >= 40:
        severity = 'high'
    else:
        severity = 'critical'
    
    # Generate summary flags
    summary_flags = []
    for flag, count in sorted(total_flags.items(), key=lambda x: -x[1])[:5]:
        if count > 0:
            summary_flags.append(f"{flag} ({count}x)")
    
    return {
        'final_integrity_score': final_score,
        'average_cheat_probability': round(avg_score, 4),
        'max_cheat_probability': round(max_score, 4),
        'suspicious_frame_percentage': round(suspicious_percentage, 2),
        'total_frames_analyzed': len(frame_scores),
        'tab_switch_count': tab_switch_count,
        'duration_seconds': round(duration_seconds, 1),
        'severity': severity,
        'top_flags': summary_flags,
        'review_required': final_score < 70 or suspicious_percentage > 20
    }
