"""
Metrics Aggregator - Aggregates detection metrics for a proctoring session
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MetricsAggregator:
    """
    Aggregates all detection metrics for a proctoring session.
    
    Tracks counts of suspicious behaviors across all frames
    and computes normalized ratios (0-100) for scoring.
    """
    
    session_id: str
    
    # Frame counter
    frame_count: int = 0
    
    # Detection counters
    face_absent_count: int = 0
    multi_face_count: int = 0
    head_deviation_count: int = 0
    gaze_diversion_count: int = 0
    hand_anomaly_count: int = 0
    prohibited_item_count: int = 0
    
    # Tab switches (from frontend events)
    tab_switch_count: int = 0
    
    # Motion tracking
    motion_spike_count: int = 0
    
    # Detailed tracking
    prohibited_items_seen: List[str] = field(default_factory=list)
    deviation_types: Dict[str, int] = field(default_factory=dict)
    
    # Timestamps
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_frame_at: datetime = field(default_factory=datetime.utcnow)
    
    def update(self, detections: Dict[str, Any]):
        """
        Update metrics from single frame detection results.
        
        Args:
            detections: Combined detection results from all detectors
        """
        self.frame_count += 1
        self.last_frame_at = datetime.utcnow()
        
        # Face absence
        if not detections.get("face_present", True):
            self.face_absent_count += 1
        
        # Multiple faces
        num_faces = detections.get("num_faces", 1)
        if num_faces > 1:
            self.multi_face_count += 1
        
        # Head deviation
        deviation = detections.get("deviation", "normal")
        if deviation not in ["normal", "unknown"]:
            self.head_deviation_count += 1
            self.deviation_types[deviation] = self.deviation_types.get(deviation, 0) + 1
        
        # Gaze diversion
        gaze = detections.get("gaze_direction", "center")
        if gaze not in ["center", "unknown"]:
            self.gaze_diversion_count += 1
        
        # Hand anomaly (no hands visible when expected)
        # During exam, hands should generally be visible
        if not detections.get("hands_visible", True):
            self.hand_anomaly_count += 1
        
        # Prohibited items
        if detections.get("has_prohibited", False):
            self.prohibited_item_count += 1
            items = detections.get("prohibited_items", [])
            for item in items:
                if item not in self.prohibited_items_seen:
                    self.prohibited_items_seen.append(item)
    
    def add_tab_switch(self):
        """Record a tab switch event"""
        self.tab_switch_count += 1
    
    def add_motion_spike(self):
        """Record a motion spike event"""
        self.motion_spike_count += 1
    
    def get_ratios(self) -> Dict[str, float]:
        """
        Calculate normalized ratios (0-100) for all metrics.
        
        Returns:
            Dict with metric names and their percentage values
        """
        total = max(1, self.frame_count)
        
        return {
            "face_absence": (self.face_absent_count / total) * 100,
            "multi_person": (self.multi_face_count / total) * 100,
            "head_deviation": (self.head_deviation_count / total) * 100,
            "gaze_diversion": (self.gaze_diversion_count / total) * 100,
            "hand_anomaly": (self.hand_anomaly_count / total) * 100,
            "prohibited_items": (self.prohibited_item_count / total) * 100,
            "tab_switches": float(self.tab_switch_count),
            "motion_spikes": float(self.motion_spike_count)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get complete metrics summary.
        
        Returns:
            Dict with all aggregated metrics
        """
        duration = (self.last_frame_at - self.started_at).total_seconds()
        
        return {
            "session_id": self.session_id,
            "frame_count": self.frame_count,
            "duration_seconds": duration,
            "ratios": self.get_ratios(),
            "counts": {
                "face_absent": self.face_absent_count,
                "multi_face": self.multi_face_count,
                "head_deviation": self.head_deviation_count,
                "gaze_diversion": self.gaze_diversion_count,
                "hand_anomaly": self.hand_anomaly_count,
                "prohibited_items": self.prohibited_item_count,
                "tab_switches": self.tab_switch_count
            },
            "prohibited_items_seen": self.prohibited_items_seen,
            "deviation_breakdown": self.deviation_types
        }
    
    def reset(self):
        """Reset all counters"""
        self.frame_count = 0
        self.face_absent_count = 0
        self.multi_face_count = 0
        self.head_deviation_count = 0
        self.gaze_diversion_count = 0
        self.hand_anomaly_count = 0
        self.prohibited_item_count = 0
        self.tab_switch_count = 0
        self.motion_spike_count = 0
        self.prohibited_items_seen = []
        self.deviation_types = {}
        self.started_at = datetime.utcnow()
        self.last_frame_at = datetime.utcnow()
