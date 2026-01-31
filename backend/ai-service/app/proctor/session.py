"""
Proctor Session - Manages a single proctoring session

Enhanced with AutoOEP temporal behavior analysis and static classification.
"""

import uuid
import logging
import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional, List

from .detectors import (
    FaceDetector,
    HeadPoseEstimator,
    GazeTracker,
    ProhibitedObjectDetector,
    HandDetector,
    AudioDetector,
    BlinkDetector,
    FaceVerifier
)
from .metrics import MetricsAggregator
from .scoring import IntegrityScorer, FlagGenerator
from .scoring.cheat_score import calculate_cheat_score, calculate_session_integrity
from .temporal_predictor import TemporalPredictor, get_temporal_predictor
from .static_classifier import StaticClassifier, get_static_classifier
from .utils.frame_quality import check_frame_quality
from .utils.logging import log_session_start, log_session_end, log_flag_triggered

logger = logging.getLogger(__name__)


class ProctorSession:
    """
    Manages a single proctoring session.
    
    Orchestrates all detectors, aggregates metrics,
    and computes integrity scores in real-time.
    """
    
    def __init__(
        self,
        assessment_id: str,
        student_id: str,
        session_id: Optional[str] = None
    ):
        """
        Initialize a new proctoring session.
        
        Args:
            assessment_id: ID of the assessment being proctored
            student_id: ID of the student being proctored
            session_id: Optional custom session ID (auto-generated if not provided)
        """
        self.id = session_id or f"EXM_{uuid.uuid4().hex[:6].upper()}"
        self.assessment_id = assessment_id
        self.student_id = student_id
        self.started_at = datetime.utcnow()
        self.is_active = True
        
        # Initialize detectors (lazy loaded)
        self._face_detector: Optional[FaceDetector] = None
        self._head_pose: Optional[HeadPoseEstimator] = None
        self._gaze_tracker: Optional[GazeTracker] = None
        self._object_detector: Optional[ProhibitedObjectDetector] = None
        self._hand_detector: Optional[HandDetector] = None
        self._audio_detector: Optional[AudioDetector] = None
        self._blink_detector: Optional[BlinkDetector] = None
        self._face_verifier: Optional[FaceVerifier] = None
        
        # Initialize metrics, scoring, and flagging
        self.metrics = MetricsAggregator(session_id=self.id)
        self.scorer = IntegrityScorer()
        self.flagger = FlagGenerator()
        
        # Previous frame for motion detection
        self._prev_frame: Optional[np.ndarray] = None
        
        # AutoOEP temporal analysis
        self._temporal_predictor: Optional[TemporalPredictor] = None
        self._static_classifier: Optional[StaticClassifier] = None
        self.cheat_score_history: List[float] = []
        self.flag_counts: Dict[str, int] = {}
        
        # Log session start
        log_session_start(self.id, assessment_id, student_id)
        
        logger.info(f"Proctoring session started: {self.id}")
    
    @property
    def face_detector(self) -> FaceDetector:
        """Lazy load face detector"""
        if self._face_detector is None:
            self._face_detector = FaceDetector()
        return self._face_detector
    
    @property
    def head_pose(self) -> HeadPoseEstimator:
        """Lazy load head pose estimator"""
        if self._head_pose is None:
            self._head_pose = HeadPoseEstimator()
        return self._head_pose
    
    @property
    def gaze_tracker(self) -> GazeTracker:
        """Lazy load gaze tracker"""
        if self._gaze_tracker is None:
            self._gaze_tracker = GazeTracker()
        return self._gaze_tracker
    
    @property
    def object_detector(self) -> ProhibitedObjectDetector:
        """Lazy load object detector"""
        if self._object_detector is None:
            self._object_detector = ProhibitedObjectDetector()
        return self._object_detector
    
    @property
    def hand_detector(self) -> HandDetector:
        """Lazy load hand detector"""
        if self._hand_detector is None:
            self._hand_detector = HandDetector()
        return self._hand_detector
    
    @property
    def audio_detector(self) -> AudioDetector:
        """Lazy load audio detector"""
        if self._audio_detector is None:
            self._audio_detector = AudioDetector()
        return self._audio_detector
    
    @property
    def blink_detector(self) -> BlinkDetector:
        """Lazy load blink detector"""
        if self._blink_detector is None:
            self._blink_detector = BlinkDetector()
        return self._blink_detector
    
    @property
    def face_verifier(self) -> FaceVerifier:
        """Lazy load face verifier"""
        if self._face_verifier is None:
            self._face_verifier = FaceVerifier()
        return self._face_verifier
    
    @property
    def temporal_predictor(self) -> TemporalPredictor:
        """Lazy load temporal predictor (LSTM from AutoOEP)"""
        if self._temporal_predictor is None:
            self._temporal_predictor = get_temporal_predictor()
        return self._temporal_predictor
    
    @property
    def static_classifier(self) -> StaticClassifier:
        """Lazy load static classifier (LightGBM from AutoOEP)"""
        if self._static_classifier is None:
            self._static_classifier = get_static_classifier()
        return self._static_classifier
    
    def process_frame(self, frame: np.ndarray, timestamp: float = 0.0) -> Dict[str, Any]:
        """
        Process a single frame through the detection pipeline.
        
        Args:
            frame: BGR image from webcam
            timestamp: Frame timestamp (seconds since session start)
            
        Returns:
            Dict with current_score, active_flags, and detection details
        """
        if not self.is_active:
            return {"error": "Session is not active"}
        
        # Quality check
        quality = check_frame_quality(frame)
        if not quality["is_valid"]:
            logger.debug(f"Frame quality issues: {quality['issues']}")
            # Still update metrics but with limited data
            return {
                "processed": False,
                "quality_issues": quality["issues"],
                "current_score": self._get_current_score(),
                "active_flags": self._get_current_flags()
            }
        
        # Run detectors
        detections = self._run_detectors(frame)
        
        # Update metrics
        self.metrics.update(detections)
        
        # Calculate current score and flags (legacy)
        ratios = self.metrics.get_ratios()
        legacy_score = self.scorer.compute(ratios)
        current_flags = self.flagger.generate(ratios)
        
        # Track flag counts
        for flag in current_flags:
            self.flag_counts[flag] = self.flag_counts.get(flag, 0) + 1
        
        # AutoOEP: Add to temporal history and get predictions
        temporal_result = None
        static_result = None
        unified_score = None
        
        try:
            # Format detections for AutoOEP models
            formatted = self._format_for_autooep(detections)
            
            # Static (per-frame) classification
            if self.static_classifier.is_ready:
                static_result = self.static_classifier.predict(formatted)
            
            # Temporal analysis (sequence-based)
            self.temporal_predictor.add_frame(formatted, timestamp)
            if self.temporal_predictor.is_ready:
                temporal_result = self.temporal_predictor.predict()
            
            # Unified cheat score
            static_prob = static_result['probability'] if static_result else 0.0
            temporal_prob = temporal_result['probability'] if temporal_result and temporal_result.get('ready') else 0.0
            unified = calculate_cheat_score(static_prob, temporal_prob, current_flags)
            unified_score = unified['unified_probability']
            self.cheat_score_history.append(unified_score)
            
        except Exception as e:
            logger.warning(f"AutoOEP analysis error: {e}")
        
        # Store previous frame for motion detection
        self._prev_frame = frame.copy()
        
        return {
            "processed": True,
            "current_score": legacy_score,
            "active_flags": current_flags,
            "detections": detections,
            "frame_count": self.metrics.frame_count,
            # AutoOEP enhanced analysis
            "temporal_analysis": temporal_result,
            "static_analysis": static_result,
            "unified_cheat_score": unified_score
        }
    
    def _run_detectors(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run all detectors on a frame.
        
        Args:
            frame: BGR image
            
        Returns:
            Combined detection results
        """
        detections = {}
        
        # Face detection
        try:
            face_result = self.face_detector.detect(frame)
            detections["face_present"] = face_result["face_present"]
            detections["num_faces"] = face_result["num_faces"]
            
            # If face detected, run pose and gaze
            if face_result["face_present"] and face_result["landmarks"]:
                landmarks = face_result["landmarks"][0]  # First face
                
                # Head pose
                pose_result = self.head_pose.estimate(frame, landmarks)
                detections["deviation"] = pose_result["deviation"]
                detections["x_rotation"] = pose_result["x_rotation"]
                detections["y_rotation"] = pose_result["y_rotation"]
                detections["z_rotation"] = pose_result["z_rotation"]
                
                # Gaze tracking
                gaze_result = self.gaze_tracker.track(frame, landmarks)
                detections["gaze_direction"] = gaze_result["gaze_direction"]
                detections["iris_ratio"] = gaze_result["iris_ratio"]
                
                # Blink detection
                blink_result = self.blink_detector.detect(landmarks)
                detections["is_blinking"] = blink_result["is_blinking"]
                detections["total_blinks"] = blink_result["total_blinks"]
                detections["blink_rate"] = blink_result["blink_rate"]
            else:
                detections["deviation"] = "unknown"
                detections["gaze_direction"] = "unknown"
                detections["is_blinking"] = False
                
        except Exception as e:
            logger.warning(f"Face detection error: {e}")
            detections["face_present"] = False
            detections["num_faces"] = 0
        
        # Object detection
        try:
            obj_result = self.object_detector.detect(frame)
            detections["has_prohibited"] = obj_result["has_prohibited"]
            detections["prohibited_items"] = obj_result["prohibited_items"]
        except Exception as e:
            logger.warning(f"Object detection error: {e}")
            detections["has_prohibited"] = False
            detections["prohibited_items"] = []
        
        # Hand detection
        try:
            hand_result = self.hand_detector.detect(frame)
            detections["hands_visible"] = hand_result["hands_visible"]
            detections["num_hands"] = hand_result["num_hands"]
        except Exception as e:
            logger.warning(f"Hand detection error: {e}")
            detections["hands_visible"] = True  # Assume visible on error
        
        return detections
    
    def _get_current_score(self) -> int:
        """Get current integrity score"""
        ratios = self.metrics.get_ratios()
        return self.scorer.compute(ratios)
    
    def _get_current_flags(self) -> list:
        """Get current active flags"""
        ratios = self.metrics.get_ratios()
        return self.flagger.generate(ratios)
    
    def add_tab_switch(self):
        """Record a tab switch event"""
        self.metrics.add_tab_switch()
        logger.info(f"Tab switch recorded for session {self.id}")
    
    def _format_for_autooep(self, detections: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format detector results for AutoOEP models.
        
        Converts our detector output format to the format expected
        by the temporal predictor and static classifier.
        """
        # Build formatted dict for LSTM/LightGBM
        formatted = {
            'face_detected': detections.get('face_present', False),
            'face_count': detections.get('num_faces', 0),
            'face_verified': True,  # Default to true if not checked
            
            # Gaze features
            'gaze': {
                'iris_position': 0.5,  # Default center
                'iris_ratio': detections.get('iris_ratio', 0.5),
                'direction': self._gaze_to_numeric(detections.get('gaze_direction', 'center')),
                'zone': 0,
                'is_looking_away': detections.get('gaze_direction', 'center') != 'center'
            },
            
            # Head pose
            'head_pose': {
                'pitch': detections.get('x_rotation', 0.0),
                'yaw': detections.get('y_rotation', 0.0),
                'roll': detections.get('z_rotation', 0.0),
                'distance': 0.0,
                'is_suspicious': detections.get('deviation', 'center') not in ['center', 'unknown']
            },
            
            # Mouth
            'mouth': {
                'is_open': False,  # Would need mouth detection
                'aperture': 0.0
            },
            
            # Objects
            'objects': self._parse_prohibited_items(detections.get('prohibited_items', [])),
            
            # Hands
            'hands': {
                'detected': detections.get('hands_visible', True),
                'suspicious_position': False
            },
            
            # Audio (if available)
            'audio': {
                'suspicious': False,
                'amplitude': 0.0
            },
            
            # Blinks
            'blinks': {
                'count': detections.get('total_blinks', 0),
                'rate': detections.get('blink_rate', 0.0)
            }
        }
        
        return formatted
    
    def _gaze_to_numeric(self, direction: str) -> int:
        """Convert gaze direction string to numeric value."""
        mapping = {
            'center': 0,
            'left': 1,
            'right': 2,
            'up': 3,
            'down': 4,
            'away': 5,
            'unknown': 0
        }
        return mapping.get(direction.lower() if isinstance(direction, str) else 'center', 0)
    
    def _parse_prohibited_items(self, items: list) -> Dict[str, bool]:
        """Parse prohibited items list into object detection dict."""
        result = {
            'phone': False,
            'book_open': False,
            'book_closed': False,
            'earpiece': False,
            'paper': False,
            'watch': False,
            'headphone': False,
            'all_objects': items or []
        }
        
        if not items:
            return result
        
        item_mapping = {
            'phone': 'phone',
            'cell phone': 'phone',
            'mobile': 'phone',
            'book': 'book_open',
            'open book': 'book_open',
            'closed book': 'book_closed',
            'earpiece': 'earpiece',
            'earphone': 'earpiece',
            'paper': 'paper',
            'sheet': 'paper',
            'watch': 'watch',
            'headphone': 'headphone',
            'headphones': 'headphone'
        }
        
        for item in items:
            item_lower = item.lower() if isinstance(item, str) else str(item).lower()
            for key, mapped in item_mapping.items():
                if key in item_lower:
                    result[mapped] = True
                    break
        
        return result
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize the session and return final results.
        
        Returns:
            Final proctoring results with integrity score and flags
        """
        self.is_active = False
        
        # Calculate final metrics
        ratios = self.metrics.get_ratios()
        final_score = self.scorer.compute(ratios)
        flags = self.flagger.generate(ratios)
        review_required = self.flagger.requires_review(flags, final_score)
        
        # Log session end
        log_session_end(
            self.id,
            final_score,
            flags,
            self.metrics.frame_count
        )
        
        # Log any triggered flags
        for flag in flags:
            log_flag_triggered(
                self.id,
                flag,
                ratios.get(flag, 0),
                self.flagger.thresholds.get(flag, 0)
            )
        
        # Close detectors
        self._cleanup()
        
        result = {
            "session_id": self.id,
            "assessment_id": self.assessment_id,
            "student_id": self.student_id,
            "integrity_score": final_score,
            "flags": flags,
            "review_required": review_required,
            "frames_processed": self.metrics.frame_count,
            "duration_seconds": (datetime.utcnow() - self.started_at).total_seconds(),
            "metrics_summary": self.metrics.get_summary()
        }
        
        logger.info(f"Session {self.id} finalized: score={final_score}, flags={flags}")
        
        return result
    
    def _cleanup(self):
        """Clean up resources"""
        if self._hand_detector is not None:
            try:
                self._hand_detector.close()
            except Exception:
                pass
