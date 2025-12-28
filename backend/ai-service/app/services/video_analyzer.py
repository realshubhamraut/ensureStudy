"""
Video Analyzer for Soft Skills Evaluation

Uses computer vision to analyze:
- Eye contact (gaze direction)
- Head pose (stability and orientation)
- Posture/body position (if visible)
- Hand gestures

Integrates detection logic from proctoring module.
"""

import cv2
import numpy as np
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FrameAnalysisResult:
    """Result from analyzing a single frame"""
    face_present: bool = False
    gaze_direction: str = "unknown"  # center, left, right
    head_deviation: str = "normal"  # normal, up, down, left, right
    hands_visible: bool = False
    num_hands: int = 0


@dataclass
class VideoAnalysisResult:
    """Aggregated results from video analysis"""
    # Eye contact
    eye_contact_score: float = 75.0
    gaze_center_ratio: float = 0.75
    gaze_diversion_ratio: float = 0.0
    
    # Head pose
    head_stability_score: float = 75.0
    head_forward_ratio: float = 0.75
    
    # Hand gestures
    hand_gesture_score: float = 70.0
    hands_visible_ratio: float = 0.5
    
    # Posture (derived from head stability)
    posture_score: float = 75.0
    
    # Metadata
    frames_analyzed: int = 0
    face_present_ratio: float = 0.0


class VideoAnalyzer:
    """
    Analyzes video for soft skills evaluation metrics.
    
    Uses the same detection logic as the proctoring module
    but computes different metrics focused on communication skills.
    """
    
    def __init__(self):
        """Initialize the video analyzer with lazy-loaded detectors"""
        self._face_detector = None
        self._head_pose = None
        self._gaze_tracker = None
        self._hand_detector = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialize detectors"""
        if self._initialized:
            return
        
        try:
            from ..proctor.detectors import (
                FaceDetector,
                HeadPoseEstimator,
                GazeTracker,
                HandDetector
            )
            
            self._face_detector = FaceDetector()
            self._head_pose = HeadPoseEstimator()
            self._gaze_tracker = GazeTracker()
            self._hand_detector = HandDetector()
            self._initialized = True
            logger.info("VideoAnalyzer initialized with CV detectors")
            
        except ImportError as e:
            logger.warning(f"Could not import proctoring detectors: {e}")
            self._initialized = True  # Don't retry
        except Exception as e:
            logger.error(f"Error initializing detectors: {e}")
            self._initialized = True
    
    def analyze_frame(self, frame: np.ndarray) -> FrameAnalysisResult:
        """
        Analyze a single video frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            FrameAnalysisResult with detection results
        """
        self._ensure_initialized()
        
        result = FrameAnalysisResult()
        
        if self._face_detector is None:
            return result
        
        try:
            # Face detection
            face_result = self._face_detector.detect(frame)
            result.face_present = face_result.get("face_present", False)
            
            if result.face_present and face_result.get("landmarks"):
                landmarks = face_result["landmarks"][0]
                
                # Head pose
                if self._head_pose:
                    pose_result = self._head_pose.estimate(frame, landmarks)
                    result.head_deviation = pose_result.get("deviation", "normal")
                
                # Gaze tracking
                if self._gaze_tracker:
                    gaze_result = self._gaze_tracker.track(frame, landmarks)
                    result.gaze_direction = gaze_result.get("gaze_direction", "unknown")
            
            # Hand detection
            if self._hand_detector:
                hand_result = self._hand_detector.detect(frame)
                result.hands_visible = hand_result.get("hands_visible", False)
                result.num_hands = hand_result.get("num_hands", 0)
                
        except Exception as e:
            logger.warning(f"Frame analysis error: {e}")
        
        return result
    
    def analyze_video(
        self, 
        video_path: str, 
        sample_rate: int = 5
    ) -> VideoAnalysisResult:
        """
        Analyze a video file for soft skills metrics.
        
        Args:
            video_path: Path to video file
            sample_rate: Analyze every Nth frame (for performance)
            
        Returns:
            VideoAnalysisResult with aggregated scores
        """
        self._ensure_initialized()
        
        result = VideoAnalysisResult()
        
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return result
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return result
        
        # Counters
        frame_count = 0
        analyzed_count = 0
        face_present_count = 0
        gaze_center_count = 0
        head_forward_count = 0
        hands_visible_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Sample frames
                if frame_count % sample_rate != 0:
                    continue
                
                # Analyze frame
                frame_result = self.analyze_frame(frame)
                analyzed_count += 1
                
                if frame_result.face_present:
                    face_present_count += 1
                
                if frame_result.gaze_direction == "center":
                    gaze_center_count += 1
                
                if frame_result.head_deviation == "normal":
                    head_forward_count += 1
                
                if frame_result.hands_visible:
                    hands_visible_count += 1
                    
        finally:
            cap.release()
        
        if analyzed_count == 0:
            return result
        
        # Calculate ratios
        result.frames_analyzed = analyzed_count
        result.face_present_ratio = face_present_count / analyzed_count
        result.gaze_center_ratio = gaze_center_count / max(1, face_present_count)
        result.gaze_diversion_ratio = 1.0 - result.gaze_center_ratio
        result.head_forward_ratio = head_forward_count / max(1, face_present_count)
        result.hands_visible_ratio = hands_visible_count / analyzed_count
        
        # Calculate scores (0-100)
        result.eye_contact_score = self._calculate_eye_contact_score(
            result.gaze_center_ratio,
            result.face_present_ratio
        )
        
        result.head_stability_score = self._calculate_head_stability_score(
            result.head_forward_ratio
        )
        
        result.hand_gesture_score = self._calculate_hand_gesture_score(
            result.hands_visible_ratio
        )
        
        result.posture_score = self._calculate_posture_score(
            result.head_forward_ratio,
            result.face_present_ratio
        )
        
        logger.info(
            f"Video analyzed: {analyzed_count} frames, "
            f"eye_contact={result.eye_contact_score:.1f}, "
            f"head_stability={result.head_stability_score:.1f}"
        )
        
        return result
    
    def analyze_video_bytes(
        self, 
        video_bytes: bytes,
        sample_rate: int = 5
    ) -> VideoAnalysisResult:
        """
        Analyze video from bytes (e.g., uploaded file).
        
        Args:
            video_bytes: Raw video file bytes
            sample_rate: Analyze every Nth frame
            
        Returns:
            VideoAnalysisResult with aggregated scores
        """
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            temp_path = f.name
        
        try:
            return self.analyze_video(temp_path, sample_rate)
        finally:
            # Clean up
            try:
                Path(temp_path).unlink()
            except Exception:
                pass
    
    def _calculate_eye_contact_score(
        self, 
        gaze_center_ratio: float,
        face_present_ratio: float
    ) -> float:
        """
        Calculate eye contact score.
        
        Good eye contact = looking at camera (center gaze) most of the time.
        """
        # If face not present much, lower score
        if face_present_ratio < 0.5:
            base_score = 50.0
        else:
            base_score = 60.0
        
        # Add points for center gaze
        gaze_contribution = gaze_center_ratio * 40.0
        
        score = base_score + gaze_contribution
        return min(100.0, max(0.0, score))
    
    def _calculate_head_stability_score(self, head_forward_ratio: float) -> float:
        """
        Calculate head stability/posture score.
        
        Stable = head mostly forward-facing.
        """
        # Base score
        base_score = 50.0
        
        # Forward ratio contributes up to 50 points
        forward_contribution = head_forward_ratio * 50.0
        
        score = base_score + forward_contribution
        return min(100.0, max(0.0, score))
    
    def _calculate_hand_gesture_score(self, hands_visible_ratio: float) -> float:
        """
        Calculate hand gesture score.
        
        Some hand movement is good (active speaker), 
        but too much or too little may indicate nervousness.
        """
        # Optimal range: 30-70% visible
        if 0.3 <= hands_visible_ratio <= 0.7:
            score = 80.0 + (hands_visible_ratio - 0.3) * 50.0
        elif hands_visible_ratio < 0.3:
            # Too hidden - might be stiff
            score = 50.0 + hands_visible_ratio * 100.0
        else:
            # Very visible - could be fine or excessive
            score = 70.0 + (1.0 - hands_visible_ratio) * 50.0
        
        return min(100.0, max(0.0, score))
    
    def _calculate_posture_score(
        self, 
        head_forward_ratio: float,
        face_present_ratio: float
    ) -> float:
        """
        Calculate posture score.
        
        Derived from head position and face visibility.
        """
        # If face consistently visible and head stable = good posture
        stability = head_forward_ratio * 0.6 + face_present_ratio * 0.4
        
        score = 50.0 + stability * 50.0
        return min(100.0, max(0.0, score))
    
    def cleanup(self):
        """Release resources"""
        if self._hand_detector:
            try:
                self._hand_detector.close()
            except Exception:
                pass


# Global analyzer instance (lazy loaded)
_analyzer: Optional[VideoAnalyzer] = None


def get_video_analyzer() -> VideoAnalyzer:
    """Get or create the global VideoAnalyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = VideoAnalyzer()
    return _analyzer
