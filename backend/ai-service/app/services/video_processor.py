"""
Video Processor Service - Intelligent Page Turn Detection
Extracts key frames from videos of handwritten notes

Algorithm:
1. Analyze video to calculate adaptive thresholds
2. Detect SCENE CHANGE (page turn = sudden large motion)
3. Wait for motion to SETTLE (diff drops below baseline)
4. Capture BEST FRAME from settled period
5. Repeat until video ends

Technologies used:
- OpenCV for video processing
- Frame differencing for scene change detection
- Histogram comparison for content change
- Adaptive thresholds based on video content
- Laplacian variance for blur/quality detection
"""
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessingState(Enum):
    """State machine states for page turn detection"""
    INITIAL = "initial"              # Start of video, capture first page
    WAITING_FOR_TURN = "waiting"     # Stable page, waiting for turn
    PAGE_TURNING = "turning"         # High motion detected, page is turning
    SETTLING = "settling"            # Motion decreased, waiting for stability


@dataclass
class ExtractedFrame:
    """Represents an extracted frame from video"""
    frame_number: int
    timestamp: float  # seconds
    image: np.ndarray
    quality_score: float  # 0-1 based on blur detection
    page_number: int


class VideoProcessor:
    """
    Intelligent page turn detection for handwritten notes videos
    
    Features:
    - Adaptive threshold calculation
    - Scene change detection for page turns
    - Motion analysis with sliding window
    - Best frame selection based on sharpness
    - Automatic page counting
    """
    
    def __init__(
        self,
        scene_change_multiplier: float = 2.0,     # Multiplier over baseline for scene change
        settling_multiplier: float = 1.2,         # Multiplier for settling detection
        min_settle_frames: int = 5,               # Frames needed to confirm settling
        min_frames_between_pages: int = 20,       # Minimum frames between page captures
        blur_threshold: float = 100.0,            # Laplacian variance threshold
        max_pages: int = 100                      # Maximum pages to extract
    ):
        self.scene_change_multiplier = scene_change_multiplier
        self.settling_multiplier = settling_multiplier
        self.min_settle_frames = min_settle_frames
        self.min_frames_between_pages = min_frames_between_pages
        self.blur_threshold = blur_threshold
        self.max_pages = max_pages
    
    def _analyze_video(self, cap: cv2.VideoCapture) -> Tuple[float, float, float]:
        """
        Pre-analyze video to calculate adaptive thresholds
        
        Returns:
            (mean_diff, std_diff, scene_threshold)
        """
        diffs = []
        prev_gray = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                diffs.append(np.mean(diff))
            
            prev_gray = gray.copy()
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        if not diffs:
            return 5.0, 2.0, 10.0
        
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        
        # Scene change threshold: significantly above normal
        scene_threshold = mean_diff + (std_diff * self.scene_change_multiplier)
        
        logger.info(f"Video analysis: mean_diff={mean_diff:.2f}, std={std_diff:.2f}, scene_threshold={scene_threshold:.2f}")
        
        return mean_diff, std_diff, scene_threshold
    
    def extract_frames(
        self,
        video_path: str,
        progress_callback: Optional[callable] = None
    ) -> List[ExtractedFrame]:
        """
        Extract one frame per page using intelligent page turn detection
        
        Args:
            video_path: Path to video file
            progress_callback: Optional callback(progress: float, message: str)
        
        Returns:
            List of ExtractedFrame objects, one per detected page
        """
        logger.info(f"Processing video with page turn detection: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {fps:.1f} FPS, {total_frames} frames, {duration:.1f}s duration")
        
        # Pre-analyze video for adaptive thresholds
        if progress_callback:
            progress_callback(0.0, "Analyzing video...")
        
        mean_diff, std_diff, scene_threshold = self._analyze_video(cap)
        
        # Calculate settling threshold (motion returned to near-normal)
        settle_threshold = mean_diff * self.settling_multiplier
        
        logger.info(f"Thresholds: scene_change={scene_threshold:.2f}, settling={settle_threshold:.2f}")
        
        # State machine variables
        state = ProcessingState.INITIAL
        extracted_pages: List[ExtractedFrame] = []
        
        prev_frame = None
        prev_gray = None
        settle_buffer: List[Tuple[int, np.ndarray, float]] = []  # (frame_num, image, quality)
        last_capture_frame = -self.min_frames_between_pages
        page_number = 0
        
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Report progress
                if progress_callback and frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 0.9 + 0.1  # 10-100%
                    progress_callback(progress, f"Analyzing frame {frame_idx}/{total_frames}")
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                quality = self._calculate_quality(frame)
                
                if prev_frame is None:
                    prev_frame = frame.copy()
                    prev_gray = gray.copy()
                    state = ProcessingState.INITIAL
                    settle_buffer = [(frame_idx, frame.copy(), quality)]
                    frame_idx += 1
                    continue
                
                # Calculate frame difference
                frame_diff = self._calculate_frame_diff(prev_gray, gray)
                
                is_scene_change = frame_diff > scene_threshold
                is_settled = frame_diff <= settle_threshold
                
                # State machine logic
                if state == ProcessingState.INITIAL:
                    # Collect initial frames for first page
                    settle_buffer.append((frame_idx, frame.copy(), quality))
                    if len(settle_buffer) >= self.min_settle_frames:
                        # Capture first page
                        best_frame = self._select_best_frame(settle_buffer)
                        if best_frame is not None:
                            page_number += 1
                            extracted_pages.append(ExtractedFrame(
                                frame_number=best_frame[0],
                                timestamp=best_frame[0] / fps,
                                image=best_frame[1],
                                quality_score=best_frame[2],
                                page_number=page_number
                            ))
                            last_capture_frame = frame_idx
                            logger.info(f"Page {page_number} captured at frame {best_frame[0]}")
                        state = ProcessingState.WAITING_FOR_TURN
                        settle_buffer = []
                
                elif state == ProcessingState.WAITING_FOR_TURN:
                    # Waiting for a page turn (high motion)
                    if is_scene_change and (frame_idx - last_capture_frame) >= self.min_frames_between_pages:
                        state = ProcessingState.PAGE_TURNING
                        settle_buffer = []
                        logger.debug(f"Page turn started at frame {frame_idx}")
                
                elif state == ProcessingState.PAGE_TURNING:
                    # Page is turning, wait for motion to decrease
                    if is_settled:
                        # Motion decreased, start settling phase
                        state = ProcessingState.SETTLING
                        settle_buffer = [(frame_idx, frame.copy(), quality)]
                        logger.debug(f"Motion settling at frame {frame_idx}")
                
                elif state == ProcessingState.SETTLING:
                    if is_settled:
                        settle_buffer.append((frame_idx, frame.copy(), quality))
                        
                        if len(settle_buffer) >= self.min_settle_frames:
                            # Page has settled, capture it
                            best_frame = self._select_best_frame(settle_buffer)
                            if best_frame is not None:
                                page_number += 1
                                extracted_pages.append(ExtractedFrame(
                                    frame_number=best_frame[0],
                                    timestamp=best_frame[0] / fps,
                                    image=best_frame[1],
                                    quality_score=best_frame[2],
                                    page_number=page_number
                                ))
                                last_capture_frame = frame_idx
                                logger.info(f"Page {page_number} captured at frame {best_frame[0]}")
                            
                            state = ProcessingState.WAITING_FOR_TURN
                            settle_buffer = []
                            
                            # Check max pages
                            if page_number >= self.max_pages:
                                logger.info(f"Reached max pages limit: {self.max_pages}")
                                break
                    elif is_scene_change:
                        # Another burst of motion, go back to turning
                        state = ProcessingState.PAGE_TURNING
                        settle_buffer = []
                    else:
                        # Moderate motion, reset settle buffer but stay in settling
                        settle_buffer = []
                
                prev_frame = frame.copy()
                prev_gray = gray.copy()
                frame_idx += 1
            
            # Handle end of video - capture any remaining content
            if state == ProcessingState.SETTLING and len(settle_buffer) >= 2:
                best_frame = self._select_best_frame(settle_buffer)
                if best_frame is not None:
                    page_number += 1
                    extracted_pages.append(ExtractedFrame(
                        frame_number=best_frame[0],
                        timestamp=best_frame[0] / fps,
                        image=best_frame[1],
                        quality_score=best_frame[2],
                        page_number=page_number
                    ))
                    logger.info(f"Page {page_number} captured at end of video")
        
        finally:
            cap.release()
        
        logger.info(f"Extracted {len(extracted_pages)} pages from video")
        return extracted_pages
    
    def _calculate_frame_diff(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """Calculate mean absolute difference between frames"""
        diff = cv2.absdiff(prev_gray, curr_gray)
        return float(np.mean(diff))
    
    def _calculate_quality(self, frame: np.ndarray) -> float:
        """
        Calculate quality score based on blur detection
        Uses Laplacian variance method - higher = sharper
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range
        quality = min(1.0, laplacian_var / (self.blur_threshold * 2))
        return quality
    
    def _select_best_frame(
        self,
        settle_buffer: List[Tuple[int, np.ndarray, float]]
    ) -> Optional[Tuple[int, np.ndarray, float]]:
        """
        Select the best frame from settle buffer based on quality
        Returns (frame_number, image, quality_score) or None
        """
        if not settle_buffer:
            return None
        
        # Sort by quality and return best
        sorted_frames = sorted(settle_buffer, key=lambda x: x[2], reverse=True)
        return sorted_frames[0]
    
    def save_frames(
        self,
        frames: List[ExtractedFrame],
        output_dir: str,
        prefix: str = "page"
    ) -> List[str]:
        """
        Save extracted frames to disk
        
        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for frame in frames:
            filename = f"{prefix}_{frame.page_number:03d}_{frame.timestamp:.2f}s.png"
            filepath = output_path / filename
            cv2.imwrite(str(filepath), frame.image)
            saved_paths.append(str(filepath))
        
        logger.info(f"Saved {len(saved_paths)} pages to {output_dir}")
        return saved_paths


# Convenience function for quick processing
def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    progress_callback: Optional[callable] = None
) -> List[str]:
    """
    Quick helper to extract page frames from video and save to disk
    
    Returns:
        List of saved image file paths
    """
    processor = VideoProcessor()
    frames = processor.extract_frames(video_path, progress_callback)
    return processor.save_frames(frames, output_dir)
