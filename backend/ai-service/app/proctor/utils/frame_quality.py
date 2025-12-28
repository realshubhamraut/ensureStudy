"""
Frame Quality Checker - Validates frame quality before processing
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def check_frame_quality(
    frame: np.ndarray,
    min_brightness: float = 40,
    max_brightness: float = 220,
    min_blur_score: float = 50,
    min_size: Tuple[int, int] = (100, 100)
) -> Dict[str, Any]:
    """
    Check frame quality for proctoring.
    
    Args:
        frame: BGR image from OpenCV
        min_brightness: Minimum average brightness (0-255)
        max_brightness: Maximum average brightness (0-255)
        min_blur_score: Minimum Laplacian variance for blur detection
        min_size: Minimum (width, height) dimensions
        
    Returns:
        Dict with:
            - is_valid: bool
            - issues: List of quality issues
            - brightness: float (0-255)
            - blur_score: float
            - dimensions: Tuple[int, int]
    """
    issues = []
    
    if frame is None or frame.size == 0:
        return {
            "is_valid": False,
            "issues": ["empty_frame"],
            "brightness": 0,
            "blur_score": 0,
            "dimensions": (0, 0)
        }
    
    height, width = frame.shape[:2]
    dimensions = (width, height)
    
    # Check dimensions
    if width < min_size[0] or height < min_size[1]:
        issues.append("too_small")
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Check brightness
    brightness = np.mean(gray)
    if brightness < min_brightness:
        issues.append("too_dark")
    elif brightness > max_brightness:
        issues.append("too_bright")
    
    # Check blur using Laplacian variance
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < min_blur_score:
        issues.append("too_blurry")
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "brightness": float(brightness),
        "blur_score": float(blur_score),
        "dimensions": dimensions
    }


def is_motion_detected(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    threshold: float = 30.0,
    min_area_ratio: float = 0.1
) -> Tuple[bool, float]:
    """
    Detect if significant motion occurred between frames.
    
    Args:
        prev_frame: Previous BGR frame
        curr_frame: Current BGR frame
        threshold: Pixel difference threshold
        min_area_ratio: Minimum ratio of changed pixels
        
    Returns:
        Tuple of (motion_detected, change_ratio)
    """
    if prev_frame is None or curr_frame is None:
        return (False, 0.0)
    
    # Ensure same size
    if prev_frame.shape != curr_frame.shape:
        return (False, 0.0)
    
    try:
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        prev_blur = cv2.GaussianBlur(prev_gray, (21, 21), 0)
        curr_blur = cv2.GaussianBlur(curr_gray, (21, 21), 0)
        
        # Compute absolute difference
        diff = cv2.absdiff(prev_blur, curr_blur)
        
        # Threshold to get binary mask
        _, thresh = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)
        
        # Calculate ratio of changed pixels
        total_pixels = thresh.shape[0] * thresh.shape[1]
        changed_pixels = cv2.countNonZero(thresh)
        change_ratio = changed_pixels / total_pixels
        
        motion_detected = change_ratio >= min_area_ratio
        
        return (motion_detected, change_ratio)
        
    except Exception as e:
        logger.warning(f"Motion detection error: {e}")
        return (False, 0.0)
