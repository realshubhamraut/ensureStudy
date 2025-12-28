
"""
Image Preprocessing Utilities for Answer Sheet Evaluation

Functions:
- detect_blur: Check image sharpness
- detect_skew_angle: Find document rotation
- correct_skew: Rotate to correct angle
- normalize_lighting: CLAHE enhancement
- extract_best_frame: Get sharpest video frame
- ImagePreprocessor: Complete pipeline class
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional


def detect_blur(image: np.ndarray, threshold: float = 100.0) -> Tuple[float, bool, str]:
    """Detect if image is blurry using Laplacian variance."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    is_blurry = variance < threshold

    if variance < 50:
        quality = "very_blurry"
    elif variance < 100:
        quality = "slightly_blurry"
    elif variance < 300:
        quality = "good"
    else:
        quality = "very_sharp"

    return variance, is_blurry, quality


def detect_skew_angle(image: np.ndarray) -> float:
    """Detect skew angle using Hough lines."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 < angle < 45:
            angles.append(angle)

    return np.median(angles) if angles else 0.0


def correct_skew(image: np.ndarray, angle: float = None) -> np.ndarray:
    """Rotate image to correct skew."""
    if angle is None:
        angle = detect_skew_angle(image)

    if abs(angle) < 0.5:
        return image

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def normalize_lighting(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Normalize lighting using CLAHE."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
    else:
        l = image.copy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_normalized = clahe.apply(l)

    if len(image.shape) == 3:
        lab_normalized = cv2.merge([l_normalized, a, b])
        return cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
    return l_normalized


def extract_best_frame(video_path: str, sample_every: int = 5) -> Tuple[np.ndarray, int, float]:
    """Extract sharpest frame from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    best_frame, best_score, best_idx = None, -1, -1
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            score, _, _ = detect_blur(frame)
            if score > best_score:
                best_score, best_frame, best_idx = score, frame.copy(), frame_idx
        frame_idx += 1

    cap.release()
    if best_frame is None:
        raise ValueError("No valid frames found")
    return best_frame, best_idx, best_score


class ImagePreprocessor:
    """Complete preprocessing pipeline."""

    def __init__(self, blur_threshold=100.0, clahe_clip=2.0, auto_deskew=True):
        self.blur_threshold = blur_threshold
        self.clahe_clip = clahe_clip
        self.auto_deskew = auto_deskew

    def process(self, image: np.ndarray) -> dict:
        """Run preprocessing pipeline."""
        quality_score, is_blurry, quality_label = detect_blur(image, self.blur_threshold)

        current = normalize_lighting(image, self.clahe_clip)

        skew_angle = 0.0
        if self.auto_deskew:
            skew_angle = detect_skew_angle(current)
            if abs(skew_angle) > 0.5:
                current = correct_skew(current, skew_angle)

        return {
            "image": current,
            "quality_score": quality_score,
            "quality_label": quality_label,
            "is_acceptable": not is_blurry,
            "skew_angle": skew_angle
        }
