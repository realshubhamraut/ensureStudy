"""
Advanced Image Enhancement Service
Complete document processing pipeline for handwritten notes

Features:
- Document corner detection using contour analysis
- Perspective correction to perfectly flat A4/Letter output
- Auto-rotation and deskewing
- Adaptive brightness/contrast normalization using CLAHE
- Intelligent noise reduction
- Shadow removal
- Sharpening for OCR optimization

Technologies:
- OpenCV for image processing
- NumPy for mathematical operations
"""
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Standard paper sizes in pixels at 300 DPI
A4_SIZE = (2480, 3508)  # 210mm x 297mm at 300 DPI
LETTER_SIZE = (2550, 3300)  # 8.5" x 11" at 300 DPI


@dataclass
class ImageMetrics:
    """Quality metrics for an image"""
    brightness: float  # 0-255
    contrast: float    # Standard deviation
    sharpness: float   # Laplacian variance
    noise_level: float # Estimated noise


@dataclass
class DocumentCorners:
    """Detected document corners"""
    top_left: Tuple[int, int]
    top_right: Tuple[int, int]
    bottom_right: Tuple[int, int]
    bottom_left: Tuple[int, int]
    confidence: float  # 0-1 detection confidence


@dataclass
class EnhancedImage:
    """Result of image enhancement"""
    original: np.ndarray
    enhanced: np.ndarray
    thumbnail: np.ndarray
    metrics_before: ImageMetrics
    metrics_after: ImageMetrics
    corners_detected: bool
    rotation_angle: float


class ImageEnhancer:
    """
    Advanced image enhancement for handwritten notes
    
    Complete Pipeline:
    1. Document Detection - Find page corners
    2. Perspective Correction - Flatten to perfect rectangle
    3. A4/Letter Sizing - Resize to standard dimensions
    4. Rotation Correction - Fix tilt using text line detection
    5. Shadow Removal - Handle uneven lighting
    6. Brightness/Contrast - Normalize using CLAHE
    7. Noise Reduction - Clean while preserving edges
    8. Sharpening - Enhance text clarity for OCR
    """
    
    def __init__(
        self,
        output_size: Tuple[int, int] = A4_SIZE,
        target_brightness: float = 200.0,
        denoise_strength: int = 8,
        sharpen_strength: float = 1.2,
        thumbnail_size: Tuple[int, int] = (200, 280),
        auto_orient: bool = True
    ):
        self.output_size = output_size
        self.target_brightness = target_brightness
        self.denoise_strength = denoise_strength
        self.sharpen_strength = sharpen_strength
        self.thumbnail_size = thumbnail_size
        self.auto_orient = auto_orient
    
    def enhance(
        self,
        image: np.ndarray,
        crop_document: bool = True,
        apply_binarization: bool = False
    ) -> EnhancedImage:
        """
        Apply full enhancement pipeline to an image
        
        Args:
            image: Input BGR image
            crop_document: Auto-detect and crop document from background
            apply_binarization: Apply adaptive thresholding for pure B/W
        
        Returns:
            EnhancedImage with all results
        """
        logger.info(f"Enhancing image: {image.shape}")
        
        # Calculate initial metrics
        metrics_before = self._calculate_metrics(image)
        
        enhanced = image.copy()
        corners_detected = False
        rotation_angle = 0.0
        
        # Step 1: Document Detection and Perspective Correction
        if crop_document:
            corners = self._detect_document_corners(enhanced)
            if corners and corners.confidence > 0.5:
                enhanced = self._perspective_transform(enhanced, corners)
                corners_detected = True
                logger.info(f"Document corners detected with {corners.confidence:.2f} confidence")
        
        # Step 2: Resize to standard A4 dimensions
        enhanced = self._resize_to_standard(enhanced)
        
        # Step 3: Auto-rotation (deskew)
        enhanced, rotation_angle = self._deskew(enhanced)
        if abs(rotation_angle) > 0.1:
            logger.info(f"Deskewed by {rotation_angle:.2f} degrees")
        
        # Step 4: Shadow removal
        enhanced = self._remove_shadows(enhanced)
        
        # Step 5: Brightness and contrast normalization
        enhanced = self._normalize_lighting(enhanced)
        
        # Step 6: Noise reduction
        enhanced = self._denoise(enhanced)
        
        # Step 7: Sharpening
        enhanced = self._sharpen(enhanced)
        
        # Step 8: Optional binarization
        if apply_binarization:
            enhanced = self._adaptive_binarize(enhanced)
        
        # Calculate final metrics
        metrics_after = self._calculate_metrics(enhanced)
        
        # Generate thumbnail
        thumbnail = cv2.resize(enhanced, self.thumbnail_size, interpolation=cv2.INTER_AREA)
        
        logger.info(f"Enhancement complete. Brightness: {metrics_before.brightness:.1f} -> {metrics_after.brightness:.1f}")
        
        return EnhancedImage(
            original=image,
            enhanced=enhanced,
            thumbnail=thumbnail,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            corners_detected=corners_detected,
            rotation_angle=rotation_angle
        )
    
    def _detect_document_corners(self, image: np.ndarray) -> Optional[DocumentCorners]:
        """
        Detect the four corners of a document in the image
        Uses multiple techniques for robust detection
        """
        h, w = image.shape[:2]
        
        # Resize for faster processing
        scale = min(1.0, 1000 / max(h, w))
        small = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Convert to grayscale
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try multiple edge detection methods
        corners = None
        best_confidence = 0
        
        # Method 1: Canny edge detection
        for thresh1 in [30, 50, 75]:
            for thresh2 in [100, 150, 200]:
                edges = cv2.Canny(blurred, thresh1, thresh2)
                edges = cv2.dilate(edges, None, iterations=1)
                
                result = self._find_document_contour(edges, gray.shape)
                if result and result[1] > best_confidence:
                    corners, best_confidence = result
        
        # Method 2: Adaptive threshold
        if best_confidence < 0.7:
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            edges = cv2.Canny(thresh, 50, 150)
            
            result = self._find_document_contour(edges, gray.shape)
            if result and result[1] > best_confidence:
                corners, best_confidence = result
        
        if corners is None:
            return None
        
        # Scale corners back to original size
        corners = corners / scale
        
        return DocumentCorners(
            top_left=tuple(corners[0].astype(int)),
            top_right=tuple(corners[1].astype(int)),
            bottom_right=tuple(corners[2].astype(int)),
            bottom_left=tuple(corners[3].astype(int)),
            confidence=best_confidence
        )
    
    def _find_document_contour(
        self, 
        edges: np.ndarray, 
        shape: Tuple[int, int]
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Find the largest 4-point contour that looks like a document"""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        h, w = shape
        image_area = h * w
        
        for contour in contours[:5]:  # Check top 5 largest
            area = cv2.contourArea(contour)
            
            # Skip if too small (less than 10% of image) or too large (more than 98%)
            if area < image_area * 0.1 or area > image_area * 0.98:
                continue
            
            # Approximate to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                # Check if it's roughly rectangular
                pts = approx.reshape(4, 2)
                ordered = self._order_points(pts)
                
                # Calculate aspect ratio
                width = np.linalg.norm(ordered[1] - ordered[0])
                height = np.linalg.norm(ordered[2] - ordered[1])
                aspect = max(width, height) / (min(width, height) + 1e-6)
                
                # A4 aspect ratio is ~1.414, allow some variation
                if 1.0 < aspect < 2.0:
                    confidence = min(1.0, area / (image_area * 0.5))
                    return (ordered, confidence)
        
        return None
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum of coordinates: smallest = top-left, largest = bottom-right
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Difference of coordinates: smallest = top-right, largest = bottom-left
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        
        return rect
    
    def _perspective_transform(
        self, 
        image: np.ndarray, 
        corners: DocumentCorners
    ) -> np.ndarray:
        """Apply perspective transform to flatten the document"""
        # Source points
        src = np.array([
            corners.top_left,
            corners.top_right,
            corners.bottom_right,
            corners.bottom_left
        ], dtype=np.float32)
        
        # Calculate output dimensions maintaining aspect ratio
        width_top = np.linalg.norm(np.array(corners.top_right) - np.array(corners.top_left))
        width_bottom = np.linalg.norm(np.array(corners.bottom_right) - np.array(corners.bottom_left))
        width = int(max(width_top, width_bottom))
        
        height_left = np.linalg.norm(np.array(corners.bottom_left) - np.array(corners.top_left))
        height_right = np.linalg.norm(np.array(corners.bottom_right) - np.array(corners.top_right))
        height = int(max(height_left, height_right))
        
        # Destination points
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Apply transform
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, (width, height))
        
        return warped
    
    def _resize_to_standard(self, image: np.ndarray) -> np.ndarray:
        """Resize image to standard A4 dimensions"""
        h, w = image.shape[:2]
        target_w, target_h = self.output_size
        
        # Determine orientation
        if self.auto_orient:
            if w > h:  # Landscape
                target_w, target_h = target_h, target_w
        
        # Resize maintaining aspect ratio then pad/crop
        aspect = w / h
        target_aspect = target_w / target_h
        
        if aspect > target_aspect:
            # Image is wider, fit by width
            new_w = target_w
            new_h = int(target_w / aspect)
        else:
            # Image is taller, fit by height
            new_h = target_h
            new_w = int(target_h * aspect)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create white canvas and center the image
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def _deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect and correct text skew angle"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=100, 
            minLineLength=100, 
            maxLineGap=10
        )
        
        if lines is None or len(lines) < 5:
            return image, 0.0
        
        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider nearly horizontal lines (within 45 degrees)
                if -45 < angle < 45:
                    angles.append(angle)
        
        if not angles:
            return image, 0.0
        
        # Use median angle
        median_angle = np.median(angles)
        
        # Only correct if angle is significant but not too extreme
        if abs(median_angle) < 0.5 or abs(median_angle) > 10:
            return image, 0.0
        
        # Rotate image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Calculate new bounding box size
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust rotation matrix
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=(255, 255, 255))
        
        # Resize back to original dimensions
        rotated = cv2.resize(rotated, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        return rotated, median_angle
    
    def _remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """Remove shadows and even out lighting"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Merge back
        lab_clahe = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """Normalize brightness and contrast"""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate current brightness
        current_brightness = np.mean(l)
        
        # Adjust brightness
        if current_brightness < self.target_brightness:
            alpha = self.target_brightness / (current_brightness + 1)
            alpha = min(alpha, 1.5)  # Limit adjustment
            l = cv2.convertScaleAbs(l, alpha=alpha, beta=0)
        
        # Enhance contrast
        l = cv2.normalize(l, None, 20, 235, cv2.NORM_MINMAX)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Reduce noise while preserving edges"""
        # Use bilateral filter for edge-preserving smoothing
        denoised = cv2.bilateralFilter(
            image, 
            d=9, 
            sigmaColor=75, 
            sigmaSpace=75
        )
        
        # Additional denoising for color images
        denoised = cv2.fastNlMeansDenoisingColored(
            denoised,
            None,
            self.denoise_strength,
            self.denoise_strength,
            7,
            21
        )
        
        return denoised
    
    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image for better text clarity"""
        # Unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
        sharpened = cv2.addWeighted(
            image, 1 + self.sharpen_strength,
            gaussian, -self.sharpen_strength,
            0
        )
        
        return sharpened
    
    def _adaptive_binarize(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for pure black and white"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21, 10
        )
        
        # Convert back to BGR
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def _calculate_metrics(self, image: np.ndarray) -> ImageMetrics:
        """Calculate image quality metrics"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        noise = float(np.median(np.abs(gray - np.median(gray))) * 1.4826)
        
        return ImageMetrics(
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            noise_level=noise
        )


def batch_enhance_images(
    image_paths: List[str],
    output_dir: str,
    progress_callback: Optional[callable] = None
) -> List[str]:
    """
    Enhance multiple images and save to output directory
    
    Returns:
        List of saved file paths
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    enhancer = ImageEnhancer()
    saved_paths = []
    
    for i, img_path in enumerate(image_paths):
        if progress_callback:
            progress = (i + 1) / len(image_paths)
            progress_callback(progress, f"Enhancing image {i+1}/{len(image_paths)}")
        
        try:
            image = cv2.imread(img_path)
            if image is None:
                logger.warning(f"Could not load image: {img_path}")
                continue
            
            result = enhancer.enhance(image)
            
            # Save enhanced image
            filename = f"enhanced_{i:03d}.png"
            save_path = output_path / filename
            cv2.imwrite(str(save_path), result.enhanced)
            saved_paths.append(str(save_path))
            
            logger.info(f"Enhanced and saved: {save_path}")
            
        except Exception as e:
            logger.error(f"Error enhancing {img_path}: {e}")
    
    return saved_paths
