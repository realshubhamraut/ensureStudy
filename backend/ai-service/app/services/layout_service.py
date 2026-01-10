"""
Layout Service - Document Structure Analysis

Ported from ml/notebooks/digitize_layout_detection.ipynb

Detects:
- Text lines from image
- Groups lines into paragraphs/blocks
- Determines reading order
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class BBox:
    """Bounding box with utility methods"""
    x: int
    y: int
    w: int
    h: int
    
    @property
    def x2(self) -> int:
        return self.x + self.w
    
    @property
    def y2(self) -> int:
        return self.y + self.h
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)
    
    @property
    def area(self) -> int:
        return self.w * self.h
    
    def to_list(self) -> List[int]:
        return [self.x, self.y, self.w, self.h]


@dataclass
class TextLine:
    """A detected text line"""
    bbox: BBox
    text: str = ""
    confidence: float = 0.0
    line_index: int = 0


@dataclass
class TextBlock:
    """A block of text (paragraph/section)"""
    lines: List[TextLine] = field(default_factory=list)
    bbox: BBox = None
    block_type: str = "paragraph"  # paragraph, header, list
    block_index: int = 0
    
    @property
    def text(self) -> str:
        return '\n'.join(line.text for line in self.lines if line.text)
    
    def to_dict(self) -> dict:
        return {
            "block_index": self.block_index,
            "block_type": self.block_type,
            "bbox": self.bbox.to_list() if self.bbox else None,
            "lines": [
                {
                    "line_index": line.line_index,
                    "bbox": line.bbox.to_list() if line.bbox else None,
                    "text": line.text,
                    "confidence": line.confidence
                }
                for line in self.lines
            ],
            "text": self.text
        }


@dataclass
class PageLayout:
    """Complete layout of a page"""
    blocks: List[TextBlock] = field(default_factory=list)
    width: int = 0
    height: int = 0
    reading_order: List[int] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "reading_order": self.reading_order,
            "blocks": [block.to_dict() for block in self.blocks]
        }


class LineDetector:
    """Detect text lines using morphological operations"""
    
    def __init__(
        self,
        min_line_height: int = 10,
        max_line_height: int = 200,
        min_line_width: int = 50,
        horizontal_kernel_size: int = 40
    ):
        self.min_line_height = min_line_height
        self.max_line_height = max_line_height
        self.min_line_width = min_line_width
        self.horizontal_kernel_size = horizontal_kernel_size
    
    def detect(self, image: np.ndarray) -> List[BBox]:
        """Detect text lines in an image. Returns list of BBox."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Dilate horizontally to connect characters
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (self.horizontal_kernel_size, 1)
        )
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Dilate vertically slightly to merge close lines
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        dilated = cv2.dilate(dilated, kernel_v, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and convert to bboxes
        lines = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter by size
            if h < self.min_line_height or h > self.max_line_height:
                continue
            if w < self.min_line_width:
                continue
            
            lines.append(BBox(x, y, w, h))
        
        # Sort by y position (top to bottom)
        lines.sort(key=lambda b: b.y)
        
        return lines


class BlockDetector:
    """Group text lines into blocks/paragraphs"""
    
    def __init__(
        self,
        line_spacing_threshold: float = 2.0,
        indent_threshold: int = 50
    ):
        self.line_spacing_threshold = line_spacing_threshold
        self.indent_threshold = indent_threshold
    
    def group_lines_to_blocks(
        self, 
        lines: List[BBox],
        image_height: int
    ) -> List[TextBlock]:
        """Group lines into logical blocks/paragraphs"""
        if not lines:
            return []
        
        # Calculate average line height
        avg_height = np.mean([line.h for line in lines])
        max_gap = avg_height * self.line_spacing_threshold
        
        # Calculate left margin (most common x position)
        x_positions = [line.x for line in lines]
        median_x = np.median(x_positions)
        
        blocks = []
        current_lines = []
        
        for i, line in enumerate(lines):
            if not current_lines:
                current_lines.append(line)
                continue
            
            prev_line = current_lines[-1]
            gap = line.y - prev_line.y2
            
            # Check if this starts a new block
            is_new_block = False
            
            # Large gap
            if gap > max_gap:
                is_new_block = True
            
            # Significant indent (paragraph start)
            if line.x - median_x > self.indent_threshold:
                is_new_block = True
            
            if is_new_block and current_lines:
                blocks.append(self._create_block(current_lines, len(blocks)))
                current_lines = []
            
            current_lines.append(line)
        
        # Don't forget the last block
        if current_lines:
            blocks.append(self._create_block(current_lines, len(blocks)))
        
        return blocks
    
    def _create_block(self, lines: List[BBox], index: int) -> TextBlock:
        """Create a TextBlock from a list of line bboxes"""
        min_x = min(l.x for l in lines)
        min_y = min(l.y for l in lines)
        max_x = max(l.x2 for l in lines)
        max_y = max(l.y2 for l in lines)
        
        text_lines = [
            TextLine(bbox=line, line_index=i)
            for i, line in enumerate(lines)
        ]
        
        return TextBlock(
            lines=text_lines,
            bbox=BBox(min_x, min_y, max_x - min_x, max_y - min_y),
            block_type="paragraph",
            block_index=index
        )


class ReadingOrderDetector:
    """Determine natural reading order of text blocks"""
    
    def __init__(self, column_threshold: float = 0.3):
        self.column_threshold = column_threshold
    
    def detect_reading_order(
        self, 
        blocks: List[TextBlock],
        page_width: int
    ) -> List[int]:
        """Determine reading order for blocks. Returns list of block indices."""
        if not blocks:
            return []
        
        # Detect columns
        columns = self._detect_columns(blocks, page_width)
        
        # Sort: left-to-right columns, top-to-bottom within column
        ordered_indices = []
        for column in columns:
            column.sort(key=lambda idx: blocks[idx].bbox.y)
            ordered_indices.extend(column)
        
        return ordered_indices
    
    def _detect_columns(
        self, 
        blocks: List[TextBlock],
        page_width: int
    ) -> List[List[int]]:
        """Detect columns in the page layout"""
        if len(blocks) <= 1:
            return [[i] for i in range(len(blocks))]
        
        # Get x-center of each block
        x_centers = np.array([b.bbox.center[0] for b in blocks]).reshape(-1, 1)
        threshold = page_width * self.column_threshold
        
        # Simple clustering by sorting
        indices_sorted = np.argsort(x_centers.flatten())
        
        columns = []
        current_column = [indices_sorted[0]]
        
        for i in range(1, len(indices_sorted)):
            curr_idx = indices_sorted[i]
            prev_idx = current_column[-1]
            
            if abs(x_centers[curr_idx] - x_centers[prev_idx]) < threshold:
                current_column.append(curr_idx)
            else:
                columns.append(current_column)
                current_column = [curr_idx]
        
        columns.append(current_column)
        
        # Sort columns left to right
        columns.sort(key=lambda col: min(x_centers[idx][0] for idx in col))
        
        return columns


class LayoutAnalyzer:
    """Complete document layout analysis"""
    
    def __init__(self):
        self.line_detector = LineDetector()
        self.block_detector = BlockDetector()
        self.reading_order_detector = ReadingOrderDetector()
    
    def analyze(self, image: np.ndarray) -> PageLayout:
        """
        Perform complete layout analysis on an image.
        
        Returns PageLayout with all detected elements.
        """
        h, w = image.shape[:2]
        
        # Step 1: Detect lines
        line_bboxes = self.line_detector.detect(image)
        logger.debug(f"Detected {len(line_bboxes)} lines")
        
        # Step 2: Group into blocks
        blocks = self.block_detector.group_lines_to_blocks(line_bboxes, h)
        logger.debug(f"Grouped into {len(blocks)} blocks")
        
        # Step 3: Determine reading order
        reading_order = self.reading_order_detector.detect_reading_order(blocks, w)
        
        return PageLayout(
            blocks=blocks,
            width=w,
            height=h,
            reading_order=reading_order
        )
    
    def attach_ocr_text(
        self, 
        layout: PageLayout, 
        ocr_lines: List[dict]
    ) -> PageLayout:
        """
        Attach OCR text to layout blocks by matching bounding boxes.
        
        Args:
            layout: Analyzed page layout
            ocr_lines: List of {"bbox": [x,y,w,h], "text": "...", "confidence": 0.9}
        """
        if not ocr_lines:
            return layout
        
        # For each block and line, find matching OCR text
        for block in layout.blocks:
            for line in block.lines:
                best_match = None
                best_overlap = 0
                
                for ocr_line in ocr_lines:
                    ocr_bbox = ocr_line.get("bbox", [0, 0, 0, 0])
                    if len(ocr_bbox) < 4:
                        continue
                    
                    # Calculate overlap
                    overlap = self._bbox_overlap(
                        line.bbox, 
                        BBox(*ocr_bbox)
                    )
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = ocr_line
                
                if best_match and best_overlap > 0.3:
                    line.text = best_match.get("text", "")
                    line.confidence = best_match.get("confidence", 0.0)
        
        return layout
    
    def _bbox_overlap(self, bbox1: BBox, bbox2: BBox) -> float:
        """Calculate intersection over union of two bboxes"""
        x1 = max(bbox1.x, bbox2.x)
        y1 = max(bbox1.y, bbox2.y)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = bbox1.area + bbox2.area - intersection
        return intersection / union if union > 0 else 0.0


# Convenience function
def analyze_page_layout(image: np.ndarray) -> PageLayout:
    """Analyze page layout and return structured result."""
    analyzer = LayoutAnalyzer()
    return analyzer.analyze(image)
