"""
Prohibited Object Detector - Detects prohibited items using YOLO

Refactored from: AutoOEP/Proctor/feature_extractor.py + VisionUtils/yolotest.py
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Set, Optional

logger = logging.getLogger(__name__)


class ProhibitedObjectDetector:
    """
    Detects prohibited objects in frames using YOLO.
    
    Detected objects:
    - cell phone
    - chits (notes)
    - closedbook / openbook
    - earpiece
    - headphone
    - watch
    - sheet (paper)
    """
    
    # Prohibited items list (from AutoOEP)
    PROHIBITED_ITEMS: Set[str] = {
        'cell phone',
        'chits',
        'closedbook',
        'earpiece',
        'headphone',
        'openbook',
        'sheet',
        'watch'
    }
    
    def __init__(self, model_path: Optional[str] = None, confidence: float = 0.5):
        """
        Initialize object detector.
        
        Args:
            model_path: Path to YOLO model weights. If None, uses default from model_loader.
            confidence: Minimum confidence threshold for detections.
        """
        self.confidence = confidence
        self.model = None
        self._model_path = model_path
        self._model_loaded = False
    
    def _ensure_model(self):
        """Lazy load YOLO model"""
        if self.model is not None:
            return
        
        if self._model_loaded:
            return
        
        try:
            if self._model_path:
                from ultralytics import YOLO
                self.model = YOLO(self._model_path)
            else:
                from ..models import get_yolo_model
                self.model = get_yolo_model()
            self._model_loaded = True
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self._model_loaded = True  # Don't retry
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect prohibited objects in a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            dict with:
                - prohibited_items: List of detected item names
                - has_prohibited: bool - True if any prohibited items found
                - detections: List of dicts with 'name', 'confidence', 'bbox'
                - item_scores: Dict mapping item name to 1/0
        """
        if frame is None or frame.size == 0:
            return self._default_result()
        
        self._ensure_model()
        
        if self.model is None:
            return self._default_result()
        
        try:
            # Run YOLO inference
            results = self.model.predict(
                frame,
                conf=self.confidence,
                verbose=False
            )
            
            detected_items: Set[str] = set()
            detections: List[Dict[str, Any]] = []
            
            for result in results:
                if result.boxes is None:
                    continue
                    
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = self.model.names.get(cls_id, f"class_{cls_id}")
                    
                    # Check if it's a prohibited item
                    name_lower = name.lower()
                    if name_lower in self.PROHIBITED_ITEMS:
                        detected_items.add(name_lower)
                        detections.append({
                            "name": name_lower,
                            "confidence": conf,
                            "bbox": box.xyxy[0].tolist()
                        })
            
            # Create item scores (1 if detected, 0 if not)
            item_scores = {item: 1 if item in detected_items else 0 
                          for item in self.PROHIBITED_ITEMS}
            
            return {
                "prohibited_items": list(detected_items),
                "has_prohibited": len(detected_items) > 0,
                "detections": detections,
                "item_scores": item_scores
            }
            
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return self._default_result()
    
    def _default_result(self) -> Dict[str, Any]:
        """Return default result when detection fails or no model"""
        return {
            "prohibited_items": [],
            "has_prohibited": False,
            "detections": [],
            "item_scores": {item: 0 for item in self.PROHIBITED_ITEMS}
        }
    
    def draw_detections(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        Draw detection boxes on frame (for debugging).
        
        Args:
            frame: BGR image
            result: Result from detect()
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in result.get("detections", []):
            bbox = det.get("bbox", [])
            if len(bbox) < 4:
                continue
            
            x1, y1, x2, y2 = [int(v) for v in bbox]
            name = det.get("name", "unknown")
            conf = det.get("confidence", 0)
            
            # Draw red box for prohibited items
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Label
            label = f"{name} {conf:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
        
        # Warning text if prohibited items found
        if result.get("has_prohibited"):
            items = ", ".join(result.get("prohibited_items", []))
            cv2.putText(
                annotated,
                f"PROHIBITED: {items}",
                (10, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        return annotated
