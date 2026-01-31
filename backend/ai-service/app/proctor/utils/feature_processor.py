"""
Feature Extractor - Comprehensive proctoring feature extraction

Ported from AutoOEP/Proctor/feature_extractor.py
Consolidates all feature extraction for LSTM/LightGBM models.
"""

import logging
import numpy as np
from typing import Dict, Any, Set, Optional

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """
    Processes raw detection results into features for ML models.
    
    Handles:
    - Categorical encoding (iris_pos, mouth_zone, gaze_direction, gaze_zone)
    - One-hot encoding of prohibited items
    - Numeric defaults and coercion
    - Feature ordering for model input
    """
    
    # Prohibited items recognized by the models
    ALL_OBJECTS: Set[str] = {
        'cell phone', 'chits', 'closedbook', 'earpiece', 
        'headphone', 'openbook', 'sheet', 'watch'
    }
    
    # Categorical mappings
    MAPPINGS = {
        'iris_pos': {'center': 0, 'left': 1, 'right': 2},
        'mouth_zone': {'green': 0, 'yellow': 1, 'orange': 2, 'red': 3},
        'gaze_direction': {'forward': 0, 'left': 1, 'right': 2, 'up': 3, 'down': 4},
        'gaze_zone': {'white': 0, 'yellow': 1, 'red': 2}
    }
    
    # Default values for missing features
    NAN_MAPPINGS = {
        'iris_pos': -1, 
        'mouth_zone': -1, 
        'gaze_direction': -1, 
        'gaze_zone': -1,
        'H-Distance': 10000.0, 
        'F-Distance': 10000.0,
    }
    
    # Column order for model input (must match training)
    FEATURE_COLUMNS = [
        'timestamp', 'verification_result', 'num_faces', 'iris_pos', 'iris_ratio',
        'mouth_zone', 'mouth_area', 'x_rotation', 'y_rotation', 'z_rotation',
        'radial_distance', 'gaze_direction', 'gaze_zone', 'watch', 'headphone',
        'closedbook', 'earpiece', 'cell phone', 'openbook', 'chits', 'sheet',
        'H-Distance', 'F-Distance'
    ]
    
    @staticmethod
    def to_float(val: Any, default: float = 0.0) -> float:
        """Convert value to float with default fallback."""
        if val is None:
            return default
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)
        if isinstance(val, str):
            v = val.strip()
            if v == '' or v.lower() == 'nan':
                return default
            try:
                return float(v)
            except Exception:
                return default
        return default
    
    @staticmethod
    def to_int(val: Any, default: int = 0) -> int:
        """Convert value to int with default fallback."""
        if val is None:
            return default
        if isinstance(val, bool):
            return int(val)
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, (float, np.floating)):
            return int(val)
        if isinstance(val, str):
            v = val.strip()
            if v == '' or v.lower() == 'nan':
                return default
            try:
                return int(float(v))
            except Exception:
                return default
        return default
    
    @classmethod
    def encode_categorical(cls, value: Any, category: str) -> int:
        """Encode a categorical value to its numeric representation."""
        if value is None:
            return cls.NAN_MAPPINGS.get(category, -1)
        
        key = str(value).lower().strip()
        mapping = cls.MAPPINGS.get(category, {})
        return mapping.get(key, cls.NAN_MAPPINGS.get(category, -1))
    
    @classmethod
    def encode_prohibited_items(cls, items: Any) -> Dict[str, int]:
        """One-hot encode prohibited items."""
        result = {obj: 0 for obj in cls.ALL_OBJECTS}
        
        if items is None:
            return result
        
        # Handle different input types
        if isinstance(items, (list, tuple, set)):
            observed = set(str(item).lower().strip() for item in items if item)
        elif isinstance(items, str):
            # Parse string like "['cell phone', 'book']"
            import re
            s = items.strip().strip('[]')
            observed = set()
            for token in re.split(r"[,;|]", s):
                t = token.strip().strip("'\"").lower()
                if t:
                    observed.add(t)
        else:
            observed = set()
        
        # Match observed items to known objects
        for obj in cls.ALL_OBJECTS:
            if obj in observed:
                result[obj] = 1
            # Also check partial matches
            for obs in observed:
                if obj in obs or obs in obj:
                    result[obj] = 1
        
        return result
    
    @classmethod
    def process_raw_features(cls, raw_features: Dict[str, Any], timestamp: float = 0.0) -> Dict[str, Any]:
        """
        Process raw detection results into model-ready features.
        
        Args:
            raw_features: Raw detection results from ProctorSession
            timestamp: Frame timestamp
            
        Returns:
            Processed feature dictionary in correct order
        """
        processed = {}
        
        # Timestamp
        processed['timestamp'] = timestamp
        
        # Face verification
        processed['verification_result'] = cls.to_int(
            raw_features.get('verification_result', 
                            raw_features.get('face_verified', 1)), 1
        )
        
        # Face count
        processed['num_faces'] = cls.to_int(
            raw_features.get('num_faces', 
                            raw_features.get('face_count', 1)), 1
        )
        
        # Iris/gaze features
        gaze = raw_features.get('gaze', {})
        processed['iris_pos'] = cls.encode_categorical(
            gaze.get('iris_position', raw_features.get('iris_pos', 'center')), 
            'iris_pos'
        )
        processed['iris_ratio'] = cls.to_float(
            gaze.get('iris_ratio', raw_features.get('iris_ratio', 0.5)), 0.5
        )
        
        # Mouth features
        mouth = raw_features.get('mouth', {})
        processed['mouth_zone'] = cls.encode_categorical(
            mouth.get('zone', raw_features.get('mouth_zone', 'green')), 
            'mouth_zone'
        )
        processed['mouth_area'] = cls.to_float(
            mouth.get('aperture', raw_features.get('mouth_area', 0.0)), 0.0
        )
        
        # Head pose
        head = raw_features.get('head_pose', {})
        processed['x_rotation'] = cls.to_float(
            head.get('pitch', raw_features.get('x_rotation', 0.0)), 0.0
        )
        processed['y_rotation'] = cls.to_float(
            head.get('yaw', raw_features.get('y_rotation', 0.0)), 0.0
        )
        processed['z_rotation'] = cls.to_float(
            head.get('roll', raw_features.get('z_rotation', 0.0)), 0.0
        )
        processed['radial_distance'] = cls.to_float(
            head.get('distance', raw_features.get('radial_distance', 0.0)), 0.0
        )
        
        # Gaze direction and zone
        processed['gaze_direction'] = cls.encode_categorical(
            gaze.get('direction', raw_features.get('gaze_direction', 'forward')),
            'gaze_direction'
        )
        processed['gaze_zone'] = cls.encode_categorical(
            gaze.get('zone', raw_features.get('gaze_zone', 'white')),
            'gaze_zone'
        )
        
        # Prohibited items (one-hot encoded)
        objects = raw_features.get('objects', {})
        prohibited_items = raw_features.get('prohibited_items', [])
        
        # Get items from either format
        items_to_encode = prohibited_items if prohibited_items else []
        if isinstance(objects, dict) and objects.get('all_objects'):
            items_to_encode = objects['all_objects']
        
        item_encoding = cls.encode_prohibited_items(items_to_encode)
        for obj in cls.ALL_OBJECTS:
            # Also check direct object dict if available
            if isinstance(objects, dict) and objects.get(obj.replace(' ', '_'), False):
                item_encoding[obj] = 1
            processed[obj] = item_encoding[obj]
        
        # Distance features (dual-camera)
        processed['H-Distance'] = cls.to_float(
            raw_features.get('H-Distance', 
                            raw_features.get('hand_distance', 10000.0)), 10000.0
        )
        processed['F-Distance'] = cls.to_float(
            raw_features.get('F-Distance', 
                            raw_features.get('face_distance', 10000.0)), 10000.0
        )
        
        return processed
    
    @classmethod
    def to_feature_vector(cls, processed_features: Dict[str, Any]) -> np.ndarray:
        """
        Convert processed features to a numpy array in correct column order.
        
        Args:
            processed_features: Processed feature dictionary
            
        Returns:
            numpy array of features
        """
        return np.array([
            processed_features.get(col, 0.0) for col in cls.FEATURE_COLUMNS
        ], dtype=np.float32)
    
    @classmethod
    def extract_and_process(cls, raw_features: Dict[str, Any], timestamp: float = 0.0) -> np.ndarray:
        """
        Full pipeline: extract and process features to vector.
        
        Args:
            raw_features: Raw detection results
            timestamp: Frame timestamp
            
        Returns:
            Feature vector as numpy array
        """
        processed = cls.process_raw_features(raw_features, timestamp)
        return cls.to_feature_vector(processed)


# Singleton instance
_feature_processor: Optional[FeatureProcessor] = None


def get_feature_processor() -> FeatureProcessor:
    """Get or create singleton feature processor."""
    global _feature_processor
    if _feature_processor is None:
        _feature_processor = FeatureProcessor()
    return _feature_processor
