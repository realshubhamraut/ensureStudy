"""
Static Classifier - LightGBM-based per-frame cheating detection

Uses AutoOEP's pre-trained LightGBM model to classify individual
frames as cheating/non-cheating based on extracted features.
"""

import os
import logging
import numpy as np
import joblib
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class StaticClassifier:
    """
    Wraps the LightGBM model for per-frame cheating classification.
    
    Provides instant classification without requiring a sequence of frames.
    Useful for immediate alerts on obvious cheating behaviors.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5
    ):
        """
        Initialize the static classifier.
        
        Args:
            model_path: Path to saved LightGBM model (.pkl)
            threshold: Probability threshold for cheating classification
        """
        self.threshold = threshold
        self.model = None
        self.scaler = None
        self.metadata = None
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
        else:
            # Try default path
            default_path = self._get_default_model_path()
            if default_path:
                self.load_model(default_path)
    
    def _get_default_model_path(self) -> Optional[str]:
        """Get default model path from models directory."""
        base_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 
                        'models', 'proctoring', 'best_models'),
            '/Users/proxim/projects/ensureStudy/models/proctoring/best_models',
        ]
        
        for base in base_paths:
            # Look for lightgbm model files
            if os.path.isdir(base):
                for filename in os.listdir(base):
                    if filename.startswith('lightgbm_cheating_model') and filename.endswith('.pkl'):
                        return os.path.join(base, filename)
        return None
    
    def load_model(self, model_path: str) -> bool:
        """
        Load pre-trained LightGBM model and associated scaler.
        
        Args:
            model_path: Path to .pkl model file
            
        Returns:
            True if loaded successfully
        """
        try:
            # Load model
            self.model = joblib.load(model_path)
            
            # Try to load scaler from same directory
            model_dir = os.path.dirname(model_path)
            model_id = os.path.basename(model_path).replace('lightgbm_cheating_model_', '').replace('.pkl', '')
            
            scaler_path = os.path.join(model_dir, f'scaler_{model_id}.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                
            metadata_path = os.path.join(model_dir, f'model_metadata_{model_id}.pkl')
            if os.path.exists(metadata_path):
                self.metadata = joblib.load(metadata_path)
            
            logger.info(f"[STATIC] ✅ Loaded LightGBM model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"[STATIC] ❌ Failed to load model: {e}")
            return False
    
    def extract_features(self, detection_results: Dict[str, Any]) -> np.ndarray:
        """
        Extract features for static classification.
        
        Args:
            detection_results: Output from ProctorSession detectors
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Face detection
        features.append(1 if detection_results.get('face_detected', True) else 0)
        features.append(detection_results.get('face_count', 1))
        
        # Face verification
        features.append(1 if detection_results.get('face_verified', True) else 0)
        
        # Gaze features
        gaze = detection_results.get('gaze', {})
        features.append(gaze.get('iris_position', 0.5))
        features.append(gaze.get('is_looking_away', 0))
        features.append(gaze.get('direction', 0))
        
        # Head pose
        head = detection_results.get('head_pose', {})
        features.append(head.get('pitch', 0.0))
        features.append(head.get('yaw', 0.0))
        features.append(head.get('roll', 0.0))
        features.append(1 if head.get('is_suspicious', False) else 0)
        
        # Mouth
        mouth = detection_results.get('mouth', {})
        features.append(1 if mouth.get('is_open', False) else 0)
        features.append(mouth.get('aperture', 0.0))
        
        # Objects detected
        objects = detection_results.get('objects', {})
        features.append(1 if objects.get('phone', False) else 0)
        features.append(1 if objects.get('book_open', False) else 0)
        features.append(1 if objects.get('book_closed', False) else 0)
        features.append(1 if objects.get('earpiece', False) else 0)
        features.append(1 if objects.get('paper', False) else 0)
        features.append(len(objects.get('all_objects', [])))  # Total object count
        
        # Hands
        hands = detection_results.get('hands', {})
        features.append(1 if hands.get('detected', False) else 0)
        features.append(1 if hands.get('suspicious_position', False) else 0)
        
        # Audio
        audio = detection_results.get('audio', {})
        features.append(1 if audio.get('suspicious', False) else 0)
        features.append(audio.get('amplitude', 0.0))
        
        # Blink
        blinks = detection_results.get('blinks', {})
        features.append(blinks.get('count', 0))
        features.append(blinks.get('rate', 0.0))
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a single frame as cheating/non-cheating.
        
        Args:
            detection_results: Detection results from ProctorSession
            
        Returns:
            Dict with 'probability', 'is_cheating', 'confidence'
        """
        if self.model is None:
            logger.warning("[STATIC] Model not loaded")
            return {
                'probability': 0.0,
                'is_cheating': False,
                'confidence': 0.0,
                'model_ready': False
            }
        
        try:
            # Extract features
            features = self.extract_features(detection_results)
            
            # Scale if scaler available
            if self.scaler:
                features = self.scaler.transform(features)
            
            # Predict probability
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                probability = proba[1] if len(proba) > 1 else proba[0]
            else:
                probability = float(self.model.predict(features)[0])
            
            is_cheating = probability >= self.threshold
            
            return {
                'probability': round(float(probability), 4),
                'is_cheating': is_cheating,
                'confidence': abs(probability - 0.5) * 2,
                'threshold': self.threshold,
                'model_ready': True
            }
            
        except Exception as e:
            logger.error(f"[STATIC] Prediction error: {e}")
            return {
                'probability': 0.0,
                'is_cheating': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None


# Singleton instance
_static_classifier: Optional[StaticClassifier] = None


def get_static_classifier() -> StaticClassifier:
    """Get or create the singleton static classifier."""
    global _static_classifier
    if _static_classifier is None:
        _static_classifier = StaticClassifier()
    return _static_classifier
