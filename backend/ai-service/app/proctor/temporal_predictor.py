"""
Temporal Predictor - LSTM-based behavior sequence analysis

Uses AutoOEP's pre-trained LSTM model to detect cheating patterns
over time by analyzing sequences of frame features.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Optional, Dict, Any, List
import joblib

logger = logging.getLogger(__name__)

# Feature columns expected by the temporal model (in exact order)
TEMPORAL_FEATURE_COLS = [
    'timestamp',
    'verification_result',
    'num_faces',
    'iris_pos',
    'iris_ratio',
    'mouth_zone',
    'mouth_area',
    'x_rotation',
    'y_rotation',
    'z_rotation',
    'radial_distance',
    'gaze_direction',
    'gaze_zone',
    'watch',
    'headphone',
    'closedbook',
    'earpiece',
    'cell phone',
    'openbook',
    'chits',
    'sheet',
    'H-Distance',
    'F-Distance',
]


class LSTMModel(nn.Module):
    """LSTM-based sequence classifier (from AutoOEP)."""
    
    def __init__(self, input_size, hidden_size=128, fc_hidden=32, 
                 output_size=1, dropout=0.35, pooling="last"):
        super(LSTMModel, self).__init__()
        
        self.pooling = pooling
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        
        if pooling == "attention":
            self.attn = nn.Linear(hidden_size, 1)
        
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, fc_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden, output_size)
        
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        
        if self.pooling == "last":
            x = h_n[-1]
        elif self.pooling == "mean":
            x = out.mean(dim=1)
        elif self.pooling == "attention":
            attn_weights = torch.softmax(self.attn(out), dim=1)
            x = torch.sum(out * attn_weights, dim=1)
        
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x.squeeze(1)


class TemporalPredictor:
    """
    Wraps the LSTM temporal model for real-time cheat detection.
    
    Maintains a sliding window of frame features and predicts
    cheating probability based on behavior patterns over time.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        window_size: int = 15,
        threshold: float = 0.4,
        device: Optional[str] = None
    ):
        """
        Initialize the temporal predictor.
        
        Args:
            model_path: Path to saved model checkpoint (.pt)
            window_size: Number of frames in sliding window
            threshold: Probability threshold for cheating classification
            device: 'cuda', 'cpu', or None for auto-detect
        """
        self.window_size = window_size
        self.threshold = threshold
        self.input_size = len(TEMPORAL_FEATURE_COLS)
        
        # Device selection
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model and scaler
        self.model: Optional[nn.Module] = None
        self.scaler = None
        
        # Feature history buffer
        self.feature_history: deque = deque(maxlen=window_size)
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
        else:
            # Try default path
            default_path = self._get_default_model_path()
            if default_path and os.path.exists(default_path):
                self.load_model(default_path)
    
    def _get_default_model_path(self) -> Optional[str]:
        """Get default model path from models directory."""
        base_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 
                        'models', 'proctoring', 'best_models'),
            '/Users/proxim/projects/ensureStudy/models/proctoring/best_models',
        ]
        
        for base in base_paths:
            model_file = os.path.join(base, 'temporal_proctor_trained_on_processed.pt')
            if os.path.exists(model_file):
                return model_file
        return None
    
    def load_model(self, checkpoint_path: str) -> bool:
        """
        Load pre-trained LSTM model and scaler from checkpoint.
        
        Args:
            checkpoint_path: Path to .pt checkpoint file
            
        Returns:
            True if loaded successfully
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Extract model config
            input_size = checkpoint.get('input_size', self.input_size)
            hidden_size = checkpoint.get('hidden_size', 128)
            window_size = checkpoint.get('window_size', self.window_size)
            
            self.window_size = window_size
            self.feature_history = deque(maxlen=window_size)
            
            # Create and load model
            self.model = LSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                fc_hidden=32,
                output_size=1,
                dropout=0.35,
                pooling="last"
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load scaler
            if 'scaler' in checkpoint:
                self.scaler = checkpoint['scaler']
            
            logger.info(f"[TEMPORAL] ✅ Loaded model from {checkpoint_path}")
            logger.info(f"[TEMPORAL] Window size: {self.window_size}, Device: {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"[TEMPORAL] ❌ Failed to load model: {e}")
            return False
    
    def extract_features(self, detection_results: Dict[str, Any], timestamp: float = 0.0) -> np.ndarray:
        """
        Extract temporal features from detection results.
        
        Args:
            detection_results: Output from ProctorSession detectors
            timestamp: Current frame timestamp
            
        Returns:
            Feature vector as numpy array
        """
        features = {}
        
        # Timestamp
        features['timestamp'] = timestamp
        
        # Face verification
        features['verification_result'] = 1.0 if detection_results.get('face_verified', True) else 0.0
        
        # Face count
        features['num_faces'] = detection_results.get('face_count', 1)
        
        # Gaze features
        gaze = detection_results.get('gaze', {})
        features['iris_pos'] = gaze.get('iris_position', 0.5)
        features['iris_ratio'] = gaze.get('iris_ratio', 0.5)
        features['gaze_direction'] = gaze.get('direction', 0)
        features['gaze_zone'] = gaze.get('zone', 0)
        
        # Mouth features
        mouth = detection_results.get('mouth', {})
        features['mouth_zone'] = 1 if mouth.get('is_open', False) else 0
        features['mouth_area'] = mouth.get('aperture', 0.0)
        
        # Head pose
        head = detection_results.get('head_pose', {})
        features['x_rotation'] = head.get('pitch', 0.0)
        features['y_rotation'] = head.get('yaw', 0.0)
        features['z_rotation'] = head.get('roll', 0.0)
        features['radial_distance'] = head.get('distance', 0.0)
        
        # Object detection
        objects = detection_results.get('objects', {})
        features['watch'] = 1 if objects.get('watch', False) else 0
        features['headphone'] = 1 if objects.get('headphone', False) else 0
        features['closedbook'] = 1 if objects.get('book_closed', False) else 0
        features['earpiece'] = 1 if objects.get('earpiece', False) else 0
        features['cell phone'] = 1 if objects.get('phone', False) else 0
        features['openbook'] = 1 if objects.get('book_open', False) else 0
        features['chits'] = 1 if objects.get('cheat_sheet', False) else 0
        features['sheet'] = 1 if objects.get('paper', False) else 0
        
        # Hand distances (from dual-camera)
        features['H-Distance'] = detection_results.get('hand_distance', 0.0)
        features['F-Distance'] = detection_results.get('face_distance', 0.0)
        
        # Convert to numpy array in correct order
        feature_vector = np.array([features.get(col, 0.0) for col in TEMPORAL_FEATURE_COLS])
        return feature_vector
    
    def add_frame(self, detection_results: Dict[str, Any], timestamp: float = 0.0) -> None:
        """
        Add a frame's features to the history buffer.
        
        Args:
            detection_results: Detection results from ProctorSession
            timestamp: Frame timestamp
        """
        features = self.extract_features(detection_results, timestamp)
        self.feature_history.append(features)
    
    def predict(self) -> Optional[Dict[str, Any]]:
        """
        Run temporal prediction on current feature history.
        
        Returns:
            Dict with 'probability', 'is_cheating', 'confidence' or None if not ready
        """
        if self.model is None:
            logger.warning("[TEMPORAL] Model not loaded")
            return None
        
        if len(self.feature_history) < self.window_size:
            return {
                'ready': False,
                'frames_needed': self.window_size - len(self.feature_history),
                'probability': 0.0,
                'is_cheating': False
            }
        
        try:
            # Create sequence from history
            sequence = np.array(list(self.feature_history))
            
            # Scale features
            if self.scaler:
                sequence = self.scaler.transform(sequence)
            
            # Convert to tensor
            X = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(X)
                probability = torch.sigmoid(logits).item()
            
            is_cheating = probability >= self.threshold
            
            return {
                'ready': True,
                'probability': round(probability, 4),
                'is_cheating': is_cheating,
                'confidence': abs(probability - 0.5) * 2,  # 0-1 scale
                'window_size': self.window_size,
                'threshold': self.threshold
            }
            
        except Exception as e:
            logger.error(f"[TEMPORAL] Prediction error: {e}")
            return None
    
    def reset(self) -> None:
        """Clear the feature history buffer."""
        self.feature_history.clear()
    
    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None
    
    @property
    def frames_collected(self) -> int:
        """Number of frames currently in buffer."""
        return len(self.feature_history)


# Singleton instance
_temporal_predictor: Optional[TemporalPredictor] = None


def get_temporal_predictor() -> TemporalPredictor:
    """Get or create the singleton temporal predictor."""
    global _temporal_predictor
    if _temporal_predictor is None:
        _temporal_predictor = TemporalPredictor()
    return _temporal_predictor
