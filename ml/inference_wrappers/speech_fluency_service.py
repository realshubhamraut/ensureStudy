"""
Speech Fluency Inference Service

Provides real-time speech fluency analysis for mock interviews.
Uses the trained wav2vec2 + GRU model.
"""
import os
import json
import torch
import torch.nn as nn
import torchaudio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FluencyResult:
    """Speech fluency analysis result"""
    fluency_score: float  # 0-10 scale
    hesitation_count: int
    confidence: float  # Model confidence
    duration_seconds: float
    words_per_minute: Optional[float] = None


class SpeechFluencyGRU(nn.Module):
    """GRU-based fluency analyzer (matches training architecture)"""
    
    def __init__(self, wav2vec_hidden=768, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=wav2vec_hidden,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        gru_output_size = hidden_size * 2
        
        self.fluency_head = nn.Sequential(
            nn.Linear(gru_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.hesitation_head = nn.Sequential(
            nn.Linear(gru_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.ReLU()
        )
    
    def forward(self, features):
        gru_out, _ = self.gru(features)
        pooled = gru_out.mean(dim=1)
        
        fluency = self.fluency_head(pooled).squeeze(-1)
        hesitation = self.hesitation_head(pooled).squeeze(-1)
        
        return {'fluency_score': fluency, 'hesitation_count': hesitation}


class SpeechFluencyService:
    """
    Speech Fluency Analysis Service
    
    Analyzes audio for:
    - Fluency score (0-10)
    - Hesitation/filler word count
    - Speaking confidence
    
    Usage:
        service = SpeechFluencyService()
        result = service.analyze('/path/to/audio.wav')
        print(f"Fluency: {result.fluency_score}/10")
    """
    
    def __init__(self, model_dir: str = None, device: str = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_dir = Path(model_dir or os.getenv('SPEECH_MODEL_DIR', 'ml/models/speech_model'))
        
        self.sample_rate = 16000
        self.max_length_seconds = 30
        
        self.wav2vec2 = None
        self.model = None
        self._load_model()
        
        logger.info(f"[SpeechFluency] Initialized on {self.device}")
    
    def _load_model(self):
        """Load wav2vec2 and trained GRU model"""
        try:
            from transformers import Wav2Vec2Model
            
            # Load wav2vec2
            self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            self.wav2vec2.eval()
            self.wav2vec2.to(self.device)
            
            # Load trained GRU model
            config_path = self.model_dir / 'config.json'
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
            else:
                config = {'hidden_size': 256, 'num_gru_layers': 2}
            
            self.model = SpeechFluencyGRU(
                hidden_size=config.get('hidden_size', 256),
                num_layers=config.get('num_gru_layers', 2)
            )
            
            # Load weights if available
            weights_path = self.model_dir / 'speech_fluency_model.pth'
            if weights_path.exists():
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                logger.info(f"[SpeechFluency] Loaded weights from {weights_path}")
            else:
                logger.warning(f"[SpeechFluency] No weights found, using random initialization")
            
            self.model.eval()
            self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"[SpeechFluency] Model loading failed: {e}")
            raise
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)
        
        # Truncate if too long
        max_samples = self.max_length_seconds * self.sample_rate
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        
        return waveform
    
    def analyze(self, audio_path: str) -> FluencyResult:
        """
        Analyze speech fluency from audio file.
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            
        Returns:
            FluencyResult with scores and metrics
        """
        try:
            # Preprocess
            waveform = self.preprocess_audio(audio_path)
            duration = len(waveform) / self.sample_rate
            
            waveform = waveform.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Extract wav2vec2 features
                features = self.wav2vec2(waveform).last_hidden_state
                
                # Predict with GRU model
                outputs = self.model(features)
                
                fluency_score = outputs['fluency_score'].item() * 10  # Scale to 0-10
                hesitation_count = int(round(outputs['hesitation_count'].item()))
            
            # Estimate confidence based on model outputs
            confidence = min(1.0, fluency_score / 10 + 0.3)
            
            return FluencyResult(
                fluency_score=round(fluency_score, 1),
                hesitation_count=hesitation_count,
                confidence=round(confidence, 2),
                duration_seconds=round(duration, 1)
            )
            
        except Exception as e:
            logger.error(f"[SpeechFluency] Analysis failed: {e}")
            # Return default result on error
            return FluencyResult(
                fluency_score=5.0,
                hesitation_count=0,
                confidence=0.0,
                duration_seconds=0.0
            )
    
    def analyze_batch(self, audio_paths: list) -> list:
        """Analyze multiple audio files"""
        return [self.analyze(path) for path in audio_paths]


# Singleton instance
_service: Optional[SpeechFluencyService] = None


def get_speech_fluency_service() -> SpeechFluencyService:
    """Get or create speech fluency service singleton"""
    global _service
    if _service is None:
        _service = SpeechFluencyService()
    return _service


def analyze_speech_fluency(audio_path: str) -> Dict[str, Any]:
    """
    Convenience function to analyze speech fluency.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dict with fluency_score, hesitation_count, confidence
    """
    service = get_speech_fluency_service()
    result = service.analyze(audio_path)
    
    return {
        'fluency_score': result.fluency_score,
        'hesitation_count': result.hesitation_count,
        'confidence': result.confidence,
        'duration_seconds': result.duration_seconds
    }
