"""
Audio Detector - Detects suspicious audio during exams

Refactored from: Artificial-Intelligence-based-Online-Exam-Proctoring-System/audio_detection.py

Features:
- Analyzes audio amplitude for suspicious sounds
- Cross-platform support (no winsound dependency)
- Configurable threshold for sensitivity
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AudioAnalysisResult:
    """Result of audio analysis"""
    suspicious: bool
    amplitude: float
    threshold: float
    message: str


class AudioDetector:
    """
    Detects suspicious audio patterns during exams.
    
    Analyzes audio samples for:
    - Loud noises (potential speech)
    - Sustained noise above threshold
    - Frequency patterns (optional)
    
    Note: Requires PyAudio for microphone access. Falls back gracefully if unavailable.
    """
    
    # Default configuration
    DEFAULT_THRESHOLD = 2000  # Amplitude threshold for suspicious audio
    DEFAULT_SAMPLE_RATE = 44100
    DEFAULT_CHUNK_SIZE = 1024
    
    def __init__(
        self, 
        threshold: int = DEFAULT_THRESHOLD,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_size: int = DEFAULT_CHUNK_SIZE
    ):
        """
        Initialize audio detector.
        
        Args:
            threshold: Amplitude threshold for detecting suspicious audio
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of frames per buffer
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Track consecutive suspicious detections
        self._consecutive_suspicious = 0
        self._total_samples = 0
        self._suspicious_samples = 0
        
        # PyAudio availability check (lazy)
        self._pyaudio_available: Optional[bool] = None
    
    def _check_pyaudio(self) -> bool:
        """Check if PyAudio is available"""
        if self._pyaudio_available is None:
            try:
                import pyaudio
                self._pyaudio_available = True
                logger.info("PyAudio is available for audio detection")
            except ImportError:
                self._pyaudio_available = False
                logger.warning("PyAudio not installed. Audio detection disabled.")
        return self._pyaudio_available
    
    def analyze_samples(self, audio_data: bytes) -> AudioAnalysisResult:
        """
        Analyze audio samples for suspicious patterns.
        
        Args:
            audio_data: Raw audio bytes (int16 format)
            
        Returns:
            AudioAnalysisResult with detection status
        """
        try:
            # Convert bytes to numpy array
            samples = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate amplitude
            amplitude = float(np.max(np.abs(samples)))
            
            self._total_samples += 1
            
            # Check if amplitude exceeds threshold
            is_suspicious = amplitude > self.threshold
            
            if is_suspicious:
                self._suspicious_samples += 1
                self._consecutive_suspicious += 1
                message = f"Suspicious audio detected (amplitude: {amplitude:.0f})"
            else:
                self._consecutive_suspicious = 0
                message = "Audio normal"
            
            return AudioAnalysisResult(
                suspicious=is_suspicious,
                amplitude=amplitude,
                threshold=self.threshold,
                message=message
            )
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return AudioAnalysisResult(
                suspicious=False,
                amplitude=0.0,
                threshold=self.threshold,
                message=f"Analysis error: {str(e)}"
            )
    
    def analyze_base64(self, audio_base64: str) -> Dict[str, Any]:
        """
        Analyze base64-encoded audio data.
        
        Args:
            audio_base64: Base64 encoded audio samples
            
        Returns:
            Dict with analysis results
        """
        import base64
        
        try:
            audio_bytes = base64.b64decode(audio_base64)
            result = self.analyze_samples(audio_bytes)
            
            return {
                "suspicious": result.suspicious,
                "amplitude": result.amplitude,
                "threshold": result.threshold,
                "message": result.message,
                "consecutive_suspicious": self._consecutive_suspicious,
                "suspicious_ratio": self._suspicious_samples / max(1, self._total_samples)
            }
            
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return {
                "suspicious": False,
                "amplitude": 0.0,
                "threshold": self.threshold,
                "message": f"Decode error: {str(e)}",
                "consecutive_suspicious": 0,
                "suspicious_ratio": 0.0
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get accumulated audio metrics"""
        return {
            "total_samples": self._total_samples,
            "suspicious_samples": self._suspicious_samples,
            "suspicious_ratio": self._suspicious_samples / max(1, self._total_samples),
            "current_consecutive_suspicious": self._consecutive_suspicious,
            "threshold": self.threshold
        }
    
    def reset(self):
        """Reset detection counters"""
        self._consecutive_suspicious = 0
        self._total_samples = 0
        self._suspicious_samples = 0
