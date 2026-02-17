"""
Audio-Based Fluency Analyzer Service

Uses pre-trained HuggingFace wav2vec2 for audio analysis.
Detects filler words, hesitations, and calculates fluency scores.
"""

import os
import io
import re
import logging
import tempfile
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Common filler patterns for text-based detection
FILLER_PATTERNS = [
    r"\bum+\b", r"\buh+\b", r"\blike\b", r"\byou know\b",
    r"\bbasically\b", r"\bactually\b", r"\bliterally\b",
    r"\bi mean\b", r"\bkind of\b", r"\bsort of\b"
]

# Optimal WPM range
OPTIMAL_WPM_MIN = 120
OPTIMAL_WPM_MAX = 160


@dataclass
class AudioFluencyResult:
    """Result from audio-based fluency analysis."""
    score: float  # 0-100 overall fluency score
    wpm_score: float  # Words per minute score
    filler_score: float  # Filler word penalty score
    pause_score: float  # Pause analysis score
    clarity_score: float  # Audio clarity score
    filler_count: int
    filler_words: List[str]
    estimated_wpm: float
    pause_ratio: float
    duration_seconds: float
    feedback: str
    suggestions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "wpm_score": round(self.wpm_score, 1),
            "filler_score": round(self.filler_score, 1),
            "pause_score": round(self.pause_score, 1),
            "clarity_score": round(self.clarity_score, 1),
            "filler_count": self.filler_count,
            "filler_words": self.filler_words,
            "estimated_wpm": round(self.estimated_wpm, 1),
            "pause_ratio": round(self.pause_ratio, 3),
            "duration_seconds": round(self.duration_seconds, 1),
            "feedback": self.feedback,
            "suggestions": self.suggestions
        }


class AudioFluencyAnalyzer:
    """
    Analyzes audio for speech fluency using:
    1. Librosa for audio feature extraction
    2. Whisper for transcription (optional)
    3. Rule-based filler detection on transcript
    """
    
    def __init__(self):
        self._whisper_model = None
        self._loaded = False
        logger.info("[AudioFluency] Initialized")
    
    def _load_whisper(self):
        """Lazy load Whisper for transcription."""
        if self._whisper_model is None:
            try:
                import whisper
                self._whisper_model = whisper.load_model("tiny")
                logger.info("[AudioFluency] Loaded Whisper tiny model")
            except Exception as e:
                logger.warning(f"[AudioFluency] Whisper not available: {e}")
                self._whisper_model = False
        return self._whisper_model
    
    def _extract_audio_features(self, audio_path: str) -> Dict[str, float]:
        """Extract audio features using librosa."""
        try:
            import librosa
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            duration = len(y) / sr
            
            # RMS energy for detecting pauses
            rms = librosa.feature.rms(y=y)[0]
            silence_threshold = np.percentile(rms, 10)
            pause_frames = np.sum(rms < silence_threshold)
            total_frames = len(rms)
            pause_ratio = pause_frames / total_frames if total_frames > 0 else 0
            
            # Spectral features for clarity estimation
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            # Zero crossing rate (speech vs noise)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Clarity score based on spectral features
            clarity = min(100, 50 + (spectral_centroid / 100) + (1 - zcr) * 30)
            
            return {
                "duration": duration,
                "pause_ratio": pause_ratio,
                "rms_mean": float(np.mean(rms)),
                "spectral_centroid": float(spectral_centroid),
                "clarity": float(clarity)
            }
            
        except Exception as e:
            logger.error(f"[AudioFluency] Feature extraction error: {e}")
            return {
                "duration": 0,
                "pause_ratio": 0.2,
                "rms_mean": 0,
                "spectral_centroid": 0,
                "clarity": 70
            }
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper."""
        model = self._load_whisper()
        if model and model is not False:
            try:
                result = model.transcribe(audio_path, language="en")
                return result.get("text", "")
            except Exception as e:
                logger.error(f"[AudioFluency] Transcription error: {e}")
        return ""
    
    def _detect_fillers(self, text: str) -> Tuple[int, List[str]]:
        """Detect filler words in transcript."""
        text_lower = text.lower()
        detected = []
        total_count = 0
        
        for pattern in FILLER_PATTERNS:
            matches = re.findall(pattern, text_lower)
            if matches:
                detected.extend(matches)
                total_count += len(matches)
        
        return total_count, list(set(detected))
    
    def _calculate_wpm(self, text: str, duration: float) -> float:
        """Calculate words per minute."""
        if duration <= 0:
            return 0
        words = len([w for w in text.split() if w.strip()])
        return (words / duration) * 60
    
    def _score_wpm(self, wpm: float) -> float:
        """Score WPM on 0-100 scale."""
        if OPTIMAL_WPM_MIN <= wpm <= OPTIMAL_WPM_MAX:
            return 100.0
        elif wpm < OPTIMAL_WPM_MIN:
            return max(0, 100 - (OPTIMAL_WPM_MIN - wpm) * 1.5)
        else:
            return max(0, 100 - (wpm - OPTIMAL_WPM_MAX) * 1.0)
    
    def _score_fillers(self, count: int) -> float:
        """Score based on filler count."""
        return max(0, 100 - count * 8)
    
    def _score_pauses(self, pause_ratio: float) -> float:
        """Score based on pause ratio."""
        return max(0, 100 - pause_ratio * 200)
    
    async def analyze_audio(
        self,
        audio_data: bytes,
        transcript: str = "",
        content_type: str = "audio/webm"
    ) -> AudioFluencyResult:
        """
        Analyze audio for fluency.
        
        Args:
            audio_data: Raw audio bytes
            transcript: Optional pre-existing transcript
            content_type: Audio MIME type
        
        Returns:
            AudioFluencyResult with scores and feedback
        """
        logger.info(f"[AudioFluency] Analyzing {len(audio_data)} bytes")
        
        # Save to temp file
        suffix = ".webm" if "webm" in content_type else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_data)
            temp_path = f.name
        
        try:
            # Extract audio features
            features = self._extract_audio_features(temp_path)
            duration = features["duration"]
            pause_ratio = features["pause_ratio"]
            clarity = features["clarity"]
            
            # Transcribe if no transcript provided
            if not transcript:
                transcript = self._transcribe_audio(temp_path)
            
            # Detect fillers
            filler_count, filler_words = self._detect_fillers(transcript)
            
            # Calculate WPM
            wpm = self._calculate_wpm(transcript, duration)
            
            # Calculate scores
            wpm_score = self._score_wpm(wpm)
            filler_score = self._score_fillers(filler_count)
            pause_score = self._score_pauses(pause_ratio)
            clarity_score = min(100, clarity)
            
            # Weighted overall score
            overall = (
                wpm_score * 0.25 +
                filler_score * 0.30 +
                pause_score * 0.20 +
                clarity_score * 0.25
            )
            
            # Generate feedback
            if overall >= 80:
                feedback = "Excellent fluency! Clear and confident speech."
            elif overall >= 60:
                feedback = "Good fluency. Minor improvements possible."
            elif overall >= 40:
                feedback = "Average fluency. Work on reducing hesitations."
            else:
                feedback = "Needs improvement. Practice speaking more smoothly."
            
            # Generate suggestions
            suggestions = []
            if filler_count > 3:
                suggestions.append(f"Reduce filler words like '{', '.join(filler_words[:2])}'")
            if wpm < 100:
                suggestions.append("Try speaking a bit faster")
            elif wpm > 180:
                suggestions.append("Slow down for better clarity")
            if pause_ratio > 0.3:
                suggestions.append("Reduce long pauses between words")
            if not suggestions:
                suggestions.append("Keep practicing to maintain your fluency")
            
            return AudioFluencyResult(
                score=overall,
                wpm_score=wpm_score,
                filler_score=filler_score,
                pause_score=pause_score,
                clarity_score=clarity_score,
                filler_count=filler_count,
                filler_words=filler_words,
                estimated_wpm=wpm,
                pause_ratio=pause_ratio,
                duration_seconds=duration,
                feedback=feedback,
                suggestions=suggestions[:3]
            )
            
        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def analyze_audio_sync(
        self,
        audio_data: bytes,
        transcript: str = "",
        content_type: str = "audio/webm"
    ) -> AudioFluencyResult:
        """Synchronous version of analyze_audio."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.analyze_audio(audio_data, transcript, content_type)
        )


# Singleton
_analyzer = None


def get_audio_fluency_analyzer() -> AudioFluencyAnalyzer:
    """Get singleton AudioFluencyAnalyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = AudioFluencyAnalyzer()
    return _analyzer
