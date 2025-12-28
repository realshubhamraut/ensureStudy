"""
Tests for Proctoring Detectors

Tests the new audio detection, blink detection, and face verification features.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestAudioDetector:
    """Tests for AudioDetector"""
    
    def test_initialization(self):
        """Test detector initializes correctly"""
        from app.proctor.detectors import AudioDetector
        
        detector = AudioDetector()
        assert detector.threshold == 2000
        assert detector._total_samples == 0
    
    def test_analyze_samples_normal(self):
        """Test normal audio below threshold"""
        from app.proctor.detectors import AudioDetector
        
        detector = AudioDetector(threshold=5000)
        
        # Create low amplitude audio
        samples = np.array([100, -100, 200, -200], dtype=np.int16)
        result = detector.analyze_samples(samples.tobytes())
        
        assert result.suspicious == False
        assert result.amplitude == 200.0
    
    def test_analyze_samples_suspicious(self):
        """Test suspicious audio above threshold"""
        from app.proctor.detectors import AudioDetector
        
        detector = AudioDetector(threshold=1000)
        
        # Create high amplitude audio
        samples = np.array([5000, -5000, 10000, -10000], dtype=np.int16)
        result = detector.analyze_samples(samples.tobytes())
        
        assert result.suspicious == True
        assert result.amplitude == 10000.0
    
    def test_metrics_accumulation(self):
        """Test that metrics accumulate correctly"""
        from app.proctor.detectors import AudioDetector
        
        detector = AudioDetector(threshold=1000)
        
        # Normal sample
        normal = np.array([100, -100], dtype=np.int16)
        detector.analyze_samples(normal.tobytes())
        
        # Suspicious sample
        suspicious = np.array([5000, -5000], dtype=np.int16)
        detector.analyze_samples(suspicious.tobytes())
        
        metrics = detector.get_metrics()
        assert metrics["total_samples"] == 2
        assert metrics["suspicious_samples"] == 1
        assert metrics["suspicious_ratio"] == 0.5


class TestBlinkDetector:
    """Tests for BlinkDetector"""
    
    def test_initialization(self):
        """Test detector initializes correctly"""
        from app.proctor.detectors import BlinkDetector
        
        detector = BlinkDetector()
        assert detector.ear_threshold == 0.25
        assert detector._total_blinks == 0
    
    def test_detect_without_landmarks(self):
        """Test detection with no landmarks"""
        from app.proctor.detectors import BlinkDetector
        
        detector = BlinkDetector()
        result = detector.detect(None)
        
        assert result["is_blinking"] == False
        assert result["total_blinks"] == 0
    
    def test_detect_with_mock_landmarks(self):
        """Test detection with mock landmarks"""
        from app.proctor.detectors import BlinkDetector
        
        detector = BlinkDetector()
        
        # Create mock landmarks (68 points)
        # Set eye points to simulate open eyes
        landmarks = np.zeros((68, 2))
        
        # Left eye (36-41): open eye shape
        landmarks[36] = [100, 100]  # left corner
        landmarks[37] = [110, 90]   # upper left
        landmarks[38] = [120, 90]   # upper right
        landmarks[39] = [130, 100]  # right corner
        landmarks[40] = [120, 110]  # lower right
        landmarks[41] = [110, 110]  # lower left
        
        # Right eye (42-47): open eye shape
        landmarks[42] = [160, 100]
        landmarks[43] = [170, 90]
        landmarks[44] = [180, 90]
        landmarks[45] = [190, 100]
        landmarks[46] = [180, 110]
        landmarks[47] = [170, 110]
        
        result = detector.detect(landmarks)
        
        # Should not be blinking with open eyes
        assert "is_blinking" in result
        assert "avg_ear" in result
        assert result["avg_ear"] > 0
    
    def test_reset(self):
        """Test reset functionality"""
        from app.proctor.detectors import BlinkDetector
        
        detector = BlinkDetector()
        detector._total_blinks = 10
        detector._frame_count = 100
        
        detector.reset()
        
        assert detector._total_blinks == 0
        assert detector._frame_count == 0


class TestFaceVerifier:
    """Tests for FaceVerifier"""
    
    def test_initialization(self):
        """Test verifier initializes correctly"""
        from app.proctor.detectors import FaceVerifier
        
        verifier = FaceVerifier()
        assert verifier.model_name == "VGG-Face"
        assert verifier._registered_face_path is None
    
    def test_verify_without_registration(self):
        """Test verification fails without registered face"""
        from app.proctor.detectors import FaceVerifier
        
        verifier = FaceVerifier()
        result = verifier.verify(np.zeros((480, 640, 3), dtype=np.uint8))
        
        assert result["verified"] == False
        assert "No reference face registered" in result["message"]
    
    def test_is_available(self):
        """Test deepface availability check"""
        from app.proctor.detectors import FaceVerifier
        
        verifier = FaceVerifier()
        # Should not crash even if deepface is not installed
        available = verifier.is_available()
        assert isinstance(available, bool)
    
    def test_metrics(self):
        """Test metrics retrieval"""
        from app.proctor.detectors import FaceVerifier
        
        verifier = FaceVerifier()
        verifier._verification_count = 10
        verifier._verified_count = 8
        
        metrics = verifier.get_metrics()
        
        assert metrics["verification_count"] == 10
        assert metrics["verified_count"] == 8
        assert metrics["verification_rate"] == 0.8


class TestProctorAPIEndpoints:
    """Tests for new API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
    
    def test_audio_stream_no_session(self, client):
        """Test audio stream endpoint with invalid session"""
        response = client.post(
            "/api/proctor/audio-stream",
            json={
                "session_id": "nonexistent",
                "audio_base64": "dGVzdA=="
            }
        )
        assert response.status_code == 404
    
    def test_register_face_no_session(self, client):
        """Test face registration with invalid session"""
        response = client.post(
            "/api/proctor/register-face",
            json={
                "session_id": "nonexistent",
                "image_base64": "dGVzdA=="
            }
        )
        assert response.status_code == 404
    
    def test_verify_face_no_session(self, client):
        """Test face verification with invalid session"""
        response = client.post(
            "/api/proctor/verify-face",
            json={
                "session_id": "nonexistent",
                "frame_base64": "dGVzdA=="
            }
        )
        assert response.status_code == 404


class TestProctorSessionIntegration:
    """Integration tests for ProctorSession with new detectors"""
    
    def test_session_has_new_detectors(self):
        """Test session has new detector properties"""
        from app.proctor.session import ProctorSession
        
        session = ProctorSession(
            assessment_id="test-assessment",
            student_id="test-student"
        )
        
        # Check detector properties exist
        assert hasattr(session, 'audio_detector')
        assert hasattr(session, 'blink_detector')
        assert hasattr(session, 'face_verifier')
        
        # Finalize to cleanup
        session.finalize()
