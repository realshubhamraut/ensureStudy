# Page 84: Proctoring Detectors — Deep Dive

> Supplements Pages 14, 79 with the full modular detector architecture: BlinkDetector (EAR), FaceVerifier (DeepFace), HandDetector (MediaPipe), AudioDetector, TemporalPredictor (LSTM), and unified CheatScore calculator.

---

## 84.1 Detector Architecture

```mermaid\nflowchart TB\n    subgraph DETECTORS[\"🎥 Frame Detectors\"]\n        direction LR\n        FD[\"face_detector.py<br/>YOLO + MediaPipe\"]\n        GT[\"gaze_tracker.py<br/>Iris-based gaze\"]\n        HP[\"head_pose.py<br/>PnP estimation\"]\n        BD[\"blink_detector.py<br/>EAR algorithm\"]\n        FV[\"face_verifier.py<br/>DeepFace identity\"]\n        HD[\"hand_detector.py<br/>MediaPipe hands\"]\n        AD[\"audio_detector.py<br/>Amplitude analysis\"]\n        OD[\"object_detector.py<br/>YOLO objects\"]\n    end\n\n    DETECTORS --> SC[\"static_classifier.py<br/>LightGBM per-frame\"]\n    DETECTORS --> TP[\"temporal_predictor.py<br/>LSTM sequence (15 frames)\"]\n\n    subgraph SCORING[\"📊 Scoring Pipeline\"]\n        direction LR\n        IS[\"integrity_scorer.py\"]\n        CS[\"cheat_score.py<br/>Unified scorer\"]\n        FG[\"flag_generator.py<br/>Flag rules\"]\n    end\n\n    SC -->|\"40% weight\"| CS\n    TP -->|\"60% weight\"| CS\n    CS --> FG --> IS\n\n    style BD fill:#f59e0b,color:#000\n    style FV fill:#f59e0b,color:#000\n    style HD fill:#f59e0b,color:#000\n    style AD fill:#f59e0b,color:#000\n    style TP fill:#ef4444,color:#fff\n    style CS fill:#ef4444,color:#fff\n```

---

## 84.2 BlinkDetector — Eye Aspect Ratio

### Source: `proctor/detectors/blink_detector.py` (246 lines)

Uses dlib 68-point facial landmarks (points 36-47) to calculate Eye Aspect Ratio:

```python
# EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
# Where p1-p6 are the 6 landmark points of one eye
# Open eye EAR ≈ 0.2-0.4, Closed eye EAR < 0.25

class BlinkDetector:
    LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
    DEFAULT_EAR_THRESHOLD = 0.25
    DEFAULT_CONSEC_FRAMES = 2  # Frames to confirm a blink
    
    def detect(self, landmarks: np.ndarray) -> Dict:
        left_ear = self._calculate_ear(landmarks[self.LEFT_EYE_INDICES])
        right_ear = self._calculate_ear(landmarks[self.RIGHT_EYE_INDICES])
        avg_ear = (left_ear + right_ear) / 2.0
        
        is_blinking = avg_ear < self.ear_threshold
        # Count confirmed blinks (consecutive frames above threshold)
        return {
            "is_blinking": bool,
            "left_ear": float,
            "right_ear": float,
            "avg_ear": float,
            "total_blinks": int,
            "blink_rate": float   # blinks per frame
        }
```

---

## 84.3 FaceVerifier — Identity Verification

### Source: `proctor/detectors/face_verifier.py` (255 lines)

Uses DeepFace library to verify student identity against registered photo.

```python
class FaceVerifier:
    # Supported backends
    DEFAULT_MODEL = "VGG-Face"       # Also: ArcFace, Facenet, OpenFace
    DEFAULT_BACKEND = "opencv"       # Also: retinaface, mtcnn
    DEFAULT_DISTANCE_METRIC = "cosine"  # Also: euclidean, euclidean_l2
    DEFAULT_THRESHOLD = 0.4
    
    def register_face(self, face_image: np.ndarray) -> Dict:
        """Save reference face (temp file for DeepFace)"""
    
    def register_face_base64(self, image_base64: str) -> Dict:
        """Register from base64-encoded JPEG"""
    
    def verify(self, frame: np.ndarray) -> Dict:
        """Compare live frame to registered face"""
        # Returns: {verified, confidence, distance, threshold, message}
```

### Verification Flow
```
1. register_face(photo) → saves to temp file
2. verify(webcam_frame) → DeepFace.verify(frame, reference)
   → Returns {verified: bool, confidence: 1 - distance}
```

---

## 84.4 HandDetector — MediaPipe Hands

### Source: `proctor/detectors/hand_detector.py` (218 lines)

Detects hands using MediaPipe Hands solution (21 landmarks per hand):

```python
class HandDetector:
    def __init__(self, max_hands=2, min_confidence=0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=max_hands,
            min_detection_confidence=min_confidence
        )
    
    def detect(self, frame: np.ndarray) -> Dict:
        # Returns: {num_hands, hands_visible, landmarks[], handedness[]}
```

### Use in Proctoring
- `num_hands > 0` during written exam → flag (hands should be on keyboard)
- Hand presence tracking for behavioral analysis

---

## 84.5 AudioDetector — Amplitude Analysis

### Source: `proctor/detectors/audio_detector.py` (181 lines)

Analyzes raw audio samples for suspicious sounds (speech, external noise):

```python
class AudioDetector:
    DEFAULT_THRESHOLD = 2000    # int16 amplitude
    DEFAULT_SAMPLE_RATE = 44100
    
    def analyze_samples(self, audio_data: bytes) -> AudioAnalysisResult:
        samples = np.frombuffer(audio_data, dtype=np.int16)
        amplitude = float(np.max(np.abs(samples)))
        return AudioAnalysisResult(
            suspicious=amplitude > self.threshold,
            amplitude=amplitude,
            message="Suspicious audio detected" if suspicious else "Audio normal"
        )
    
    def analyze_base64(self, audio_base64: str) -> Dict:
        """For WebSocket binary audio data"""
```

---

## 84.6 TemporalPredictor — LSTM Sequence Analysis

### Source: `proctor/temporal_predictor.py` (343 lines)

Pre-trained LSTM model from AutoOEP that analyzes **sequences** of frame features:

```python
# 15 input features per frame
FEATURE_NAMES = [
    'face_detected', 'face_count', 'object_count',
    'x_rotation', 'y_rotation', 'z_rotation',
    'radial_distance', 'gaze_direction', 'gaze_zone',
    'watch', 'headphone', 'closedbook', 'earpiece',
    'cell phone', 'openbook', 'chits', 'sheet',
    'H-Distance', 'F-Distance'
]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, fc_hidden=32, output_size=1):
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, output_size)

class TemporalPredictor:
    def __init__(self, window_size=15, threshold=0.4):
        """Sliding window of 15 frames → LSTM → cheat probability"""
    
    def add_frame(self, detection_results: Dict, timestamp: float):
        """Extract features and append to buffer"""
    
    def predict(self) -> Dict:
        """When buffer full (15 frames), run LSTM prediction"""
        # Returns: {probability, is_cheating, confidence}
```

---

## 84.7 Unified CheatScore Calculator

### Source: `proctor/scoring/cheat_score.py` (179 lines)

Combines static (per-frame LightGBM) + temporal (LSTM) + flag penalties:

```python
FLAG_WEIGHTS = {
    'phone_detected': 0.25,  'multiple_faces': 0.20,
    'no_face': 0.15,         'book_detected': 0.15,
    'looking_away': 0.10,    'suspicious_head_pose': 0.08,
    'suspicious_audio': 0.08, 'tab_switch': 0.05,
    'mouth_open_talking': 0.05, 'earpiece_detected': 0.20,
}

def calculate_cheat_score(
    static_prob,         # LightGBM per-frame (0-1)
    temporal_prob,       # LSTM sequence (0-1)
    active_flags,        # Current violation flags
    static_weight=0.4,   # 40% static
    temporal_weight=0.6  # 60% temporal
) -> Dict:
    base_score = (static_weight * static_prob) + (temporal_weight * temporal_prob)
    flag_penalty = sum(FLAG_WEIGHTS.get(f, 0.03) for f in active_flags)
    unified_score = min(1.0, base_score + flag_penalty)
    
    # Severity: <0.3=low, 0.3-0.5=medium, 0.5-0.7=high, >0.7=critical
```

### Session Integrity Report

```python
def calculate_session_integrity(frame_scores, total_flags, tab_switch_count):
    # Penalties: max_score ≥ 0.8 → -15pts, suspicious_pct/5 → up to -15pts
    # Tab switches → -2pts each (max -10pts)
    # Final: 0-100 integrity score
```

| Integrity Score | Severity | Review Required |
|----------------|----------|-----------------|
| ≥ 80 | Low | No |
| 60-79 | Medium | If suspicious > 20% |
| 40-59 | High | Yes |
| < 40 | Critical | Yes + flag for manual review |
