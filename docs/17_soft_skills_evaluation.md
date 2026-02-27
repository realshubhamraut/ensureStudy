# Page 17: Soft Skills Evaluation Pipeline

---

## 17.1 Overview

The soft skills evaluation system assesses students on **non-academic competencies** during mock interviews and presentations. It uses computer vision for posture and gaze analysis, audio processing for speech fluency, and ML models for gesture recognition.

### Source Locations

| Component | Location |
|-----------|----------|
| API Routes | `backend/ai-service/app/api/routes/softskills.py` |
| Frontend | `frontend/components/softskills/` (GazeIndicator, PostureSkeleton) |
| ML Training | `ml/softskills/` (86 files) |
| Datasets | `ml/softskills/datasets/gestures/hagrid/` (HaGRID gesture dataset) |
| Inference | `ml/inference_wrappers/speech_fluency_service.py` |
| Models | `ml/models/filler_detection/` (XGBoost filler classifier) |

---

## 17.2 Evaluation Categories

| Category | Weight | Metrics | Detection Method |
|----------|--------|---------|-----------------|
| **Eye Contact** | 25% | Gaze direction, off-screen ratio | Pupil tracking (same as proctor gaze) |
| **Posture** | 20% | Spine angle, shoulder alignment | MediaPipe Pose (33 landmarks) |
| **Gestures** | 15% | Hand movement quality, nervous habits | HaGRID-based gesture classifier |
| **Speech Fluency** | 25% | Filler words, pace, pauses | Audio analysis + XGBoost classifier |
| **Confidence** | 15% | Composite of above + voice stability | Multi-signal fusion |

---

## 17.3 Eye Contact & Gaze Analysis

Reuses the proctoring `GazeTracker` but with **different thresholds**:

| Metric | Proctoring Threshold | Soft Skills Threshold |
|--------|---------------------|----------------------|
| Center gaze | > 70% required | > 50% good, > 70% excellent |
| Off-screen | < 30% warning | < 50% acceptable |
| Scoring | Binary (suspicious/ok) | Gradient (1-10 scale) |

### Frontend Component: `GazeIndicator.tsx`

Displays a real-time visual indicator showing where the student is looking, with color-coded feedback (green = camera, yellow = slightly off, red = looking away).

---

## 17.4 Posture Analysis

### MediaPipe Pose Integration

Uses 33 body landmarks to calculate:

| Metric | Calculation | Good Range |
|--------|-------------|-----------|
| Spine angle | Angle between shoulders and hips | 80°-100° (upright) |
| Shoulder alignment | Left-right shoulder Y difference | < 5° tilt |
| Head tilt | Head center vs shoulder midpoint | < 10° lateral |
| Leaning | Torso center displacement over time | < 15% frame width |
| Fidgeting | Movement variance over 10-second window | Low variance = good |

### Frontend Component: `PostureSkeleton.tsx`

Renders a skeleton overlay on the video feed showing detected landmarks with color-coded joints (green for good posture, red for poor).

---

## 17.5 Gesture Recognition

### HaGRID Dataset (Hand Gesture Recognition Image Dataset)

Source: `ml/softskills/datasets/gestures/hagrid/`

| Config | Model | Purpose |
|--------|-------|---------|
| `ConvNeXt_base.yaml` | ConvNeXt-B | Highest accuracy |
| `ResNet152.yaml` | ResNet-152 | Good accuracy, moderate speed |
| `ResNet18.yaml` | ResNet-18 | Fast, lightweight |
| `MobileNetV3_large.yaml` | MobileNetV3-L | Mobile-optimized |
| `MobileNetV3_small.yaml` | MobileNetV3-S | Ultra-lightweight |
| `VitB16.yaml` | Vision Transformer B/16 | Transformer-based |
| `SSDLiteMobileNetV3Large.yaml` | SSD + MobileNetV3 | Detection + classification |

### Gesture Categories

Classifies hand gestures during presentations:
- **Positive**: Open palms, pointing, illustrative gestures
- **Neutral**: Hands at sides, folded
- **Negative**: Fidgeting, touching face, crossed arms, nervous tapping

---

## 17.6 Speech Fluency Analysis

### Filler Word Detection

Source: `ml/models/filler_detection/`

| Model File | Type | Purpose |
|-----------|------|---------|
| `xgboost_filler_classifier.joblib` | XGBoost | Classify speech segments as filler/non-filler |
| `feature_scaler.joblib` | StandardScaler | Normalize audio features |
| `label_encoder.joblib` | LabelEncoder | Encode filler categories |

### Training Pipeline

Source: `ml/notebooks/speech_fluency_complete.ipynb`, `ml/scripts/train_fluency_model.py`

Features extracted from audio:
- MFCC coefficients (13 features)
- Pitch variation
- Speech rate (words per minute)
- Pause duration and frequency
- Energy contour

### Inference Service

Source: `ml/inference_wrappers/speech_fluency_service.py`

| Metric | Description | Scoring |
|--------|-------------|---------|
| Filler frequency | "um", "uh", "like", "you know" per minute | < 3/min = excellent |
| Speech pace | Words per minute | 120-160 WPM = good |
| Pause ratio | Silence as % of total time | 15-25% = natural |
| Pitch variation | Standard deviation of F0 | Moderate = engaging |
| Voice stability | Tremor/jitter in voice | Low = confident |

---

## 17.7 Scoring & Feedback

### Per-Session Report

```json
{
    "overall_score": 7.2,
    "categories": {
        "eye_contact": {"score": 8.0, "feedback": "Good eye contact, maintained camera focus 72% of time"},
        "posture": {"score": 6.5, "feedback": "Slight forward lean detected, try sitting more upright"},
        "gestures": {"score": 7.0, "feedback": "Natural hand movements, occasional fidgeting noted"},
        "speech_fluency": {"score": 7.5, "feedback": "Clear speech, 2.1 filler words/min (good)"},
        "confidence": {"score": 7.0, "feedback": "Steady voice, good pace at 142 WPM"}
    },
    "improvement_suggestions": [
        "Practice maintaining an upright posture",
        "Reduce slight fidgeting with hands when pausing"
    ]
}
```

---

## 17.8 Integration with Mock Interviews

The soft skills evaluation runs **alongside** mock interview sessions:

```
Student starts mock interview
  → Video + Audio captured
  → Soft skills detectors analyze in real-time
  → Interview questions scored by AI (content quality)
  → Soft skills scored by ML pipeline (delivery quality)
  → Combined report: content score + delivery score
```
