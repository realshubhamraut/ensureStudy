# Page 41: Pre-Trained Models & Model Registry

---

## 41.1 Overview

ensureStudy ships with **16 pre-trained model files** across 3 directories, covering exam proctoring, student engagement prediction, and object detection. Models are versioned with timestamps and include metadata files for reproducibility.

---

## 41.2 Model Inventory

### Source: `models/` directory

| Model File | Size | Type | Purpose |
|-----------|------|------|---------|
| `models-pretrained/OEP_YOLOv11n.pt` | ~6 MB | YOLO | Object detection (phone, book, earbuds) |
| `models-pretrained/engagement_model.pth` | ~1 MB | PyTorch | Student engagement prediction |
| `models-pretrained/lightgbm_cheating_model_20250818_132555.pkl` | ~500 KB | LightGBM | Per-frame cheating classification |
| `models-pretrained/model_metadata_20250818_132555.pkl` | ~5 KB | Pickle | Feature names, thresholds, training stats |
| `models-pretrained/scaler_20250818_132555.pkl` | ~10 KB | Pickle | Feature normalization scaler |
| `models-pretrained/temporal_proctor_trained_on_processed.pt` | ~2 MB | PyTorch | LSTM temporal behavior classifier |
| `models-pretrained/face_landmarker.task` | ~5 MB | MediaPipe | 468-point face landmark detection |
| `Models_new/xgboost_cheating_model_20251230_105224.pkl` | ~800 KB | XGBoost | Updated cheating classifier |
| `Models_new/xgboost_cheating_model_20251230_105224_metadata.pkl` | ~5 KB | Pickle | Updated model metadata |
| `engagement_model.pth` | ~1 MB | PyTorch | Engagement model (root copy) |

### Proctoring Best Models: `proctoring/best_models/`

Mirror of `models-pretrained/` for deployment:

| File | Purpose |
|------|---------|
| `OEP_YOLOv11n.pt` | YOLO object detection |
| `face_landmarker.task` | MediaPipe face landmarks |
| `lightgbm_cheating_model_20250818_132555.pkl` | Static classifier |
| `model_metadata_20250818_132555.pkl` | Model metadata |
| `scaler_20250818_132555.pkl` | Feature scaler |
| `temporal_proctor_trained_on_processed.pt` | LSTM temporal |

---

## 41.3 Model Architecture Details

### YOLOv11n (Object Detection)

```
Architecture: YOLOv11-nano
Parameters: ~2.6M
Input: 640×640 RGB frame
Output: Bounding boxes + class labels
Classes: phone, book, earbuds, person, laptop, screen
Inference: ~15ms per frame (CPU)
```

### LightGBM Static Classifier

```
Algorithm: LightGBM (Gradient Boosted Trees)
Features: 15 per-frame features from 8 detectors
  - face_detected (bool)
  - gaze_x, gaze_y (float)
  - head_yaw, head_pitch, head_roll (float)
  - eye_aspect_ratio_left, right (float)
  - mouth_aspect_ratio (float)
  - object_count (int)
  - phone_detected, book_detected (bool)
  - audio_level (float)
  - hand_near_face (bool)
Output: P(cheating) ∈ [0, 1]
Training: Labeled proctoring frames (cheating/not_cheating)
```

### LSTM Temporal Predictor

```
Architecture: 2-layer LSTM
Input: Sequence of 30 static predictions
Hidden: 64 units
Output: P(cheating_sequence) ∈ [0, 1]
Purpose: Detect sustained suspicious behavior
```

### Engagement Model (PyTorch)

```
Architecture: Multi-layer Feedforward (64 → 32 → 16 → 1)
Input: Student interaction features
Output: Engagement score ∈ [0, 1]
Features: time_on_task, click_rate, scroll_depth, quiz_attempts
Training: Student behavioral data
```

---

## 41.4 Model Loading Pattern

```python
class ModelRegistry:
    MODELS_DIR = "models/models-pretrained"
    
    _instances = {}
    
    @classmethod
    def get_model(cls, name: str):
        if name not in cls._instances:
            path = os.path.join(cls.MODELS_DIR, name)
            if name.endswith('.pt') or name.endswith('.pth'):
                cls._instances[name] = torch.load(path, map_location='cpu')
            elif name.endswith('.pkl'):
                with open(path, 'rb') as f:
                    cls._instances[name] = pickle.load(f)
            elif name.endswith('.task'):
                cls._instances[name] = MediaPipeLandmarker(path)
        return cls._instances[name]
```

---

## 41.5 Training Datasets

### Source: `datasets/proctoring_training/`

| Directory | Purpose |
|-----------|---------|
| `cheating_frames/` | Labeled positive examples (face turned away, phone visible, etc.) |
| `not_cheating_frames/` | Labeled negative examples (normal exam behavior) |

---

## 41.6 Model Versioning

Models are versioned with timestamps in filenames:

```
Format: {algorithm}_{task}_{YYYYMMDD}_{HHMMSS}.pkl

Examples:
  lightgbm_cheating_model_20250818_132555.pkl   (Aug 18, 2025)
  xgboost_cheating_model_20251230_105224.pkl    (Dec 30, 2025)
```

Each model has a corresponding `_metadata.pkl` containing:
- Feature names and order
- Training hyperparameters
- Validation metrics (precision, recall, F1)
- Training data statistics
