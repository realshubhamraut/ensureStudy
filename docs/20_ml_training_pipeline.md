# Page 20: ML Training Pipeline & Model Registry

---

## 20.1 Overview

The `ml/` directory contains **model training scripts, Jupyter notebooks, datasets, and inference wrappers** for all ML models used in ensureStudy. It covers student engagement prediction, content recommendation, proctoring model training, OCR model development, and speech analysis.

### Source: `ml/` (86 files)

---

## 20.2 PyTorch Models

### Source: `ml/deep_learning_models.py` (256 lines)

#### Model 1: StudentEngagementModel

```python
class StudentEngagementModel(nn.Module):
    """Predicts 0-1 engagement score from student behavior features"""
    # Architecture: Linear(input→64) → BN → ReLU → Dropout(0.3)
    #            → Linear(64→32) → BN → ReLU → Dropout(0.3)
    #            → Linear(32→16) → BN → ReLU → Dropout(0.3)
    #            → Linear(16→1) → Sigmoid
```

| Property | Value |
|----------|-------|
| Input features | 8 (study_hours, session_duration, completion_rate, quiz_attempts, quiz_score, days_active, resources_accessed, discussion_posts) |
| Output | Single float (0-1 engagement score) |
| Hidden layers | [64, 32, 16] with BatchNorm + Dropout(0.3) |
| Loss | MSE |
| Optimizer | Adam (lr=0.001) |
| Training data | Synthetic (5000 samples) |
| Saved to | `models/engagement_model.pth` |

#### Model 2: ContentRecommendationModel

```python
class ContentRecommendationModel(nn.Module):
    """Neural collaborative filtering for content recommendations"""
    # Architecture: User Embedding(32) + Item Embedding(32) → Concat
    #            → Linear(64→64) → ReLU → Dropout(0.2)
    #            → Linear(64→32) → ReLU → Dropout(0.2)
    #            → Linear(32→1) → Sigmoid
```

| Property | Value |
|----------|-------|
| Input | user_id + item_id |
| Embedding dim | 32 |
| Output | Relevance score (0-1) |
| Architecture | Neural Collaborative Filtering (NCF) |
| Use case | Recommend study materials to students |

#### Model 3: DifficultyPredictor

```python
class DifficultyPredictor(nn.Module):
    """Predicts optimal difficulty level for a student"""
    # Architecture: Linear(input→64) → ReLU → Dropout(0.3)
    #            → Linear(64→32) → ReLU → Dropout(0.2)
    #            → Linear(32→5) → Softmax
```

| Property | Value |
|----------|-------|
| Input | Student performance features |
| Output | 5-class probability (very_easy, easy, medium, hard, very_hard) |
| Use case | Adaptive difficulty for questions and content |

---

## 20.3 Training Notebooks

### Source: `ml/notebooks/` (15 notebooks)

| Notebook | Purpose |
|----------|---------|
| `proctor_training_overview.ipynb` | Overview of proctoring model training pipeline |
| `proctor_feature_extraction.ipynb` | Extract features from proctoring video data |
| `proctor_static_model.ipynb` | Train LightGBM static classifier |
| `proctor_temporal_model.ipynb` | Train LSTM temporal predictor |
| `AI_Proctoring_System_VIVA.ipynb` | Complete proctoring system documentation |
| `speech_fluency_complete.ipynb` | Full speech fluency analysis pipeline |
| `speech_fluency_train.ipynb` | Train filler detection model |
| `filler_detection_demo.py` | Demo script for filler detection |
| `answer_scoring.ipynb` | Train answer scoring model |
| `deep_learning_models.ipynb` | Train engagement/recommendation models |
| `htr_model_training.ipynb` | Handwritten text recognition training |
| `image_preprocessing.ipynb` | Image enhancement for OCR pipeline |
| `digitize_layout_detection.ipynb` | Document layout analysis training |
| `digitize_notes_pipeline.ipynb` | Notes digitization pipeline |
| `digitize_pdf_processing.ipynb` | PDF processing optimizations |
| `digitize_semantic_search.ipynb` | Semantic search for digitized notes |
| `question_paper_extraction.ipynb` | Extract questions from exam papers |
| `student_performance.ipynb` | Student performance analytics |

---

## 20.4 Model Registry

### Trained Model Weights

| Model | Location | Format | Size |
|-------|----------|--------|------|
| Engagement predictor | `models/engagement_model.pth` | PyTorch | ~50 KB |
| LightGBM (proctoring) | `proctor/models/weights/lightgbm_cheating_model_*.pkl` | joblib | ~500 KB |
| Feature scaler (proctoring) | `proctor/models/weights/scaler_*.pkl` | joblib | ~10 KB |
| Model metadata | `proctor/models/weights/model_metadata_*.pkl` | joblib | ~5 KB |
| LSTM temporal | `proctor/models/weights/temporal_proctor_trained_on_processed.pt` | PyTorch | ~2 MB |
| YOLOv11n (objects) | `proctor/models/weights/OEP_YOLOv11n.pt` | Ultralytics | ~6 MB |
| Face landmarks (68pt) | `proctor/models/weights/shape_predictor_68_face_landmarks.dat` | dlib | ~99 MB |
| Face landmarker | `proctor/models/weights/face_landmarker.task` | MediaPipe | ~5 MB |
| XGBoost filler det. | `ml/models/filler_detection/xgboost_filler_classifier.joblib` | joblib | ~200 KB |
| Filler scaler | `ml/models/filler_detection/feature_scaler.joblib` | joblib | ~10 KB |
| Filler label encoder | `ml/models/filler_detection/label_encoder.joblib` | joblib | ~5 KB |
| Embedding model | External (`all-mpnet-base-v2`) | HuggingFace | ~420 MB |

---

## 20.5 Training Pipeline

### Synthetic Data Generation

```python
def generate_synthetic_data(n_samples=5000):
    features = {
        'study_hours_weekly': np.random.uniform(5, 40, n_samples),
        'avg_session_duration': np.random.uniform(10, 120, n_samples),
        'completion_rate': np.random.uniform(0.2, 1.0, n_samples),
        'quiz_attempts': np.random.randint(1, 20, n_samples),
        'avg_quiz_score': np.random.uniform(0.3, 1.0, n_samples),
        'days_active_monthly': np.random.randint(1, 30, n_samples),
        'resources_accessed': np.random.randint(1, 50, n_samples),
        'discussion_posts': np.random.randint(0, 30, n_samples),
    }
    
    # Engagement = weighted combination + noise
    engagement = (
        study_hours/40 * 0.2 + completion_rate * 0.25 +
        quiz_score * 0.2 + days_active/30 * 0.15 +
        min(discussion_posts/10, 1) * 0.1 + noise
    )
    return df
```

### Training Loop

```python
# Standard PyTorch training
for epoch in range(50):
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation with early stopping (save best)
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), 'models/engagement_model.pth')
```

---

## 20.6 Inference Wrappers

### Source: `ml/inference_wrappers/speech_fluency_service.py`

Provides production-ready inference interfaces:

```python
class SpeechFluencyService:
    def __init__(self):
        self.filler_model = joblib.load("models/filler_detection/xgboost_filler_classifier.joblib")
        self.scaler = joblib.load("models/filler_detection/feature_scaler.joblib")
        
    def analyze(self, audio_path):
        features = extract_audio_features(audio_path)
        scaled = self.scaler.transform(features)
        predictions = self.filler_model.predict(scaled)
        
        return {
            "filler_count": sum(predictions),
            "filler_rate": sum(predictions) / len(predictions),
            "segments": [...]
        }
```

---

## 20.7 ML Technology Stack

| Framework | Version | Use Case |
|-----------|---------|----------|
| PyTorch | Latest | Engagement, recommendation, LSTM temporal |
| scikit-learn | Latest | Preprocessing, evaluation metrics |
| XGBoost | Latest | Filler detection classifier |
| LightGBM | Latest | Static proctoring classifier |
| Ultralytics (YOLO) | v11 | Prohibited object detection |
| MediaPipe | Latest | Face, pose, hand detection |
| dlib | Latest | Face landmark detection |
| DeepFace | Latest | Face verification |
| sentence-transformers | Latest | Text embedding (all-mpnet-base-v2) |
| OpenCV | Latest | Image processing, frame analysis |
| librosa | Latest | Audio feature extraction |
| joblib | Latest | Model serialization |
| NumPy / Pandas | Latest | Data manipulation |
| Jupyter | Latest | Interactive development |
