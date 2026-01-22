## Unified Data Directory

This directory contains all datasets and pre-trained models required for the ensureStudy platform.

### Directory Structure

```
data/
├── models-pretrained/          # Pre-trained model weights
├── proctoring-features/        # Proctoring detection datasets
├── softskills-fluency/         # Speech fluency datasets
└── softskills-gestures/        # Hand gesture recognition datasets
```

### Datasets Overview

| Dataset | Location | Size | Description |
|---------|----------|------|-------------|
| Proctoring Features | `proctoring-features/new2_csv/` | ~770KB | Extracted video features for cheating detection |
| Podcast Fillers | `softskills-fluency/podcast-fillers-hf/` | ~8GB | Audio dataset for filler word detection |
| HaGRID Gestures | `softskills-gestures/hagrid/` | ~13MB | Hand gesture recognition dataset configs |

### Pre-trained Models

| Model | File | Size | Purpose |
|-------|------|------|---------|
| YOLO Proctoring | `OEP_YOLOv11n.pt` | 5.4MB | Object detection for phones, books |
| Face Landmarker | `face_landmarker.task` | 3.8MB | MediaPipe face mesh detection |
| Temporal Proctor | `temporal_proctor_trained_on_processed.pt` | 866KB | LSTM for temporal cheating patterns |
| LightGBM Cheating | `lightgbm_cheating_model_*.pkl` | 1.7MB | Gradient boosting classifier |
| Engagement Model | `engagement_model.pth` | 23KB | Student engagement prediction |

### Proctoring Features Dataset

Frame-level features extracted from exam videos:

| Feature | Type | Description |
|---------|------|-------------|
| `face_detected` | bool | Face visible in frame |
| `face_count` | int | Number of faces detected |
| `gaze_x`, `gaze_y` | float | Eye gaze direction |
| `head_pitch`, `head_yaw` | float | Head rotation angles |
| `mouth_open` | bool | Mouth movement detection |
| `phone_detected` | bool | Mobile phone in frame |
| `label` | int | 0=normal, 1=suspicious |

### Fluency Dataset (Podcast Fillers)

Audio segments labeled with filler words:

| Column | Type | Description |
|--------|------|-------------|
| `audio` | bytes | Audio segment (parquet) |
| `text` | string | Transcription |
| `filler_words` | list | Detected fillers (um, uh, like) |
| `timestamps` | list | Filler word positions |

### Gesture Dataset (HaGRID)

Hand gesture recognition for presentation analysis:

| Gesture Class | Label | Use Case |
|---------------|-------|----------|
| call | 0 | Phone gesture |
| dislike | 1 | Thumbs down |
| like | 2 | Thumbs up |
| ok | 3 | OK sign |
| peace | 4 | Victory sign |
| stop | 5 | Stop hand |
| no_gesture | 6 | No hand detected |

### Usage

```python
# Load proctoring features
import pandas as pd
train_df = pd.read_csv('data/proctoring-features/new2_csv/train/Train_Video1_processed.csv')

# Load pre-trained model
import torch
model = torch.load('data/models-pretrained/temporal_proctor_trained_on_processed.pt')

# Load fluency dataset
from datasets import load_dataset
ds = load_dataset('parquet', data_files='data/softskills-fluency/podcast-fillers-hf/data/*.parquet')
```

### Data Sources

| Dataset | Source | License |
|---------|--------|---------|
| Proctoring Features | Internal collection | Proprietary |
| Podcast Fillers | HuggingFace Hub | CC BY 3.0 / CC BY-SA 3.0 |
| HaGRID | SberBank AI | CC BY-SA 4.0 |
