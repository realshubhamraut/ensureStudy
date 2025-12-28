# Proctoring Model Training

This folder contains scripts and utilities for training AI proctoring models.

## Contents

| File | Description |
|------|-------------|
| `temporal_trainer.py` | LSTM/GRU temporal model training (from AutoOEP) |
| `temporal_models.py` | Neural network model definitions |
| `static_trainer.py` | LightGBM/XGBoost static model training (from AutoOEP) |
| `feature_extractor.py` | Extract features from video frames |
| `proctor.py` | Core proctoring logic for feature extraction |
| `video_proctor.py` | Real-time video proctoring with dual cameras |
| `VisionUtils/` | Computer vision utilities (face, hand, YOLO) |
| `original_scripts/` | Original scripts from AI-based-OEP module |

## Original Projects

This code is consolidated from:
- **AutoOEP** - Multi-modal exam proctoring with LSTM behavior analysis
- **AI-Intelligence-based-Online-Exam-Proctoring-System** - Face/eye/audio detection

## Usage

### Training Static Model (LightGBM/XGBoost)
```bash
python static_trainer.py --train-dir ../data/proctoring/train --test-dir ../data/proctoring/test
```

### Training Temporal Model (LSTM)
```bash
python temporal_trainer.py --csv ../data/proctoring/features.csv --output ./models
```

### Feature Extraction
```bash
python feature_extractor.py --dataset /path/to/dataset --target /path/to/target_image.jpg --output-dir ../data/proctoring
```

## Model Weights

Pre-trained model weights are in:
- `../models/proctoring/final_models.zip` - AutoOEP trained models
- `../../backend/ai-service/app/proctor/models/weights/` - Deployed models

## Notebooks

For interactive training, see `../notebooks/`:
- `proctor_training_overview.ipynb`
- `proctor_static_model.ipynb`
- `proctor_temporal_model.ipynb`
