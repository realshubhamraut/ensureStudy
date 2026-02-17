#!/usr/bin/env python3
"""
🎤 Speech Fluency Analysis - Complete ML Pipeline
Optimized for M1 MacBook Air (Python 3.14 compatible)

Author: Shubham Raut
Project: EnsureStudy - AI-Powered Interview Preparation
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd
from pathlib import Path
import random
from collections import Counter
from tqdm import tqdm

# Audio processing (librosa only - torchaudio has Python 3.14 issues)
import torch
import librosa

# ML/DL
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib

print("=" * 60)
print("🎤 Speech Fluency Analysis - Complete ML Pipeline")
print("=" * 60)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'🖥️ Using device: {device}')
print(f'📦 PyTorch version: {torch.__version__}')

# ============================================
# 1. Data Loading
# ============================================
print("\n" + "=" * 60)
print("📥 Loading PodcastFillers Dataset")
print("=" * 60)

from datasets import load_dataset

dataset = load_dataset('remyxai/podcastfillers', split='train')
print(f'✅ Dataset loaded! Total samples: {len(dataset):,}')

# Sample info
sample = dataset[0]
print('\n🔍 Sample data structure:')
for key, value in sample.items():
    if key == 'audio':
        print(f'  {key}: array shape = {len(value["array"])}, sr = {value["sampling_rate"]}')
    else:
        print(f'  {key}: {value}')


# ============================================
# 2. Feature Extraction
# ============================================
print("\n" + "=" * 60)
print("🔧 Feature Extraction")
print("=" * 60)

def extract_audio_features(audio_array, sample_rate=16000):
    """
    Extract comprehensive audio features for ML.
    """
    features = {}
    
    # Ensure numpy array
    if isinstance(audio_array, torch.Tensor):
        audio_array = audio_array.numpy()
    audio_array = audio_array.astype(np.float32)
    
    try:
        # 1. MFCC (13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # 2. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=sample_rate)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # 3. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_array)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 4. RMS Energy
        rms = librosa.feature.rms(y=audio_array)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # 5. Duration
        features['duration'] = len(audio_array) / sample_rate
        
    except Exception as e:
        # Return zeros on error
        for i in range(13):
            features[f'mfcc_{i}_mean'] = 0.0
            features[f'mfcc_{i}_std'] = 0.0
        features['spectral_centroid_mean'] = 0.0
        features['spectral_centroid_std'] = 0.0
        features['spectral_rolloff_mean'] = 0.0
        features['spectral_rolloff_std'] = 0.0
        features['zcr_mean'] = 0.0
        features['zcr_std'] = 0.0
        features['rms_mean'] = 0.0
        features['rms_std'] = 0.0
        features['duration'] = 0.0
    
    return features


def extract_features_batch(dataset, max_samples=3000):
    """Extract features for multiple samples"""
    all_features = []
    labels = []
    
    for i, item in enumerate(tqdm(dataset, desc='Extracting features', total=min(max_samples, len(dataset)))):
        if i >= max_samples:
            break
        
        try:
            audio = item['audio']
            features = extract_audio_features(audio['array'], audio['sampling_rate'])
            all_features.append(features)
            
            # Get label
            if 'label' in item:
                labels.append(str(item['label']))
            elif 'filler_type' in item:
                labels.append(str(item['filler_type']))
            else:
                labels.append('unknown')
        except Exception as e:
            continue
    
    feature_df = pd.DataFrame(all_features)
    feature_df['label'] = labels
    return feature_df


# Extract features (limit to 3000 for speed on M1)
MAX_SAMPLES = 3000
print(f'🔄 Extracting audio features from {MAX_SAMPLES} samples...')
feature_df = extract_features_batch(dataset, max_samples=MAX_SAMPLES)
print(f'\n✅ Feature DataFrame shape: {feature_df.shape}')


# ============================================
# 3. EDA Summary
# ============================================
print("\n" + "=" * 60)
print("📊 Exploratory Data Analysis")
print("=" * 60)

print('\n🏷️ Label distribution:')
print(feature_df['label'].value_counts())


# ============================================
# 4. Traditional ML Models
# ============================================
print("\n" + "=" * 60)
print("🏋️ Training ML Models")
print("=" * 60)

# Prepare data
X = feature_df.drop('label', axis=1)
y = feature_df['label']

# Handle missing values
X = X.fillna(X.mean())

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f'📊 Classes: {le.classes_}')
print(f'📊 X shape: {X.shape}, y shape: {y_encoded.shape}')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'\n✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}')

# Train models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=SEED, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=SEED)
}

results = {}

for name, model in models.items():
    print(f'\n🔄 Training {name}...')
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'accuracy': acc,
        'predictions': y_pred
    }
    
    print(f'✅ {name} Accuracy: {acc:.4f}')


# ============================================
# 5. Best Model Evaluation
# ============================================
print("\n" + "=" * 60)
print("🏆 Best Model Evaluation")
print("=" * 60)

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f'\n🏆 Best Model: {best_model_name}')
print(f'📊 Accuracy: {results[best_model_name]["accuracy"]:.4f}')
print('\n📋 Classification Report:')
print(classification_report(y_test, best_predictions, target_names=le.classes_))


# ============================================
# 6. Model Export
# ============================================
print("\n" + "=" * 60)
print("💾 Saving Models")
print("=" * 60)

MODEL_DIR = Path(__file__).parent.parent / "models" / "filler_detection"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Save best model
joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
joblib.dump(scaler, MODEL_DIR / "feature_scaler.joblib")
joblib.dump(le, MODEL_DIR / "label_encoder.joblib")

# Save config
import json
config = {
    "model_type": best_model_name,
    "num_classes": len(le.classes_),
    "classes": le.classes_.tolist(),
    "feature_names": X.columns.tolist(),
    "training_samples": len(feature_df),
    "accuracy": results[best_model_name]["accuracy"]
}

with open(MODEL_DIR / "config.json", 'w') as f:
    json.dump(config, f, indent=2)

print(f'✅ Models saved to: {MODEL_DIR}')
print(f'📁 Files:')
for f in MODEL_DIR.iterdir():
    print(f'  - {f.name}')

print("\n" + "=" * 60)
print("🎉 Training Complete!")
print("=" * 60)
