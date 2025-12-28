"""
Deep Learning Model for Personalized Learning Recommendations
Uses PyTorch for training student engagement and content recommendation models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

os.makedirs('models', exist_ok=True)


class StudentDataset(Dataset):
    """Dataset for student learning behavior"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class StudentEngagementModel(nn.Module):
    """Neural network for predicting student engagement score"""
    def __init__(self, input_dim, hidden_dims=[64, 32, 16]):
        super(StudentEngagementModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output 0-1 engagement score
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ContentRecommendationModel(nn.Module):
    """Neural collaborative filtering for content recommendations"""
    def __init__(self, n_users, n_items, embedding_dim=32,hidden_dims=[64, 32]):
        super(ContentRecommendationModel, self).__init__()
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        layers = []
        prev_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Relevance score 0-1
        
        self.fc_layers = nn.Sequential(*layers)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        x = torch.cat([user_emb, item_emb], dim=1)
        return self.fc_layers(x)


class DifficultyPredictor(nn.Module):
    """Predicts optimal difficulty level for a student on a topic"""
    def __init__(self, input_dim, n_difficulty_levels=5):
        super(DifficultyPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_difficulty_levels),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)


def generate_synthetic_data(n_samples=5000):
    """Generate synthetic training data"""
    np.random.seed(42)
    
    # Student features
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
    
    df = pd.DataFrame(features)
    
    # Create engagement score (target)
    engagement = (
        df['study_hours_weekly'] / 40 * 0.2 +
        df['completion_rate'] * 0.25 +
        df['avg_quiz_score'] * 0.2 +
        df['days_active_monthly'] / 30 * 0.15 +
        np.minimum(df['discussion_posts'] / 10, 1) * 0.1 +
        np.random.uniform(0, 0.1, n_samples)
    )
    df['engagement_score'] = np.clip(engagement, 0, 1)
    
    return df


def train_engagement_model(df, epochs=50, batch_size=32, lr=0.001):
    """Train engagement prediction model"""
    print("\n=== TRAINING ENGAGEMENT MODEL ===")
    
    # Prepare data
    feature_cols = [c for c in df.columns if c != 'engagement_score']
    X = df[feature_cols].values
    y = df['engagement_score'].values.reshape(-1, 1)
    
    # Normalize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create datasets
    train_dataset = StudentDataset(X_train, y_train)
    val_dataset = StudentDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model
    model = StudentEngagementModel(input_dim=len(feature_cols))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                val_loss += criterion(outputs, labels).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/engagement_model.pth')
    
    print(f"\nBest Validation Loss: {best_val_loss:.4f}")
    print("Model saved to: models/engagement_model.pth")
    
    return model


def predict_engagement(model, student_features):
    """Predict engagement for a student"""
    model.eval()
    with torch.no_grad():
        features = torch.FloatTensor(student_features).unsqueeze(0)
        prediction = model(features)
        return prediction.item()


def main():
    """Main training pipeline for DL models"""
    print("=" * 50)
    print("DEEP LEARNING TRAINING PIPELINE")
    print("=" * 50)
    
    # Generate/load data
    df = generate_synthetic_data()
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    
    # Train engagement model
    model = train_engagement_model(df)
    
    # Demo prediction
    print("\n=== DEMO PREDICTION ===")
    sample_student = [
        25,   # study_hours_weekly
        45,   # avg_session_duration
        0.75, # completion_rate
        10,   # quiz_attempts
        0.82, # avg_quiz_score
        20,   # days_active_monthly
        30,   # resources_accessed
        5     # discussion_posts
    ]
    
    # Normalize (using rough estimates)
    sample_normalized = np.array(sample_student)
    sample_normalized = (sample_normalized - [22.5, 65, 0.6, 10, 0.65, 15, 25, 15]) / [12, 35, 0.25, 6, 0.2, 10, 15, 10]
    
    engagement = predict_engagement(model, sample_normalized)
    print(f"Sample Student Features: {sample_student}")
    print(f"Predicted Engagement Score: {engagement:.3f}")
    
    return model


if __name__ == '__main__':
    main()
