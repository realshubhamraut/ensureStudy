# Page 80: ML Model Architectures — PyTorch Deep Dive

> Supplements Page 46 (Pre-trained Models Inventory) with complete PyTorch model architectures, training pipelines, and serving infrastructure from `ml-models.md`.

---

## 80.1 Model Registry

| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| NeuralCollaborativeFiltering | user_id, item_id | score [0,1] | Content recommendation |
| ContentBasedRecommender | item_features, user_history | ranked scores | Similar content discovery |
| DifficultyPredictor | text_embedding, metadata | difficulty [1-5] | Content difficulty labeling |
| LearningPathOptimizer | completed_topics, mastery | next_topics | Learning path generation |
| DeepKnowledgeTracing | skill_history, correctness | mastery probabilities | Knowledge state estimation |

---

## 80.2 Neural Collaborative Filtering (NCF)

Dual-path architecture combining **GMF** (Generalized Matrix Factorization) and **MLP**:

```python
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, 
                 hidden_layers=[128, 64, 32]):
        super().__init__()
        
        # GMF path embeddings
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
        
        # MLP path embeddings
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers with dropout
        mlp_layers = []
        input_size = embedding_dim * 2
        for hidden_size in hidden_layers:
            mlp_layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = hidden_size
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final: concat GMF + MLP → prediction
        self.output = nn.Linear(hidden_layers[-1] + embedding_dim, 1)
    
    def forward(self, user_ids, item_ids):
        # GMF: element-wise product
        gmf_output = self.user_embedding_gmf(user_ids) * self.item_embedding_gmf(item_ids)
        
        # MLP: concatenation → deep layers
        mlp_input = torch.cat([
            self.user_embedding_mlp(user_ids),
            self.item_embedding_mlp(item_ids)
        ], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # Combine and predict
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        return torch.sigmoid(self.output(combined)).squeeze()
```

---

## 80.3 DifficultyPredictor

Estimates content difficulty (5 levels) from text embeddings + metadata:

```python
class DifficultyPredictor(nn.Module):
    def __init__(self, text_dim=768):
        super().__init__()
        
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # 10 metadata features: word count, sentence length, etc.
        self.meta_encoder = nn.Sequential(
            nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(160, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 5)  # 5 difficulty levels
        )
    
    def forward(self, text_embedding, metadata):
        text_features = self.text_encoder(text_embedding)
        meta_features = self.meta_encoder(metadata)
        combined = torch.cat([text_features, meta_features], dim=-1)
        return self.predictor(combined)  # logits for cross-entropy
    
    def predict_difficulty(self, text_embedding, metadata):
        logits = self.forward(text_embedding, metadata)
        return torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
```

---

## 80.4 LearningPathOptimizer

Sequence-to-sequence LSTM model for generating optimal learning paths:

```python
class LearningPathOptimizer(nn.Module):
    def __init__(self, num_topics, embedding_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        
        self.topic_embedding = nn.Embedding(num_topics, embedding_dim)
        
        # Encoder: user's learning history → hidden state
        self.user_encoder = nn.LSTM(
            input_size=embedding_dim + 1,  # embedding + mastery score
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True, dropout=0.2
        )
        
        # Decoder: generate next topic sequence
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_dim, num_topics)
    
    def forward(self, completed_topics, mastery_scores, 
                target_topics=None, max_length=10):
        # Encode learning history
        topic_emb = self.topic_embedding(completed_topics)
        encoder_input = torch.cat([topic_emb, mastery_scores.unsqueeze(-1)], dim=-1)
        _, (hidden, cell) = self.user_encoder(encoder_input)
        
        if self.training and target_topics is not None:
            # Teacher forcing
            target_emb = self.topic_embedding(target_topics)
            decoder_output, _ = self.decoder(target_emb, (hidden, cell))
            return self.output_proj(decoder_output)
        else:
            # Autoregressive generation
            outputs = []
            batch_size = completed_topics.size(0)
            decoder_input = self.topic_embedding(
                torch.zeros(batch_size, 1).long()
            )
            for _ in range(max_length):
                output, (hidden, cell) = self.decoder(
                    decoder_input, (hidden, cell)
                )
                logits = self.output_proj(output)
                next_topic = torch.argmax(logits, dim=-1)
                outputs.append(next_topic)
                decoder_input = self.topic_embedding(next_topic)
            return torch.cat(outputs, dim=1)
```

---

## 80.5 Deep Knowledge Tracing (DKT)

LSTM-based knowledge tracing for mastery estimation:

```python
class DeepKnowledgeTracing(nn.Module):
    def __init__(self, num_skills, embedding_dim=64, hidden_dim=128):
        super().__init__()
        
        self.skill_embedding = nn.Embedding(num_skills, embedding_dim)
        self.correct_embedding = nn.Embedding(2, embedding_dim)  # 0/1
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True, dropout=0.2
        )
        
        self.output = nn.Linear(hidden_dim, num_skills)
    
    def forward(self, skill_ids, correctness):
        """
        skill_ids: (batch, seq_len) — practiced skills
        correctness: (batch, seq_len) — 0/1 for incorrect/correct
        Returns: (batch, seq_len, num_skills) — mastery probability
        """
        combined = torch.cat([
            self.skill_embedding(skill_ids),
            self.correct_embedding(correctness)
        ], dim=-1)
        lstm_out, _ = self.lstm(combined)
        return torch.sigmoid(self.output(lstm_out))
    
    def predict_mastery(self, skill_history, correct_history):
        with torch.no_grad():
            return self.forward(skill_history, correct_history)[:, -1, :]
```

---

## 80.6 Training Pipeline with MLflow

```python
class ModelTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train(self, train_loader, val_loader):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3
        )
        criterion = (nn.BCELoss() if self.config['task'] == 'binary' 
                     else nn.CrossEntropyLoss())
        
        mlflow.set_experiment(self.config['experiment_name'])
        
        with mlflow.start_run():
            mlflow.log_params(self.config)
            best_val_loss = float('inf')
            
            for epoch in range(self.config['epochs']):
                # Train
                self.model.train()
                train_loss = sum(
                    self._train_batch(batch, optimizer, criterion)
                    for batch in train_loader
                ) / len(train_loader)
                
                # Validate
                val_loss, val_metrics = self.evaluate(val_loader, criterion)
                scheduler.step(val_loss)
                
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss, **val_metrics
                }, step=epoch)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    mlflow.pytorch.log_model(self.model, 'model')
```

---

## 80.7 ONNX Export & Serving

```python
# Export to ONNX
def export_to_onnx(model, sample_input, output_path):
    model.eval()
    torch.onnx.export(
        model, sample_input, output_path,
        input_names=['user_ids', 'item_ids'],
        output_names=['predictions'],
        dynamic_axes={
            'user_ids': {0: 'batch'},
            'item_ids': {0: 'batch'},
            'predictions': {0: 'batch'}
        }
    )

# Serve via ONNX Runtime
class ModelServer:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
    
    def predict(self, inputs: dict) -> dict:
        outputs = self.session.run(None, inputs)
        return {
            output.name: value
            for output, value in zip(self.session.get_outputs(), outputs)
        }
```

---

## 80.8 Feature Engineering

| Feature | Type | Description |
|---------|------|-------------|
| `user_activity_count` | numeric | Total interactions |
| `avg_session_duration` | numeric | Average time spent |
| `topic_completion_rate` | numeric | Completed / total topics |
| `difficulty_preference` | categorical | Preferred difficulty |
| `time_since_last_activity` | numeric | Recency signal |
| `content_text_embedding` | vector[768] | BERT embedding |
| `content_difficulty` | categorical | Labeled difficulty |

---

## 80.9 Evaluation Metrics

| Model Type | Primary Metric | Secondary |
|------------|----------------|-----------|
| Recommendation | NDCG@10 | HR@10, MRR |
| Classification | F1-macro | Accuracy, AUC |
| Regression | RMSE | MAE, R² |
| Sequence | Perplexity | BLEU (for paths) |

---

## 80.10 Technology Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Deep learning framework |
| scikit-learn | Classical ML algorithms |
| MLflow | Experiment tracking |
| ONNX Runtime | Model serving |
| Ray | Distributed training (future) |
