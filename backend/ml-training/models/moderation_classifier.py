"""
PyTorch Moderation Classifier Model
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ModerationClassifier(nn.Module):
    """
    Binary classifier for academic vs off-topic content.
    
    Uses DistilBERT as base with custom classification head.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
        
        self.num_labels = num_labels
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Tokenized input [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Logits [batch_size, num_labels]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        
        logits = self.classifier(pooled)
        return logits
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> tuple:
        """
        Get predictions with probabilities.
        
        Returns:
            (predicted_labels, probabilities)
        """
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            labels = torch.argmax(probs, dim=1)
        
        return labels, probs


class ModerationPredictor:
    """Wrapper for easy inference"""
    
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModerationClassifier()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        self.labels = {0: "academic", 1: "off_topic"}
    
    def predict(self, text: str) -> dict:
        """
        Predict if text is academic or off-topic.
        
        Args:
            text: Input text
        
        Returns:
            {"label": str, "confidence": float}
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        labels, probs = self.model.predict(input_ids, attention_mask)
        
        label_id = labels[0].item()
        confidence = probs[0][label_id].item()
        
        return {
            "label": self.labels[label_id],
            "confidence": confidence
        }
