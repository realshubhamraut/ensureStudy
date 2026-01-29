"""
Simple test for bart-large-mnli model
"""
from transformers import pipeline
import torch

print("Testing facebook/bart-large-mnli...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Create classifier with explicit device
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1  # Force CPU
)

# Simple test
text = "Tell me about photosynthesis"
labels = ["biology", "physics", "mathematics"]

print(f"\nClassifying: '{text}'")
result = classifier(text, candidate_labels=labels)

print("\nResults:")
for label, score in zip(result["labels"], result["scores"]):
    print(f"  {label}: {score:.3f}")

print("\n✅ Test completed successfully!")
