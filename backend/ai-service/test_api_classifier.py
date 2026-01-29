"""
Test subject classifier using HuggingFace API
"""
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

print("=" * 70)
print("TESTING SUBJECT CLASSIFIER (API MODE)")
print("=" * 70)

# Check if API key is set
api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
if not api_key:
    print("\n⚠️  WARNING: No HuggingFace API key found!")
    print("Set one of these environment variables:")
    print("  - HUGGINGFACE_API_KEY")
    print("  - HUGGINGFACE_TOKEN")
    print("  - HF_TOKEN")
    print("\nThe API may work with rate limiting, or fail.\n")
else:
    print(f"\n✅ API Key found: {api_key[:10]}...{api_key[-4:]}\n")

from app.services.subject_classifier import SubjectClassifier

classifier = SubjectClassifier()

test_queries = [
    "Tell me about photosynthesis in details",
    "What is Newton's third law of motion?",
    "How do I solve quadratic equations?",
]

print("Testing subject detection:\n")
for query in test_queries:
    print(f"Query: {query}")
    result = classifier.classify_subject(query)
    print(f"  → Subject: {result['display_name']}")
    print(f"  → Confidence: {result['confidence']*100:.1f}%")
    print(f"  → All scores: {result.get('all_scores', {})}")
    print()

print("=" * 70)
print("✅ TEST COMPLETED!")
print("=" * 70)
