#!/usr/bin/env python3
"""
Test the Groq-powered classification (70B model via API)
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'ai-service'))

from app.services.llm_provider import get_classifier

# Test cases: question -> expected subject
TEST_CASES = [
    ("Tell me about photosynthesis and how plants make food", "biology"),
    ("What is Newton's third law of motion?", "physics"),
    ("Solve the quadratic equation x^2 + 5x + 6 = 0", "mathematics"),
    ("if two mirrors are placed in parallel then how many number of images are formed", "physics"),
    ("Explain DNA replication process in detail", "biology"),
    ("tell me about the right hand thumb rule", "physics"),  # User's current example
    ("What is the difference between acids and bases?", "chemistry"),
]

LABELS = ["mathematics", "physics", "chemistry", "biology"]

def test_classification():
    print("\n" + "="*70)
    print("TESTING GROQ-POWERED CLASSIFIER (Llama 3.3 70B)")
    print("="*70)
    
    classifier = get_classifier()
    
    correct = 0
    total = len(TEST_CASES)
    
    for question, expected in TEST_CASES:
        print(f"\n{'─'*70}")
        print(f"Question: {question[:60]}...")
        print(f"Expected: {expected}")
        
        try:
            scores = classifier.classify(question, LABELS)
            
            # Get top prediction
            predicted = max(scores.items(), key=lambda x: x[1])
            predicted_label, confidence = predicted
            
            is_correct = predicted_label == expected
            if is_correct:
                correct += 1
                icon = "✅"
            else:
                icon = "❌"
            
            print(f"{icon} Predicted: {predicted_label} ({confidence*100:.1f}%)")
            print(f"   All scores: {', '.join(f'{k}={v*100:.0f}%' for k,v in sorted(scores.items(), key=lambda x: -x[1])[:3])}")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    accuracy = (correct / total) * 100
    print(f"ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
    print("="*70 + "\n")
    
    return accuracy

if __name__ == "__main__":
    accuracy = test_classification()
    
    # Exit code based on accuracy
    if accuracy >= 80:
        print("✅ EXCELLENT: Classification accuracy >= 80%")
        sys.exit(0)
    elif accuracy >= 60:
        print("⚠️ GOOD: Classification accuracy >= 60%")
        sys.exit(0)
    else:
        print("❌ POOR: Classification accuracy < 60%")
        sys.exit(1)
