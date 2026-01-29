"""
Test LOCAL classifier model
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.llm_provider import TextClassifier

print("=" * 70)
print("TESTING LOCAL CLASSIFIER MODEL")
print("=" * 70)

classifier = TextClassifier()

test_queries = [
    ("Tell me about photosynthesis", ["biology", "chemistry", "physics"], "biology"),
    ("What is Newton's third law of motion?", ["biology", "physics", "mathematics"], "physics"),
    ("Solve quadratic equation x^2 + 5x + 6", ["mathematics", "physics", "chemistry"], "mathematics"),
    ("two mirrors placed in parallel how many images", ["physics", "mathematics", "chemistry"], "physics"),
    ("Explain DNA replication", ["biology", "chemistry", "physics"], "biology"),
]

print("\n🧪 Running tests...\n")

correct = 0
total = len(test_queries)

for query, labels, expected in test_queries:
    result = classifier.classify(query, labels)
    predicted = max(result, key=result.get)
    confidence = result[predicted]
    
    status = "✅" if predicted == expected else "❌"
    correct += 1 if predicted == expected else 0
    
    print(f"{status} Query: {query[:50]}...")
    print(f"   Expected: {expected}")
    print(f"   Predicted: {predicted} ({confidence*100:.1f}%)")
    print(f"   All scores: {result}")
    print()

print("=" * 70)
print(f"ACCURACY: {correct}/{total} ({correct/total*100:.1f}%)")
print("=" * 70)
