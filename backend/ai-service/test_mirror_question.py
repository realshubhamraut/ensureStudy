"""
Test the keyword classifier with the mirror question
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.subject_classifier import SubjectClassifier

print("=" * 70)
print("TESTING KEYWORD CLASSIFIER WITH MIRROR QUESTION")
print("=" * 70)

classifier = SubjectClassifier()

# The actual question from the logs
question = "if two mirrors are placed in parallel then how many number of images are formed"

print(f"\nQuestion: {question}\n")

result = classifier.classify_subject(question, min_confidence=0.3)

print(f"Detected Subject: {result['display_name']}")
print(f"Confidence: {result['confidence']*100:.1f}%")
print(f"\nAll scores:")
for subject, score in sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {subject}: {score*100:.1f}%")

print("\n" + "=" * 70)
if result['subject'] == 'physics':
    print("✅ CORRECT: Detected as Physics!")
else:
    print(f"❌ WRONG: Detected as {result['display_name']}, should be Physics")
print("=" * 70)
