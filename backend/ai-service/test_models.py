"""
Test script for subject classification and follow-up question generation
"""
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

print("=" * 70)
print("TESTING SUBJECT CLASSIFIER AND FOLLOW-UP GENERATOR")
print("=" * 70)

# Test 1: Subject Classification
print("\n" + "=" * 70)
print("TEST 1: SUBJECT CLASSIFICATION")
print("=" * 70)

from app.services.subject_classifier import SubjectClassifier

classifier = SubjectClassifier()

test_queries = [
    "Tell me about photosynthesis in details",
    "What is Newton's third law of motion?",
    "How do I solve quadratic equations?",
    "Who was Napoleon Bonaparte?",
    "Explain how binary search algorithm works",
    "What causes earthquakes?",
    "Analyze the themes in Romeo and Juliet",
    "What is supply and demand?",
    "How does the human brain process emotions?"
]

print("\nTesting subject detection:\n")
for query in test_queries:
    result = classifier.classify_subject(query)
    print(f"Query: {query[:50]}...")
    print(f"  → Subject: {result['display_name']} ({result['confidence']*100:.1f}% confidence)")
    print()

# Test 2: Follow-up Question Generation
print("\n" + "=" * 70)
print("TEST 2: FOLLOW-UP QUESTION GENERATION")
print("=" * 70)

from app.services.followup_generator import generate_follow_up_questions

test_cases = [
    {
        "question": "Tell me about photosynthesis in details",
        "answer": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It occurs in the chloroplasts of plant cells.",
        "subject": "biology"
    },
    {
        "question": "What is Newton's third law?",
        "answer": "Newton's third law states that for every action, there is an equal and opposite reaction. This means that forces always come in pairs.",
        "subject": "physics"
    },
    {
        "question": "How do I solve x^2 + 5x + 6 = 0?",
        "answer": "You can solve this quadratic equation by factoring: (x+2)(x+3) = 0, so x = -2 or x = -3. Alternatively, use the quadratic formula.",
        "subject": "mathematics"
    }
]

print("\nTesting follow-up generation with subject context:\n")
for case in test_cases:
    print(f"Question: {case['question']}")
    print(f"Subject: {case['subject']}")
    
    follow_ups = generate_follow_up_questions(
        question=case['question'],
        answer_short=case['answer'],
        topic=case['subject'],
        subject=case['subject']
    )
    
    print("Follow-up questions:")
    for i, fq in enumerate(follow_ups, 1):
        print(f"  {i}. {fq}")
    print()

print("=" * 70)
print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 70)
