#!/usr/bin/env python3
"""
Test Subject Classifier

Quick test to verify subject detection works correctly.
"""
import sys
import os

# Add project to path
sys.path.insert(0, '/Users/proxim/projects/ensureStudy/backend/ai-service')

def test_subject_classification():
    """Test subject detection with various queries"""
    from app.services.subject_classifier import get_subject_classifier
    
    classifier = get_subject_classifier()
    
    test_queries = [
        "tell me about photosynthesis in details",
        "Explain Newton's laws of motion",
        "What is the quadratic formula?",
        "Solve x^2 + 5x + 6 = 0",
        "What caused World War 1?",
        "Explain how a computer works",
        "What is supply and demand?",
        "Write a Python function to reverse a string",
    ]
    
    print("\n" + "="*70)
    print("🧪 TESTING SUBJECT CLASSIFIER")
    print("="*70 + "\n")
    
    for query in test_queries:
        result = classifier.classify_subject(query)
        print(f"Query: \"{query}\"")
        print(f"  ✓ Subject: {result['display_name']}")
        print(f"  ✓ Confidence: {result['confidence']:.2f}")
        print(f"  ✓ Key: {result['subject']}")
        print()
    
    print("="*70)
    print("✅ Subject classification test complete!")
    print("="*70)


if __name__ == "__main__":
    test_subject_classification()
