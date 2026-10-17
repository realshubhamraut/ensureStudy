#!/usr/bin/env python3
"""
Test Topic Extraction and Chaining
"""
import sys
sys.path.insert(0, '/Users/proxim/projects/ensureStudy/backend/ai-service')

from app.services.web_ingest_service import worker1_extract_topic

def test_chaining():
    print("="*50)
    print("🧪 TEST: Topic Chaining")
    print("="*50)
    
    # Scene 1: Initial Query
    query1 = "Summarize the causes of French Revolution"
    topic1 = worker1_extract_topic(query1)
    print(f"\nQuery 1: '{query1}'")
    print(f"Extracted Topic: '{topic1}'")
    
    # Scene 2: Follow-up Query
    query2 = "What were the main causes?"
    
    # Mock history
    history = [
        {"role": "user", "content": query1},
        {"role": "assistant", "content": "The French Revolution had many causes..."}
    ]
    
    topic2 = worker1_extract_topic(query2, conversation_history=history)
    print(f"\nQuery 2: '{query2}'")
    print(f"Extracted Topic (with history): '{topic2}'")
    
    if "french revolution" in topic2.lower():
        print("✅ SUCCESS: Topic correctly chained")
    else:
        print(f"❌ FAIL: Topic chaining failed. Got: {topic2}")

if __name__ == "__main__":
    test_chaining()
