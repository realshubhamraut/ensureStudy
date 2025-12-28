#!/usr/bin/env python3
"""
Test cache-first API end-to-end
"""
import requests
import time

API_URL = "http://localhost:8001/api/ai-tutor/query"

def test_query(question, label):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"Question: {question}")
    print('='*60)
    
    start = time.time()
    
    try:
        response = requests.post(API_URL, json={
            "user_id": "test",
            "question": question,
            "find_resources": True
        }, timeout=120)
        
        elapsed = time.time() - start
        data = response.json()
        
        if data.get("success") and data.get("data"):
            answer = data["data"].get("answer_short", "")[:100]
            web = data["data"].get("web_resources", {})
            articles = web.get("articles", [])
            
            if articles:
                source = articles[0].get("source", "Unknown")
                cached = source == "Cache"
            else:
                cached = False
                source = "None"
            
            print(f"✅ SUCCESS in {elapsed:.1f}s")
            print(f"   Source: {source} {'(CACHED!)' if cached else '(Fresh crawl)'}")
            print(f"   Answer: {answer}...")
            return cached
        else:
            print(f"❌ FAILED: {data.get('error')}")
            return None
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

# Run tests
print("\n" + "="*60)
print("🧪 CACHE-FIRST API VERIFICATION")
print("="*60)

# Test 1: New query (should crawl)
test1 = test_query("What is the theory of relativity?", "NEW QUERY (should crawl)")

# Test 2: Same query (should cache hit)
test2 = test_query("What is the theory of relativity?", "IDENTICAL QUERY (should cache)")

# Test 3: Similar query (should cache hit via semantic match)
test3 = test_query("Explain Einstein's relativity theory", "SIMILAR QUERY (should cache)")

# Summary
print("\n" + "="*60)
print("📊 TEST RESULTS")
print("="*60)
print(f"Test 1 (New):      {'Crawled' if not test1 else 'UNEXPECTED CACHE'}")
print(f"Test 2 (Same):     {'✅ CACHED' if test2 else '❌ MISSED'}")
print(f"Test 3 (Similar):  {'✅ CACHED' if test3 else '❌ MISSED (may be expected)'}")
