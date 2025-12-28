#!/usr/bin/env python3
"""
Test Qdrant connection - Fixed for latest API
"""
import sys
sys.path.insert(0, '/Users/proxim/projects/ensureStudy/backend/ai-service')

print("[TEST] Testing Qdrant connection...")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import uuid
    
    # In-memory mode (no Docker needed)
    print("[TEST] Creating in-memory Qdrant client...")
    client = QdrantClient(":memory:")
    
    collection_name = "web_content_cache"
    
    # Create collection
    print(f"[TEST] Creating collection: {collection_name}")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"[TEST] ✅ Collection created!")
    
    # Test insert
    print("[TEST] Testing insert...")
    test_vector = [0.1] * 384
    
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=test_vector,
                payload={
                    "query": "What is photosynthesis?",
                    "answer": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
                    "sources": ["https://en.wikipedia.org/wiki/Photosynthesis"],
                    "confidence": 0.95,
                    "timestamp": "2025-12-28T12:00:00"
                }
            )
        ]
    )
    print("[TEST] ✅ Insert successful!")
    
    # Test search (using query_points for new API)
    print("[TEST] Testing search...")
    results = client.query_points(
        collection_name=collection_name,
        query=test_vector,
        limit=1
    )
    
    if results.points:
        print(f"[TEST] ✅ Search returned {len(results.points)} results")
        best = results.points[0]
        print(f"[TEST] Score: {best.score}")
        print(f"[TEST] Query: {best.payload.get('query', 'N/A')}")
        print(f"[TEST] Answer: {best.payload.get('answer', 'N/A')[:50]}...")
    else:
        print("[TEST] ⚠ No results found")
    
    print("\n" + "="*50)
    print("✅ IN-MEMORY QDRANT FULLY WORKING!")
    print("="*50)
    
except Exception as e:
    print(f"[TEST] ❌ Error: {e}")
    import traceback
    traceback.print_exc()
