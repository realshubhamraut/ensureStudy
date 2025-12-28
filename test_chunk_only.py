#!/usr/bin/env python3
"""Test JUST chunking to find the slow point"""
import sys
import time
sys.path.insert(0, '/Users/proxim/projects/ensureStudy/backend/ai-service')

# Test 15K chars
test_text = "Photosynthesis is the process by which plants convert light energy into chemical energy. " * 200
test_text = test_text[:15000]

print(f"[TEST] Text length: {len(test_text)} chars")

from app.services.chunking_service import chunk_for_qdrant

print("[TEST] Calling chunk_for_qdrant...")
start = time.time()
try:
    chunks = chunk_for_qdrant(
        text=test_text,
        source_url="https://test.com",
        source_type="test",
        source_trust=0.9
    )
    elapsed = time.time() - start
    print(f"[TEST] ✅ Created {len(chunks)} chunks in {elapsed:.2f}s")
except Exception as e:
    print(f"[TEST] ❌ Error: {e}")
    import traceback
    traceback.print_exc()
