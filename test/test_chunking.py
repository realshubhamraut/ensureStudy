#!/usr/bin/env python3
"""Test chunking performance"""
import sys
import time
sys.path.insert(0, '/Users/proxim/projects/ensureStudy/backend/ai-service')

# Create large test content
test_text = "Photosynthesis is the process by which plants convert light energy. " * 1000  # ~70K chars

print(f"[TEST] Text length: {len(test_text)} chars")
print(f"[TEST] Starting chunking...")

start = time.time()

from app.services.chunking_service import chunk_for_qdrant

try:
    chunks = chunk_for_qdrant(
        text=test_text,
        source_url="https://test.com",
        source_type="test",
        source_trust=0.9
    )
    elapsed = time.time() - start
    print(f"[TEST] Created {len(chunks)} chunks in {elapsed:.2f}s")
except Exception as e:
    print(f"[TEST] Error: {e}")
    import traceback
    traceback.print_exc()
