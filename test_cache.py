#!/usr/bin/env python3
"""
Test Web Cache Service - Full cache-first workflow
"""
import sys
import asyncio
sys.path.insert(0, '/Users/proxim/projects/ensureStudy/backend/ai-service')

async def test_cache():
    print("\n" + "="*60)
    print("🧪 WEB CACHE SERVICE TEST")
    print("="*60)
    
    from app.services.web_cache_service import (
        search_cache,
        store_in_cache,
        get_cache_stats,
        clear_cache
    )
    
    # Test 1: Clear cache for fresh start
    print("\n[TEST 1] Clearing cache...")
    clear_cache()
    
    # Test 2: Check stats
    print("\n[TEST 2] Cache stats:")
    stats = get_cache_stats()
    print(f"  Points: {stats.get('points_count', 0)}")
    print(f"  Status: {stats.get('status', 'unknown')}")
    
    # Test 3: Search empty cache
    print("\n[TEST 3] Search empty cache...")
    hit = search_cache("What is photosynthesis?")
    print(f"  Cache hit: {hit is not None}")
    
    # Test 4: Store an entry
    print("\n[TEST 4] Storing entry...")
    store_in_cache(
        query="What is photosynthesis?",
        answer="Photosynthesis is the process by which plants convert sunlight into energy. They use chlorophyll in their leaves to capture light and convert CO2 and water into glucose and oxygen.",
        sources=["https://en.wikipedia.org/wiki/Photosynthesis"],
        confidence=0.92
    )
    
    # Test 5: Search should now hit
    print("\n[TEST 5] Search for cached query...")
    hit = search_cache("What is photosynthesis?")
    if hit:
        print(f"  ✅ CACHE HIT!")
        print(f"  Similarity: {hit.similarity:.3f}")
        print(f"  Answer: {hit.answer[:80]}...")
        print(f"  Sources: {hit.sources}")
    else:
        print("  ❌ Cache miss (unexpected)")
    
    # Test 6: Similar query should also hit
    print("\n[TEST 6] Search similar query...")
    hit = search_cache("Explain photosynthesis to me")  # Similar but different
    if hit:
        print(f"  ✅ CACHE HIT (similar query matched!)")
        print(f"  Similarity: {hit.similarity:.3f}")
    else:
        print("  ❌ Cache miss (expected at 0.85 threshold)")
    
    # Test 7: Different query should miss
    print("\n[TEST 7] Search different query...")
    hit = search_cache("What caused World War 2?")  # Unrelated
    if hit:
        print(f"  Similarity: {hit.similarity:.3f}")
        print("  ⚠ Unexpected cache hit")
    else:
        print("  ✅ Cache miss (correct - different topic)")
    
    # Test 8: Final stats
    print("\n[TEST 8] Final cache stats:")
    stats = get_cache_stats()
    print(f"  Points: {stats.get('points_count', 0)}")
    
    print("\n" + "="*60)
    print("✅ CACHE SERVICE TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_cache())
