#!/usr/bin/env python3
"""
Direct test of web crawler workers
"""
import sys
import asyncio
sys.path.insert(0, '/Users/proxim/projects/ensureStudy/backend/ai-service')

from app.services.web_ingest_service import (
    worker1_extract_topic,
    worker2_duckduckgo_search,
    worker3_wikipedia_search,
    worker4_wikipedia_content,
    worker5_parallel_crawl
)

async def test_workers():
    print("\n" + "="*70)
    print("🧪 TESTING MULTI-WORKER CRAWLER")
    print("="*70)
    
    query = "photosynthesis"
    
    # Worker-1
    topic = worker1_extract_topic(query)
    
    # Worker-2
    ddg_results = worker2_duckduckgo_search(topic, max_results=2)
    
    # Worker-3
    wiki_search = await worker3_wikipedia_search(topic)
    
    # Worker-4
    if wiki_search:
        wiki_content = await worker4_wikipedia_content(wiki_search['canonical_title'])
        print(f"\n[TEST] Wiki content keys: {wiki_content.keys() if wiki_content else 'None'}")
    
    # Collect URLs
    urls = []
    if wiki_search:
        urls.append(wiki_search['url'])
    urls.extend([r.get('href') for r in ddg_results if r.get('href')])
    urls = list(dict.fromkeys(urls))[:2]
    
    print(f"\n[TEST] URLs to fetch: {urls}")
    
    # Worker-5
    if urls:
        fetched = await worker5_parallel_crawl(urls)
        print(f"\n[TEST] Fetched {len(fetched)} pages")
        for i, page in enumerate(fetched):
            print(f"  Page {i+1}: {page['url'][:50]}")
            print(f"    Status: {page['status_code']}")
            print(f"    Content length: {len(page['content']) if page['content'] else 0}")
            print(f"    Content preview: {page['content'][:100] if page['content'] else 'None'}...")
    
    print("\n" + "="*70)
    print("✅ WORKER TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(test_workers())
