#!/usr/bin/env python3
"""
COMPREHENSIVE END-TO-END TEST OF WEB CRAWLER PIPELINE
Tests all 7 workers with content truncation fix
"""
import sys
import asyncio
import time
sys.path.insert(0, '/Users/proxim/projects/ensureStudy/backend/ai-service')

# Max content for performance
MAX_CONTENT = 15000

async def test_full_pipeline():
    print("\n" + "="*70)
    print("🧪 FULL PIPELINE TEST")
    print("="*70)
    
    query = "photosynthesis"
    
    # Import workers
    from app.services.web_ingest_service import (
        worker1_extract_topic,
        worker2_duckduckgo_search,
        worker3_wikipedia_search,
        worker4_wikipedia_content,
        worker5_parallel_crawl,
        generate_embeddings,
        store_in_qdrant,
        calculate_trust_score
    )
    from app.services.content_normalizer import normalize_content
    from app.services.chunking_service import chunk_for_qdrant
    
    start_time = time.time()
    
    # WORKER-1
    print("\n[W1] Running Worker-1...")
    topic = worker1_extract_topic(query)
    
    # WORKER-2  
    print("\n[W2] Running Worker-2...")
    ddg_results = worker2_duckduckgo_search(topic, max_results=1)
    
    # WORKER-3
    print("\n[W3] Running Worker-3...")
    wiki_search = await worker3_wikipedia_search(topic)
    
    # WORKER-4
    wiki_content = None
    if wiki_search:
        print("\n[W4] Running Worker-4...")
        wiki_content = await worker4_wikipedia_content(wiki_search['canonical_title'])
    
    # Collect URLs
    urls = []
    if wiki_search:
        urls.append(wiki_search['url'])
    urls = urls[:1]  # Just 1 URL
    
    # WORKER-5
    print("\n[W5] Running Worker-5...")
    fetched_pages = await worker5_parallel_crawl(urls)
    print(f"[W5] Got {len(fetched_pages)} pages")
    
    # WORKER-6: Content Processing
    print("\n[W6] Running Worker-6...")
    all_chunks = []
    total_tokens = 0
    
    for i, page in enumerate(fetched_pages):
        print(f"[W6] Processing page {i+1}...")
        url = page['url']
        html_content = page['content']
        
        try:
            # Extract
            import trafilatura
            clean_text = trafilatura.extract(html_content)
            print(f"[W6] Extracted: {len(clean_text) if clean_text else 0} chars")
            
            if not clean_text or len(clean_text) < 100:
                print(f"[W6] ⚠ Insufficient content")
                continue
                
            # TRUNCATE for performance
            if len(clean_text) > MAX_CONTENT:
                clean_text = clean_text[:MAX_CONTENT]
                last_period = clean_text.rfind('.')
                if last_period > MAX_CONTENT * 0.8:
                    clean_text = clean_text[:last_period + 1]
                print(f"[W6] Truncated to: {len(clean_text)} chars")
            
            token_count = len(clean_text.split())
            total_tokens += token_count
            
            # Normalize
            print("[W6] Normalizing...")
            normalized = normalize_content(clean_text, url.split('/')[-1], 'text')
            print(f"[W6] Normalized: {len(normalized.text)} chars")
            
            # Chunk
            print("[W6] Chunking...")
            chunks = chunk_for_qdrant(
                text=normalized.text,
                source_url=url,
                source_type='webpage',
                source_trust=calculate_trust_score(url)
            )
            print(f"[W6] Created {len(chunks)} chunks")
            
            all_chunks.extend(chunks)
            print(f"[W6] ✅ Page {i+1} processed!")
            
        except Exception as e:
            print(f"[W6] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n[W6] Complete: {total_tokens} tokens, {len(all_chunks)} chunks")
    
    # WORKER-7: Embedding + Storage
    print("\n[W7] Running Worker-7...")
    if all_chunks:
        texts = [chunk['text'] for chunk in all_chunks]
        print(f"[W7] Generating embeddings for {len(texts)} chunks...")
        embeddings = generate_embeddings(texts)
        print(f"[W7] Generated {len(embeddings) if embeddings else 0} embeddings")
        
        if embeddings:
            print("[W7] Storing in Qdrant...")
            total_stored = store_in_qdrant(all_chunks, embeddings)
            print(f"[W7] ✅ Stored {total_stored} chunks")
    else:
        print("[W7] ⚠ No chunks to embed")
    
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print(f"✅ FULL PIPELINE TEST COMPLETE in {elapsed:.1f}s")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
