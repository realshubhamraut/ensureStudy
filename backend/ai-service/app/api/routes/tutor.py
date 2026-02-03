"""
AI Tutor Query Endpoint

POST /api/ai-tutor/query

8-Step Pipeline:
1. Receive Question
2. Academic Moderation (No LLM)
3. Embed Question (Sentence-Transformers)
4. Retrieve Context (Qdrant)
5. Apply Model Context Protocol
6. Construct Prompt
7. LLM Call (FLAN-T5)
8. Structured Output
"""
import os
import time
from fastapi import APIRouter, HTTPException

from ..schemas.tutor import (
    TutorQueryRequest,
    TutorQueryResponse,
    TutorResponseData,
    SourceInfo,
    ResponseMetadata
)
from ...services.moderation import moderate_query
from ...services.retrieval import semantic_search
from ...services.context import build_context
from ...services.reasoning import generate_answer
from ...services.recommendations import generate_recommendations
from ...services.web_resources import fetch_all_web_resources, web_resources_to_dict
from ...services.flowchart_generator import generate_concept_flowchart
from ...services.response_cache import get_response_cache, generate_context_hash
from ...services.image_service import search_images_brave
from ...services.youtube_video_service import search_videos_youtube
from ...services.followup_generator import generate_follow_up_questions
from ...utils.logging import (
    generate_request_id,
    log_query_received,
    log_moderation_result,
    log_retrieval_result_full,
    log_query_processed,
    log_error
)

router = APIRouter(prefix="/api/ai-tutor", tags=["AI Tutor"])

# Initialize cache
CACHE = get_response_cache()


@router.post("/query", response_model=TutorQueryResponse)
async def process_tutor_query(request: TutorQueryRequest) -> TutorQueryResponse:
    """
    Process student question using RAG pipeline.
    
    Uses FREE, LOCAL models:
    - Embeddings: sentence-transformers/all-mpnet-base-v2
    - LLM: google/flan-t5-base
    
    The pipeline:
    1. Moderation (no LLM, keyword-based)
    2. Embed question
    3. Retrieve from Qdrant
    4. Apply MCP (context control)
    5. Call FLAN-T5
    6. Format response
    """
    start_time = time.time()
    request_id = generate_request_id()
    retrieval_time = 0
    llm_time = 0
    
    # ========================================
    # Step 1: Log incoming request
    # ========================================
    log_query_received(
        request_id=request_id,
        user_id=request.user_id,
        question=request.question,
        subject=request.subject.value if request.subject else None
    )
    
    try:
        # ========================================
        # Step 1.5: Auto-detect subject if not provided
        # ========================================
        detected_subject = None
        subject_confidence = 0.0
        detected_subjects = []  # List of subjects for multi-matching
        
        if not request.subject:
            from ...services.subject_classifier import get_subject_classifier
            subject_classifier = get_subject_classifier()
            
            # Use LLM-based multi-subject detection (industry standard)
            multi_result = subject_classifier.classify_subject_multi(request.question)
            detected_subjects = multi_result["subjects"]
            detected_subject = multi_result["primary"]  # Most specific
            subject_confidence = multi_result["confidences"][0] if multi_result["confidences"] else 0.0
            
            print(f"[SUBJECT] 🎯 Multi-detected: {' → '.join(multi_result['display_names'])} "
                  f"(primary: {detected_subject}, confidence: {subject_confidence:.2f})")
            
            # Use primary subject if confidence is high enough
            if subject_confidence >= 0.3:
                from ...api.schemas.tutor import Subject
                try:
                    request.subject = Subject(detected_subject)
                except:
                    # If not a valid enum, keep as None
                    pass
        
        # ========================================
        # Step 1.6: Match classroom by subject (try multiple)
        # ========================================
        matched_classroom_id = request.classroom_id  # Use existing if provided
        matched_subject_name = None  # Track which subject matched
        
        if not matched_classroom_id and detected_subjects and request.user_id:
            try:
                from ...services.classroom_matcher import match_classroom_by_subject
                # Try each subject in order (most specific first)
                for subj in detected_subjects:
                    matched = await match_classroom_by_subject(request.user_id, subj)
                    if matched:
                        matched_classroom_id = matched["classroom_id"]
                        matched_subject_name = subj  # Store which subject matched
                        print(f"[CLASSROOM] 📚 Matched '{subj}' → '{matched['name']}' (ID: {matched_classroom_id[:8]}...)")
                        break
                else:
                    print(f"[CLASSROOM] ℹ️ No classroom match for subjects: {detected_subjects}")
            except Exception as e:
                print(f"[CLASSROOM] ⚠️ Classroom matching failed: {e}")
        
        # ========================================
        # Step 2: Academic Moderation (No LLM)
        # ========================================
        moderation_result = moderate_query(
            user_id=request.user_id,
            question=request.question
        )
        
        log_moderation_result(
            request_id=request_id,
            user_id=request.user_id,
            decision=moderation_result.decision,
            confidence=moderation_result.confidence,
            category=moderation_result.category
        )
        
        if moderation_result.decision == "block":
            log_error(
                "non_academic_query",
                moderation_result.reason or "Query blocked",
                request_id
            )
            return TutorQueryResponse(
                success=False,
                error={
                    "code": "non_academic_query",
                    "message": moderation_result.reason or "Please ask academic questions only."
                }
            )
        
        # ========================================
        # Step 3 & 4: Embed + Retrieve (Qdrant)
        # ========================================
        retrieval_start = time.time()
        
        # Regular semantic search
        chunks = semantic_search(
            query=request.question,
            user_id=request.user_id,
            subject=request.subject.value if request.subject else None
        )
        
        # ALWAYS search classroom materials (even if no specific classroom selected)
        # When no classroom_id, search across ALL classrooms
        classroom_chunks = []
        transcript_chunks = []
        
        # Search classroom materials (PDFs, docs, etc.)
        # Use detected subject to filter relevant materials
        try:
            from ...services.material_indexer import get_material_indexer
            indexer = get_material_indexer()
            
            # Get the subject for filtering (from request or auto-detected)
            # Skip filtering for 'general' or 'general_knowledge' - these are too broad
            search_subject = None
            skip_subjects = {'general', 'general_knowledge'}
            
            if request.subject:
                search_subject = request.subject.value
            elif detected_subject and detected_subject.lower() not in skip_subjects:
                search_subject = detected_subject
            
            if search_subject:
                print(f"[TUTOR] 🎯 Searching materials with subject filter: {search_subject}")
            else:
                print(f"[TUTOR] 📚 Searching ALL materials (no subject filter)")
            
            classroom_results = indexer.search_classroom_materials(
                query=request.question,
                classroom_id=request.classroom_id,  # None = search all
                subject=search_subject,  # Filter by detected subject
                top_k=5,
                score_threshold=0.3
            )
            
            # FALLBACK: If subject filter returns 0 results, search without subject
            if len(classroom_results) == 0 and search_subject:
                print(f"[TUTOR] 🔄 No results with subject filter, retrying without subject...")
                classroom_results = indexer.search_classroom_materials(
                    query=request.question,
                    classroom_id=request.classroom_id,
                    subject=None,  # No subject filter
                    top_k=5,
                    score_threshold=0.3
                )
                print(f"[TUTOR] 📚 Fallback search found {len(classroom_results)} chunks")
            
            # Convert to same format as regular chunks
            from ...services.retrieval import RetrievedChunk
            for r in classroom_results:
                classroom_chunks.append(RetrievedChunk(
                    document_id=r["document_id"],
                    chunk_id=f"classroom_{r['document_id']}",
                    text=r["chunk_text"],
                    similarity_score=r["similarity_score"],
                    title=r.get("title", "Classroom Material"),
                    page_number=r.get("page_number", 0),
                    url=r.get("url", "")
                ))
            print(f"[TUTOR] 📚 Found {len(classroom_chunks)} classroom material chunks" + 
                  (f" from classroom {request.classroom_id}" if request.classroom_id else " from ALL classrooms") +
                  (f" (subject: {search_subject})" if search_subject else ""))
        except Exception as e:
            print(f"[TUTOR] ⚠ Classroom material search failed: {e}")
        
        # Search meeting transcripts if classroom_id is provided
        if request.classroom_id:
            try:
                from ...services.retrieval import search_meeting_transcripts
                transcript_results = search_meeting_transcripts(
                    query=request.question,
                    classroom_id=request.classroom_id,
                    top_k=5,
                    threshold=0.4
                )
                transcript_chunks = transcript_results
                print(f"[TUTOR] 🎤 Found {len(transcript_chunks)} meeting transcript chunks")
            except Exception as e:
                print(f"[TUTOR] ⚠ Meeting transcript search failed: {e}")
        
        # Merge and sort by score (classroom materials and transcripts get priority boost)
        all_classroom_content = classroom_chunks + transcript_chunks
        if all_classroom_content:
            for chunk in all_classroom_content:
                chunk.similarity_score = min(chunk.similarity_score + 0.1, 1.0)  # Boost
            chunks = sorted(chunks + all_classroom_content, key=lambda c: c.similarity_score, reverse=True)
        
        retrieval_time = int((time.time() - retrieval_start) * 1000)
        
        # ========================================
        # Step 4.5: Search web_content for prior crawled content
        # ========================================
        # This finds chunks from previous web crawls (shell scripting, etc.)
        web_chunks_context = ""
        try:
            from qdrant_client import QdrantClient
            qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            web_client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=10)
            
            # Check if collection exists
            collections = web_client.get_collections().collections
            if any(c.name == "web_content" for c in collections):
                # Get embedding for search
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                question_embedding = model.encode(request.question, normalize_embeddings=True).tolist()
                
                web_results = web_client.query_points(
                    collection_name="web_content",
                    query=question_embedding,
                    limit=5,
                    score_threshold=0.5,  # Only relevant chunks
                    with_payload=True
                )
                
                if web_results.points:
                    print(f"[RAG] 📚 Found {len(web_results.points)} prior web chunks, adding to context")
                    for point in web_results.points:
                        text = point.payload.get('text', '')[:800]  # Limit per chunk
                        source = point.payload.get('source_url', '')
                        if text:
                            web_chunks_context += f"\n\n--- Prior Knowledge ({source[:50]}...) ---\n{text}"
        except Exception as web_err:
            print(f"[RAG] ⚠ Web content search error: {web_err}")
        
        log_retrieval_result_full(
            request_id=request_id,
            sources_count=len(chunks),
            top_score=chunks[0].similarity_score if chunks else 0.0,
            retrieval_time_ms=retrieval_time
        )
        
        # NOTE: Don't return error here if no chunks - we'll try web search if find_resources=True
        if not chunks and not request.find_resources:
            # Only fail if NOT doing web search
            return TutorQueryResponse(
                success=False,
                error={
                    "code": "no_relevant_sources",
                    "message": "No relevant study materials found. Try enabling 'Find Resources' to search the web."
                }
            )
        
        # ========================================
        # Step 5: Apply Model Context Protocol
        # ========================================
        context = build_context(
            retrieved_chunks=chunks,  # May be empty if we'll get web content
            response_mode=request.response_mode
        )
        
        # ========================================
        # Step 5.5: Web Context (HYBRID - Fast + Background PDF)
        # ========================================
        # OPTIMIZATION: Fast resources block, PDFs run in background
        # - Cache: instant
        # - Wikipedia: 2-3s (needed for context)
        # - Images/Videos: 2-3s (parallel)
        # - Full web crawl + PDFs: background task (don't wait!)
        import asyncio
        web_context = ""
        web_resources_dict = None
        cache_hit = False
        background_crawl_task = None
        
        if request.find_resources:
            try:
                from ...services.web_cache_service import search_cache, store_in_cache
                from ...services.web_ingest_service import (
                    ingest_web_resources,
                    worker1_extract_topic, 
                    worker3_wikipedia_search, 
                    worker4_wikipedia_content,
                    extract_source_name,
                    calculate_trust_score
                )
                
                # CACHE-FIRST: Check cache for similar query with matching style
                # Include language_style so "explain in detail" vs "explain simply" use different caches
                lang_style = request.language_style.value if request.language_style else "layman"
                resp_mode = request.response_mode.value if request.response_mode else "short"
                cached = search_cache(
                    request.question, 
                    threshold=0.85,
                    language_style=lang_style,
                    response_mode=resp_mode
                )
                
                if cached:
                    # CACHE HIT - Use cached content (instant!)
                    print(f"[RAG] ✅ CACHE HIT! Similarity: {cached.similarity:.3f}")
                    cache_hit = True
                    web_context = f"\n\n--- Cached Web Knowledge (similarity: {cached.similarity:.2f}) ---\n{cached.answer[:2000]}"
                    web_resources_dict = {
                        "articles": [{
                            "id": "cached_1",
                            "type": "article",
                            "title": "Cached Result",
                            "url": cached.sources[0] if cached.sources else "",
                            "source": "Cache",
                            "snippet": cached.answer[:200],
                            "trustScore": cached.confidence,
                            "relevance": int(cached.confidence * 100) if cached.confidence else 85
                        }]
                    }
                    print(f"[RAG] ⚡ Using cache - skipping web crawl!")
                else:
                    # CACHE MISS - Run fast Wikipedia + start background PDF crawl
                    print(f"[RAG] 🌐 Cache miss, starting hybrid fetch...")
                    
                    # Convert conversation_history to dicts
                    history_dicts = None
                    if request.conversation_history:
                        history_dicts = [{"role": m.role, "content": m.content} for m in request.conversation_history]
                    
                    # WORKER-1: Topic extraction (fast)
                    topic = worker1_extract_topic(request.question, history_dicts)
                    print(f"[RAG-FAST] Topic: {topic}")
                    
                    # START BACKGROUND TASK: Full web crawl with PDFs (don't await!)
                    # This will store resources in Qdrant for future queries
                    # AND push real-time updates via SSE
                    async def background_crawl():
                        try:
                            from .sse import push_loading_status, push_pdf_update, push_complete, create_stream
                            
                            # Create SSE stream for this request
                            create_stream(request_id)
                            await push_loading_status(request_id, "Searching for PDFs...", 10)
                            
                            print(f"[BACKGROUND] 📥 Starting full web crawl with PDFs...")
                            result = await ingest_web_resources(
                                query=request.question,
                                subject=request.subject.value if request.subject else None,
                                max_sources=3,
                                conversation_history=history_dicts,
                                search_pdfs=True,  # PDFs enabled!
                                user_id=request.user_id,
                                classroom_id=matched_classroom_id
                            )
                            
                            if result.success:
                                # Push each PDF as it's discovered
                                pdf_count = 0
                                for resource in result.resources:
                                    if resource.url and resource.url.lower().endswith('.pdf'):
                                        pdf_count += 1
                                        await push_pdf_update(request_id, {
                                            "id": f"pdf_{pdf_count}",
                                            "title": resource.title or f"PDF Document {pdf_count}",
                                            "url": resource.url,
                                            "source": resource.source_name or "Web",
                                            "snippet": (resource.clean_content or "")[:200],
                                            "relevance": 85
                                        })
                                
                                # Cache the result for future queries
                                sources_list = [r.url for r in result.resources if r.url]
                                combined_answer = "\n\n".join([r.clean_content[:2000] for r in result.resources if r.clean_content])
                                if combined_answer:
                                    store_in_cache(
                                        query=request.question,
                                        answer=combined_answer,
                                        sources=sources_list,
                                        confidence=0.9,
                                        language_style=lang_style,
                                        response_mode=resp_mode
                                    )
                                
                                # Signal completion
                                await push_complete(request_id, pdf_count)
                                print(f"[BACKGROUND] ✅ Crawl complete: {len(result.resources)} sources, {pdf_count} PDFs, cached!")
                            else:
                                await push_complete(request_id, 0)
                                print(f"[BACKGROUND] ⚠ Crawl failed: {result.error}")
                        except Exception as e:
                            print(f"[BACKGROUND] ❌ Crawl error: {e}")
                            try:
                                from .sse import push_complete
                                await push_complete(request_id, 0)
                            except:
                                pass
                    
                    # Fire and forget - don't await, runs in background
                    background_crawl_task = asyncio.create_task(background_crawl())
                    
                    # FAST PATH: Wikipedia + Serper articles (2-3s each, parallel)
                    wiki_search = await worker3_wikipedia_search(topic)
                    articles_list = []
                    
                    # Add Wikipedia article
                    if wiki_search:
                        wiki_data = await worker4_wikipedia_content(wiki_search['canonical_title'])
                        if wiki_data and wiki_data.get('extract'):
                            wiki_extract = wiki_data['extract'][:2000]
                            web_context = f"\n\n--- Wikipedia ---\n{wiki_extract}"
                            print(f"[RAG-FAST] ✅ Wikipedia fetched: {len(wiki_extract)} chars")
                            
                            articles_list.append({
                                "id": "wiki_1",
                                "type": "article",
                                "title": wiki_data.get('title', topic),
                                "url": wiki_data.get('url', f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"),
                                "source": "Wikipedia",
                                "snippet": wiki_extract[:200],
                                "trustScore": 0.95,
                                "relevance": 95
                            })
                    
                    # ALSO fetch Serper articles for educational sources (Khan Academy, Byju's, etc.)
                    try:
                        from ...services.search_api import SerperSearchClient
                        serper = SerperSearchClient()
                        serper_results = await serper.search(topic, num_results=5)
                        
                        if serper_results:
                            print(f"[RAG-FAST] 🔍 Serper found {len(serper_results)} educational sources")
                            for i, result in enumerate(serper_results[:4]):  # Top 4 results
                                # Skip if it's a PDF or YouTube (handled separately)
                                if result.url.endswith('.pdf') or 'youtube.com' in result.url:
                                    continue
                                articles_list.append({
                                    "id": f"article_{i+2}",
                                    "type": "article",
                                    "title": result.title,
                                    "url": result.url,
                                    "source": result.domain,
                                    "snippet": result.snippet[:200] if result.snippet else "",
                                    "trustScore": result.trust_score,
                                    "relevance": int(result.trust_score * 100)
                                })
                                # ALSO add snippet to web_context for LLM!
                                if result.snippet:
                                    web_context += f"\n\n--- {result.title} ({result.domain}) ---\n{result.snippet}"
                            print(f"[RAG-FAST] ✅ Added {len(articles_list)} articles (Wikipedia + Serper) to context")
                    except Exception as serper_err:
                        print(f"[RAG-FAST] ⚠ Serper search error: {serper_err}")
                    
                    # Build resources dict
                    if articles_list:
                        web_resources_dict = {"articles": articles_list}
                    
                    print(f"[RAG-FAST] ⚡ Fast path done, PDFs crawling in background...")
                    
            except Exception as e:
                print(f"[RAG] ⚠ Web fetch error: {e}")
        
        # ========================================
        # Fetch images and videos (fast, parallel)
        # ========================================
        try:
            # Fetch images from DuckDuckGo (fast)
            brave_images = await search_images_brave(request.question, count=3)
            if brave_images:
                if web_resources_dict is None:
                    web_resources_dict = {}
                web_resources_dict["images"] = brave_images
                print(f"[RAG] 🖼️ Added {len(brave_images)} images")
        except Exception as img_err:
            print(f"[RAG] ⚠ Image fetch error: {img_err}")
        
        try:
            # Fetch YouTube videos using YouTube Data API v3
            from ...services.youtube_video_service import search_videos_youtube
            youtube_videos = await search_videos_youtube(request.question, max_results=3)
            if youtube_videos:
                if web_resources_dict is None:
                    web_resources_dict = {}
                web_resources_dict["videos"] = [
                    {
                        "id": v.get("id"),
                        "type": "video",
                        "title": v.get("title"),
                        "url": v.get("url"),
                        "thumbnailUrl": v.get("thumbnailUrl"),
                        "embedUrl": v.get("embedUrl"),
                        "duration": v.get("duration"),
                        "source": v.get("source", "YouTube"),
                        "relevance": v.get("relevance", 90)
                    }
                    for v in youtube_videos
                ]
                print(f"[RAG] 🎬 Added {len(youtube_videos)} YouTube videos")
        except Exception as vid_err:
            print(f"[RAG] ⚠ YouTube video error: {vid_err}")
        
        # ========================================
        # Fetch classroom PDFs/web-materials
        # ========================================
        # Query the classroom for PDFs stored from previous queries
        if matched_classroom_id and request.auth_token:
            try:
                import httpx
                # Use the /materials endpoint with source=web filter
                core_api = os.getenv("CORE_API_URL", "http://localhost:8000")
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(
                        f"{core_api}/api/classroom/{matched_classroom_id}/materials",
                        params={"source": "web"},  # Filter to web-crawled materials only
                        headers={"Authorization": f"Bearer {request.auth_token}"}
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        materials = data.get("materials", []) if isinstance(data, dict) else data
                        pdfs = []
                        for i, mat in enumerate(materials[:5]):  # Limit to 5 PDFs
                            # Check if it's a PDF
                            file_name = mat.get('name', '') or mat.get('file_name', '')
                            file_url = mat.get('file_url', '')
                            is_pdf = (
                                file_name.lower().endswith('.pdf') or 
                                'pdf' in mat.get('file_type', '').lower() or
                                file_url.lower().endswith('.pdf')
                            )
                            if is_pdf:
                                pdf_entry = {
                                    "id": mat.get('id') or f"classroom_pdf_{i}",
                                    "type": "pdf",
                                    "title": mat.get('name') or file_name or 'PDF Document',
                                    "url": file_url,
                                    "source": mat.get('source', 'Web'),
                                    "snippet": (mat.get('description', '') or '')[:200],
                                    "relevance": 85,
                                    "pages": mat.get('page_count', 0)
                                }
                                pdfs.append(pdf_entry)
                        
                        if pdfs:
                            if web_resources_dict is None:
                                web_resources_dict = {}
                            web_resources_dict["pdfs"] = pdfs
                            print(f"[RAG] 📄 Added {len(pdfs)} classroom PDFs to resources")
                    else:
                        print(f"[RAG] ⚠ Materials API returned {resp.status_code}")
            except Exception as pdf_err:
                print(f"[RAG] ⚠ Classroom PDF fetch error: {pdf_err}")
        
        # Combine Qdrant context + Web context + Prior web chunks
        full_context = context.context_text
        if web_chunks_context:
            full_context = full_context + web_chunks_context
        if web_context:
            full_context = full_context + web_context
        
        # ========================================
        # Step 6 & 7: Prompt + LLM (Groq → HuggingFace fallback)
        # ========================================
        # Check LLM response cache first
        context_hash = generate_context_hash(full_context)
        subject_str = request.subject.value if request.subject else "General"
        
        cached_response = CACHE.get_llm_response(
            question=request.question,
            context_hash=context_hash,
            subject=subject_str
        )
        
        if cached_response:
            # CACHE HIT - instant response!
            print(f"[RAG] ⚡ LLM CACHE HIT - returning cached response")
            llm_response = type('LLMResponse', (), {
                'answer_short': cached_response.answer_short,
                'answer_detailed': cached_response.answer_detailed,
                'confidence': cached_response.confidence,
                'reasoning': cached_response.reasoning,
                'suggested_topics': cached_response.suggested_topics,
                'raw_response': '',
                'generation_time_ms': 0
            })()
            llm_time = 0
        else:
            # CACHE MISS - call LLM (Groq → HuggingFace fallback)
            llm_response = generate_answer(
                question=request.question,
                context=full_context,  # Now includes web content!
                subject=subject_str,
                response_mode=request.response_mode,
                language_style=request.language_style.value
            )
            llm_time = llm_response.generation_time_ms
            
            # Cache the response for future queries
            try:
                CACHE.set_llm_response(
                    question=request.question,
                    context_hash=context_hash,
                    subject=subject_str,
                    response={
                        'answer_short': llm_response.answer_short,
                        'answer_detailed': llm_response.answer_detailed,
                        'confidence': llm_response.confidence,
                        'reasoning': llm_response.reasoning,
                        'suggested_topics': llm_response.suggested_topics,
                        'generation_time_ms': llm_response.generation_time_ms
                    }
                )
                print(f"[RAG] 💾 Cached LLM response for future queries")
            except Exception as cache_err:
                print(f"[RAG] ⚠ Failed to cache response: {cache_err}")
        
        # ========================================
        # Step 8: Structured Output
        # ========================================
        recommendations = generate_recommendations(
            confidence_score=llm_response.confidence,
            question=request.question,
            subject=request.subject.value if request.subject else None,
            user_id=request.user_id,
            suggested_topics=llm_response.suggested_topics
        )
        
        # Helper function to extract a meaningful title
        def get_source_title(chunk) -> str:
            # Priority 1: Use existing title
            if chunk.title and chunk.title.strip() and chunk.title != "Source":
                return chunk.title
            
            # Priority 2: Extract from URL
            if chunk.url:
                from urllib.parse import urlparse
                parsed = urlparse(chunk.url)
                domain = parsed.netloc.replace('www.', '').replace('en.', '')
                # Extract page name from path
                path_parts = [p for p in parsed.path.split('/') if p]
                if path_parts:
                    page_name = path_parts[-1].replace('_', ' ').replace('-', ' ')
                    # Clean up file extensions
                    page_name = page_name.replace('.html', '').replace('.pdf', '')
                    if len(page_name) > 3:
                        return f"{page_name[:50]}"
                # Fallback to domain name
                if domain:
                    domain_name = domain.split('.')[0].title()
                    return f"{domain_name} Article"
            
            # Priority 3: Use topic from metadata
            topic = chunk.metadata.get("topic", "") if chunk.metadata else ""
            if topic and topic != "Source":
                return topic
            
            # Priority 4: Extract from text preview
            if hasattr(chunk, 'text') and chunk.text:
                # Get first meaningful sentence
                text = chunk.text[:100].strip()
                if text:
                    return f"{text[:40]}..."
            
            return "Study Material"
        
        sources = [
            SourceInfo(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                title=get_source_title(chunk),
                similarity_score=round(chunk.similarity_score, 3),
                url=chunk.url,
                page_number=chunk.page_number if chunk.page_number is not None else 0
            )
            for chunk in context.chunks_used
        ]
        
        total_time = int((time.time() - start_time) * 1000)
        
        # Web resources are now fetched earlier and included in context
        # web_resources_dict is populated in Step 5.5
        
        # ========================================
        # Step 10: Generate Flowchart (Optional)
        # ========================================
        flowchart_code = None
        try:
            flowchart_code = generate_concept_flowchart(
                question=request.question,
                answer=llm_response.answer_short,
                subject=request.subject.value if request.subject else None
            )
        except Exception as e:
            print(f"Flowchart generation error: {e}")
        
        # ========================================
        # Step 10.5: Generate Follow-Up Questions
        # ========================================
        follow_ups = []
        try:
            follow_ups = generate_follow_up_questions(
                question=request.question,
                answer_short=llm_response.answer_short,
                topic=request.subject.value if request.subject else "",
                subject=detected_subject  # Pass detected subject for better context
            )
        except Exception as e:
            print(f"Follow-up generation error: {e}")
        
        response_data = TutorResponseData(
            answer_short=llm_response.answer_short,
            answer_detailed=llm_response.answer_detailed,
            sources=sources,
            confidence_score=round(llm_response.confidence, 3),
            recommended_actions=recommendations,
            metadata=ResponseMetadata(
                tokens_used=context.total_tokens,
                retrieval_time_ms=retrieval_time,
                llm_time_ms=llm_time,
                request_id=request_id,
                # Use matched classroom subject if available, else most specific detected subject
                detected_subject=matched_subject_name if matched_subject_name else (detected_subject if detected_subject else None),
                subject_confidence=round(subject_confidence, 3) if detected_subject else None
            ),
            web_resources=web_resources_dict,
            flowchart_mermaid=flowchart_code,
            follow_up_questions=follow_ups if follow_ups else None
        )
        
        # Log success (but NOT full prompt, context, or embeddings)
        log_query_processed(
            request_id=request_id,
            user_id=request.user_id,
            question=request.question,
            subject=request.subject.value if request.subject else None,
            sources_count=len(sources),
            confidence=llm_response.confidence,
            retrieval_time_ms=retrieval_time,
            llm_time_ms=llm_time,
            total_time_ms=total_time,
            success=True
        )
        
        # ========================================
        # Step 11: Persist Chat to Database
        # ========================================
        if request.conversation_id and request.auth_token:
            try:
                from ...services.chat_persistence import save_chat_exchange
                
                # Prepare sources for storage
                sources_for_db = [
                    {
                        "type": "article",
                        "title": s.title,
                        "url": s.url,
                        "relevance": s.similarity_score,
                        "source": "qdrant"
                    }
                    for s in sources
                ]
                
                # Add web resources if available
                if web_resources_dict:
                    if web_resources_dict.get("articles"):
                        for article in web_resources_dict["articles"]:
                            sources_for_db.append({
                                "type": "article",
                                "title": article.get("title"),
                                "url": article.get("url"),
                                "snippet": article.get("snippet"),
                                "source": article.get("source", "web")
                            })
                    if web_resources_dict.get("videos"):
                        for video in web_resources_dict["videos"]:
                            sources_for_db.append({
                                "type": "video",
                                "title": video.get("title"),
                                "url": video.get("url"),
                                "thumbnailUrl": video.get("thumbnailUrl"),
                                "source": "YouTube"
                            })
                
                # Save the exchange asynchronously (fire and forget)
                import asyncio
                asyncio.create_task(save_chat_exchange(
                    conversation_id=request.conversation_id,
                    user_id=request.user_id,
                    user_message=request.question,
                    ai_response=llm_response.answer_detailed or llm_response.answer_short,
                    auth_token=request.auth_token,
                    sources=sources_for_db,
                    response_data={
                        "answer_short": llm_response.answer_short,
                        "answer_detailed": llm_response.answer_detailed,
                        "confidence": llm_response.confidence,
                        "flowchart": flowchart_code
                    },
                    subject=request.subject.value if request.subject else None,
                    classroom_id=request.classroom_id
                ))
                print(f"[CHAT-PERSIST] ✅ Saving chat to conversation {request.conversation_id}")
            except Exception as persist_error:
                print(f"[CHAT-PERSIST] ⚠ Failed to save chat: {persist_error}")
        
        return TutorQueryResponse(success=True, data=response_data)
    
    except Exception as e:
        total_time = int((time.time() - start_time) * 1000)
        log_error(
            "internal_error",
            str(e),
            request_id
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred. Please try again."
                }
            }
        )
