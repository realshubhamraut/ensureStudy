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
        try:
            from ...services.material_indexer import get_material_indexer
            indexer = get_material_indexer()
            classroom_results = indexer.search_classroom_materials(
                query=request.question,
                classroom_id=request.classroom_id,  # None = search all
                top_k=5,
                score_threshold=0.3
            )
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
                  (f" from classroom {request.classroom_id}" if request.classroom_id else " from ALL classrooms"))
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
        # Step 5.5: Web Content (CACHE-FIRST)
        # ========================================
        web_context = ""
        web_resources_dict = None
        cache_hit = False
        
        if request.find_resources:
            try:
                from ...services.web_cache_service import search_cache, store_in_cache
                from ...services.web_ingest_service import ingest_web_resources
                
                # CACHE-FIRST: Check cache for similar query
                cached = search_cache(request.question, threshold=0.85)
                
                if cached:
                    # CACHE HIT - Use cached content (no web crawl needed!)
                    print(f"[RAG] ✅ CACHE HIT! Similarity: {cached.similarity:.3f}")
                    cache_hit = True
                    web_context = f"\n\n--- Cached Web Knowledge (similarity: {cached.similarity:.2f}) ---\n{cached.answer[:2000]}"
                    web_resources_dict = {
                        "articles": [{
                            "id": "cached_1",
                            "title": "Cached Result",
                            "url": cached.sources[0] if cached.sources else "",
                            "source": "Cache",
                            "snippet": cached.answer[:200],
                            "trustScore": cached.confidence
                        }]
                    }
                    print(f"[RAG] ⚡ Skipped web crawl - using cache!")
                else:
                    # CACHE MISS - Crawl web
                    print(f"[RAG] 🌐 Cache miss, fetching web content for: {request.question[:50]}...")
                    
                    # Convert conversation_history to dicts for web ingest
                    history_dicts = None
                    if request.conversation_history:
                        history_dicts = [{"role": m.role, "content": m.content} for m in request.conversation_history]
                    
                    web_result = await ingest_web_resources(
                        query=request.question,
                        subject=request.subject.value if request.subject else None,
                        max_sources=2,
                        conversation_history=history_dicts
                    )
                    
                    if web_result.success and web_result.resources:
                        # Build web context from crawled content
                        web_parts = []
                        sources_list = []
                        for resource in web_result.resources[:2]:
                            if resource.clean_content:
                                snippet = resource.clean_content[:1500]
                                web_parts.append(f"[Source: {resource.title}]\n{snippet}")
                                sources_list.append(resource.url)
                        
                        if web_parts:
                            web_context = "\n\n--- Web Knowledge ---\n" + "\n\n".join(web_parts)
                            print(f"[RAG] ✅ Added {len(web_parts)} web sources to context")
                            
                            # Store in cache for future queries
                            combined_answer = "\n\n".join([r.clean_content[:2000] for r in web_result.resources if r.clean_content])
                            store_in_cache(
                                query=request.question,
                                answer=combined_answer,
                                sources=sources_list,
                                confidence=0.9
                            )
                            print(f"[RAG] 💾 Stored in cache for future use!")
                        
                        # Also prepare for UI display
                    web_resources_dict = {
                        "articles": [
                            {
                                "id": r.id,
                                "title": r.title,
                                "url": r.url,
                                "source": r.source_name,
                                "snippet": r.summary[:200] if r.summary else "",
                                "trustScore": r.trust_score
                            }
                            for r in web_result.resources if r.clean_content
                        ]
                    }
                
                # Fetch images from Brave API (parallel with articles)
                try:
                    brave_images = await search_images_brave(request.question, count=3)
                    if brave_images:
                        if web_resources_dict is None:
                            web_resources_dict = {}
                        web_resources_dict["images"] = brave_images
                        print(f"[RAG] 🖼️ Added {len(brave_images)} images from Brave")
                except Exception as img_err:
                    print(f"[RAG] ⚠ Brave image error: {img_err}")
                
                # Fetch videos from YouTube API
                try:
                    youtube_videos = await search_videos_youtube(request.question, max_results=3)
                    if youtube_videos:
                        if web_resources_dict is None:
                            web_resources_dict = {}
                        web_resources_dict["videos"] = youtube_videos
                        print(f"[RAG] 🎬 Added {len(youtube_videos)} videos from YouTube")
                except Exception as vid_err:
                    print(f"[RAG] ⚠ YouTube video error: {vid_err}")
                    
            except Exception as e:
                print(f"[RAG] ⚠ Web fetch error: {e}")
        
        # Combine Qdrant context + Web context
        full_context = context.context_text
        if web_context:
            full_context = full_context + web_context
        
        # ========================================
        # Step 6 & 7: Prompt + LLM (Mistral)
        # ========================================
        llm_response = generate_answer(
            question=request.question,
            context=full_context,  # Now includes web content!
            subject=request.subject.value if request.subject else "General",
            response_mode=request.response_mode,
            language_style=request.language_style.value
        )
        
        llm_time = llm_response.generation_time_ms
        
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
                request_id=request_id
            ),
            web_resources=web_resources_dict,
            flowchart_mermaid=flowchart_code
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
