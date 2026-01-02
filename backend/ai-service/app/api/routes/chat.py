"""
AI Tutor Chat Endpoint for PDF/Document-specific queries.

POST /api/tutor/chat

Used by the PDF AI chat bubble for document-specific questions.
"""
import time
import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tutor", tags=["Tutor Chat"])


class ChatRequest(BaseModel):
    """Chat request for document-specific queries"""
    query: str
    classroom_id: Optional[str] = None
    document_id: Optional[str] = None  # For PDF-specific queries
    document_title: Optional[str] = None
    context_type: Optional[str] = "general"  # general, pdf_document, recording


class ChatResponse(BaseModel):
    """Chat response"""
    success: bool
    answer: str = ""
    response: str = ""  # Alias for answer
    sources: List[dict] = []
    error: Optional[str] = None


@router.post("/document-chat", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint for document-specific queries.
    
    Used by the PDF AI chat bubble to ask questions about specific documents.
    Filters retrieval by document_id for precise context.
    """
    start_time = time.time()
    
    # Detailed logging for debugging
    print(f"\n{'='*60}")
    print(f"📄 PDF CHAT REQUEST")
    print(f"{'='*60}")
    print(f"📝 Query: {request.query}")
    print(f"🏫 Classroom ID: {request.classroom_id}")
    print(f"📑 Document ID: {request.document_id}")
    print(f"📖 Document Title: {request.document_title}")
    print(f"🔍 Context Type: {request.context_type}")
    print(f"{'='*60}")
    
    try:
        # Get retriever
        print(f"[CHAT] 🔄 Loading retriever...")
        from ...rag.retriever import get_retriever
        retriever = get_retriever()
        print(f"[CHAT] ✅ Retriever loaded")
        
        # Retrieve chunks - filter by document_id if provided
        print(f"[CHAT] 🔍 Searching Qdrant with filters:")
        print(f"       - classroom_id: {request.classroom_id}")
        print(f"       - document_id: {request.document_id}")
        
        chunks = retriever.retrieve(
            query=request.query,
            top_k=10,
            classroom_id=request.classroom_id,
            document_id=request.document_id,
            score_threshold=0.1  # Lower threshold for more results
        )
        
        print(f"[CHAT] 📦 Retrieved {len(chunks)} chunks")
        
        if chunks:
            print(f"[CHAT] Top 3 chunks:")
            for i, chunk in enumerate(chunks[:3], 1):
                score = chunk.get("similarity_score", 0)
                page = chunk.get("page", chunk.get("page_number", 0))
                text_preview = chunk.get("text", "")[:80].replace("\n", " ")
                print(f"       {i}. [Page {page}] Score: {score:.3f} - {text_preview}...")
        
        if not chunks:
            print(f"[CHAT] ⚠️ No chunks found for document_id: {request.document_id}")
            print(f"[CHAT] 🔄 Attempting on-demand indexing...")
            
            # Try to trigger on-demand indexing for this document
            try:
                import httpx
                import os
                import asyncio
                
                from ...services.material_indexer import get_material_indexer
                
                # Get document info from core service
                core_url = os.getenv("CORE_SERVICE_URL", "http://localhost:9000")
                async with httpx.AsyncClient(timeout=30.0) as client:
                    doc_response = await client.get(
                        f"{core_url}/api/internal/documents/{request.document_id}"
                    )
                
                if doc_response.status_code == 200:
                    doc_info = doc_response.json()
                    s3_path = doc_info.get("s3_path")
                    
                    if s3_path:
                        print(f"[CHAT] 📁 Found document: {doc_info.get('title')}")
                        print(f"[CHAT] 📂 S3 path: {s3_path}")
                        
                        # Trigger indexing
                        indexer = get_material_indexer()
                        
                        # Build file URL from S3 path
                        s3_bucket = os.getenv("AWS_S3_BUCKET", "ensurestudy-materials")
                        s3_region = os.getenv("AWS_REGION", "us-east-1")
                        file_url = f"https://{s3_bucket}.s3.{s3_region}.amazonaws.com/{s3_path}"
                        
                        print(f"[CHAT] 🔧 Indexing document: {file_url}")
                        
                        # Index the document
                        result = await indexer.index_material(
                            material_id=request.document_id,
                            file_url=file_url,
                            classroom_id=request.classroom_id or doc_info.get("class_id"),
                            subject=None,
                            document_title=doc_info.get("title"),
                            uploaded_by=doc_info.get("uploaded_by")
                        )
                        
                        if result.success:
                            print(f"[CHAT] ✅ Indexed {result.chunks_indexed} chunks!")
                            
                            # Try retrieving again after indexing
                            chunks = retriever.retrieve(
                                query=request.query,
                                top_k=10,
                                classroom_id=request.classroom_id,
                                document_id=request.document_id,
                                score_threshold=0.1
                            )
                            print(f"[CHAT] 📦 Retrieved {len(chunks)} chunks after indexing")
                        else:
                            print(f"[CHAT] ❌ Indexing failed: {result.error}")
                    else:
                        print(f"[CHAT] ⚠️ No S3 path found for document")
                else:
                    print(f"[CHAT] ⚠️ Document not found in core service: {doc_response.status_code}")
                    
            except Exception as idx_err:
                print(f"[CHAT] ⚠️ Could not trigger indexing: {idx_err}")
                import traceback
                traceback.print_exc()
            
            # If still no chunks, return helpful message
            if not chunks:
                return ChatResponse(
                    success=True,
                    answer="This document is being indexed for AI search. Please try again in a minute. If this persists, the teacher may need to re-upload the file.",
                    response="This document is being indexed for AI search. Please try again in a minute. If this persists, the teacher may need to re-upload the file.",
                    sources=[]
                )
        
        # Build context from chunks
        print(f"[CHAT] 📝 Building context from top 5 chunks...")
        context_parts = []
        for i, chunk in enumerate(chunks[:5], 1):
            text = chunk.get("text", chunk.get("content", ""))
            page = chunk.get("page", chunk.get("page_number", 0))
            context_parts.append(f"[Page {page}] {text}")
        
        context = "\n\n---\n\n".join(context_parts)
        print(f"[CHAT] 📄 Context length: {len(context)} chars")
        
        # Build prompt
        doc_name = request.document_title or "this document"
        system_prompt = f"""You are an AI tutor helping students understand "{doc_name}".
Answer questions based ONLY on the provided document context.
If the answer is not in the provided context, say "I couldn't find that information in this document."

IMPORTANT FORMATTING RULES:
- Use markdown for formatting (headers, bullet points, bold, etc.)
- For mathematical equations, use LaTeX notation with $ for inline math and $$ for block equations
  - Example: $a^2 + b^2 = c^2$ for inline math
  - Example: $$\\frac{{\\pi r^2}}{{2}}$$ for block equations
- Be thorough and educational - explain concepts clearly
- Include examples from the document when relevant"""
        
        user_prompt = f"""Question: {request.query}

Document Context:
{context}

Please provide a detailed, educational answer based on the document content above. Use LaTeX notation for any mathematical formulas."""

        # Generate response using the LLM
        print(f"[CHAT] 🤖 Generating LLM response...")
        try:
            from ...services.reasoning import generate_answer
            from ...api.schemas.tutor import ResponseMode
            
            llm_response = generate_answer(
                question=request.query,
                context=context,
                subject="General",
                response_mode=ResponseMode.DETAILED,  # Use DETAILED for fuller answers
                language_style="conversational"
            )
            
            answer = llm_response.answer_detailed or llm_response.answer_short or "I understand. Let me help you with that."
            print(f"[CHAT] ✅ LLM Response generated ({len(answer)} chars)")
            
        except Exception as llm_error:
            print(f"[CHAT] ⚠️ LLM error: {llm_error}")
            print(f"[CHAT] 🔄 Using fallback response from top chunk")
            # Fallback: Return top chunk as answer
            answer = f"Based on the document, here's what I found:\n\n{chunks[0].get('text', '')[:500]}"
        
        # Build sources
        sources = [
            {
                "page": chunk.get("page", chunk.get("page_number", 0)),
                "score": chunk.get("similarity_score", 0),
                "preview": chunk.get("text", "")[:100] + "..."
            }
            for chunk in chunks[:3]
        ]
        
        total_time = int((time.time() - start_time) * 1000)
        print(f"\n{'='*60}")
        print(f"✅ PDF CHAT COMPLETE in {total_time}ms")
        print(f"📤 Answer preview: {answer[:100]}...")
        print(f"{'='*60}\n")
        
        return ChatResponse(
            success=True,
            answer=answer,
            response=answer,
            sources=sources
        )
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ PDF CHAT ERROR: {e}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        return ChatResponse(
            success=False,
            answer="Sorry, I encountered an error processing your question.",
            response="Sorry, I encountered an error processing your question.",
            error=str(e)
        )
