"""
RAG Retriever with Qdrant Vector Search
"""
import os
from typing import Dict, List, Optional, Any
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.rag.qdrant_setup import get_qdrant_client


class RAGRetriever:
    """Retrieval-Augmented Generation with Qdrant vector search"""
    
    def __init__(self):
        self.qdrant_client = get_qdrant_client()
        # Use classroom_materials to match MaterialIndexer collection
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "classroom_materials")
        
        # Use free HuggingFace embeddings (no API key required)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # LLM is provided by llm_provider.py (HuggingFace, free)
        self.llm = None
    
    def retrieve_chunks(
        self,
        query: str,
        top_k: int = 5,
        subject_filter: Optional[str] = None,
        difficulty_filter: Optional[str] = None,
        score_threshold: float = 0.5,
        classroom_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from Qdrant.
        
        Args:
            query: Search query
            top_k: Number of results to return
            subject_filter: Filter by subject
            difficulty_filter: Filter by difficulty
            score_threshold: Minimum similarity score
            classroom_id: Filter by classroom (for classroom materials)
            document_id: Filter by specific document (for PDF-specific queries)
        
        Returns:
            List of chunk dictionaries with text, source, and score
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Build filter if needed
        filter_conditions = []
        if document_id:
            # Most specific filter - single document
            filter_conditions.append(
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            )
        if classroom_id:
            filter_conditions.append(
                FieldCondition(key="classroom_id", match=MatchValue(value=classroom_id))
            )
        if subject_filter:
            filter_conditions.append(
                FieldCondition(key="subject", match=MatchValue(value=subject_filter))
            )
        if difficulty_filter:
            filter_conditions.append(
                FieldCondition(key="difficulty", match=MatchValue(value=difficulty_filter))
            )
        
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Debug logging
        print(f"[RETRIEVER] 🔎 Query: {query[:50]}...")
        print(f"[RETRIEVER] 📊 Collection: {self.collection_name}")
        print(f"[RETRIEVER] 🎯 Filter conditions: {len(filter_conditions)}")
        for fc in filter_conditions:
            print(f"[RETRIEVER]    - {fc.key} = {fc.match.value}")
        print(f"[RETRIEVER] 📏 Embedding dimension: {len(query_embedding)}")
        print(f"[RETRIEVER] 📉 Score threshold: {score_threshold}")
        
        # Search Qdrant using query_points (qdrant-client 1.16+)
        from qdrant_client.models import QueryRequest
        
        try:
            query_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=search_filter,
                limit=top_k,
                with_payload=True,
                score_threshold=score_threshold
            )
            print(f"[RETRIEVER] ✅ Qdrant query successful")
            print(f"[RETRIEVER] 📦 Result type: {type(query_result)}")
        except Exception as e:
            print(f"[RETRIEVER] ❌ Qdrant query failed: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        # query_points returns QueryResponse with .points attribute
        results = query_result.points if hasattr(query_result, 'points') else []
        print(f"[RETRIEVER] 📊 Raw results count: {len(results)}")
        
        chunks = []
        for match in results:
            chunks.append({
                "id": match.id,
                "text": match.payload.get("full_text", match.payload.get("chunk_text", match.payload.get("text", ""))),
                "content": match.payload.get("full_text", match.payload.get("chunk_text", match.payload.get("text", ""))),
                "source": match.payload.get("source", match.payload.get("url", "unknown")),
                "source_type": match.payload.get("source_type", "teacher_material"),
                "page": match.payload.get("page_number", match.payload.get("page", 0)),
                "subject": match.payload.get("subject", "general"),
                "topic": match.payload.get("topic", ""),
                "title": match.payload.get("title", ""),
                "document_id": match.payload.get("document_id", ""),
                "classroom_id": match.payload.get("classroom_id", ""),
                "url": match.payload.get("url", ""),
                "difficulty": match.payload.get("difficulty", "medium"),
                "similarity_score": round(match.score, 4),
                "relevance_score": round(match.score, 4)
            })
        
        return chunks
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        classroom_id: Optional[str] = None,
        document_id: Optional[str] = None,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve method for TutorAgent compatibility.
        Wraps retrieve_chunks with classroom_id and document_id support.
        
        Args:
            query: Search query
            top_k: Number of results
            classroom_id: Classroom to filter by
            document_id: Specific document to filter by
            score_threshold: Minimum similarity
            
        Returns:
            List of chunk dictionaries
        """
        return self.retrieve_chunks(
            query=query,
            top_k=top_k,
            classroom_id=classroom_id,
            document_id=document_id,
            score_threshold=score_threshold
        )
    
    def generate_answer(
        self,
        query: str,
        chunks: List[Dict],
        student_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate answer using retrieved chunks.
        
        Args:
            query: User's question
            chunks: Retrieved context chunks
            student_context: Optional student-specific context
        
        Returns:
            Dictionary with answer and metadata
        """
        if not chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different topic.",
                "sources": [],
                "query": query,
                "context_used": 0
            }
        
        # Format context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['source']}, Page {chunk['page']}, Relevance: {chunk['similarity_score']:.0%}]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)
        
        # Build prompt
        system_prompt = """You are an expert educational tutor for the ensureStudy platform. 
Your goal is to help students learn effectively by providing clear, accurate, and engaging explanations.

Guidelines:
1. Answer based ONLY on the provided source material
2. Cite sources for major claims using [Source N] format
3. If the sources don't contain enough information, say so explicitly
4. Use clear, student-friendly language
5. Include relevant examples when helpful
6. Maintain an encouraging, supportive tone"""

        # Add student context if available
        context_note = ""
        if student_context:
            weak_topics = student_context.get("weak_topics", [])
            confidence = student_context.get("confidence_score", 50)
            
            if weak_topics:
                context_note += f"\n\nNote: This student struggles with: {', '.join(weak_topics)}. Provide extra clarity on these concepts."
            if confidence < 40:
                context_note += "\nNote: Student has low confidence in this area. Be extra encouraging and break down concepts step-by-step."
        
        user_prompt = f"""Student Question: {query}

Source Material:
{context}
{context_note}

Please provide a helpful, educational answer with source citations."""

        # Generate response
        response = self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        return {
            "answer": response.content,
            "sources": chunks,
            "query": query,
            "context_used": len(chunks)
        }
    
    def answer_with_rag(
        self,
        query: str,
        student_context: Optional[Dict] = None,
        top_k: int = 5,
        subject_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        End-to-end RAG: retrieve → generate
        
        Args:
            query: User's question
            student_context: Student-specific context
            top_k: Number of chunks to retrieve
            subject_filter: Optional subject filter
        
        Returns:
            Complete RAG response with answer and sources
        """
        # Retrieve relevant chunks
        chunks = self.retrieve_chunks(
            query=query,
            top_k=top_k,
            subject_filter=subject_filter
        )
        
        # Generate answer
        result = self.generate_answer(query, chunks, student_context)
        
        return result


# Singleton instance
_retriever: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    """Get RAG retriever singleton"""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever
