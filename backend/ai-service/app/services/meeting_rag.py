"""
Meeting RAG Service
Answers questions about meeting content using RAG pipeline
Uses Google Gemini (FREE) for response generation
"""
import os
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel

from app.services.meeting_embedding_service import meeting_embedding_service
from app.services.transcription_service import transcription_service

# Try to import Google Gemini
try:
    import google.generativeai as genai
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except ImportError:
    GEMINI_AVAILABLE = False


class Citation(BaseModel):
    """A citation from a meeting transcript"""
    recording_id: str
    meeting_id: str
    meeting_title: Optional[str] = None
    timestamp_start: float
    timestamp_end: float
    speaker_name: Optional[str] = None
    text: str
    relevance_score: float


class MeetingQAResponse(BaseModel):
    """Response from meeting Q&A"""
    answer: str
    citations: List[Citation]
    sources_count: int
    confidence: float


class MeetingRAGService:
    """
    RAG pipeline for answering questions about meeting content
    
    Workflow:
    1. Embed user question
    2. Search Qdrant for relevant transcript chunks
    3. Fetch full context if needed from MongoDB
    4. Build prompt with speaker attribution
    5. Generate answer with Gemini (FREE)
    6. Return answer with citations (timestamps for video seeking)
    """
    
    def __init__(self):
        self.embedding_service = meeting_embedding_service
        self.transcription_service = transcription_service
    
    async def ask_question(
        self,
        question: str,
        classroom_id: str,
        meeting_ids: Optional[List[str]] = None,
        top_k: int = 5
    ) -> MeetingQAResponse:
        """
        Answer a question about meeting content
        
        Args:
            question: The user's question
            classroom_id: Filter by classroom
            meeting_ids: Optional list of specific meetings to search
            top_k: Number of chunks to retrieve
        
        Returns:
            MeetingQAResponse with answer and citations
        """
        # Step 1: Search for relevant chunks
        search_results = await self.embedding_service.search(
            query=question,
            classroom_id=classroom_id,
            meeting_ids=meeting_ids,
            limit=top_k
        )
        
        if not search_results:
            return MeetingQAResponse(
                answer="I couldn't find any relevant information in the meeting recordings. Please make sure you're asking about topics that were discussed in class.",
                citations=[],
                sources_count=0,
                confidence=0.0
            )
        
        # Step 2: Build context from chunks
        context_parts = []
        citations = []
        
        for i, chunk in enumerate(search_results):
            # Format speaker names
            speaker_names = chunk.get('speaker_names', ['Teacher'])
            speaker = speaker_names[0] if speaker_names else 'Speaker'
            
            # Add to context
            context_parts.append(
                f"[{i+1}] {speaker} said (at {self._format_time(chunk['start_time'])}): "
                f"\"{chunk['chunk_text']}\""
            )
            
            # Create citation
            citations.append(Citation(
                recording_id=chunk['recording_id'],
                meeting_id=chunk['meeting_id'],
                timestamp_start=chunk['start_time'],
                timestamp_end=chunk['end_time'],
                speaker_name=speaker,
                text=chunk['chunk_text'][:200] + '...' if len(chunk['chunk_text']) > 200 else chunk['chunk_text'],
                relevance_score=chunk.get('score', 0.0)
            ))
        
        context = '\n\n'.join(context_parts)
        
        # Step 3: Generate answer
        try:
            if GEMINI_AVAILABLE:
                answer = await self._generate_with_gemini(question, context)
            else:
                # Fallback: Simple extractive response (no API needed)
                answer = self._generate_simple_response(question, context, citations)
            
            # Calculate confidence based on average relevance scores
            avg_score = sum(c.relevance_score for c in citations) / len(citations)
            confidence = min(1.0, avg_score * 1.2)  # Scale up slightly
            
            return MeetingQAResponse(
                answer=answer,
                citations=citations,
                sources_count=len(citations),
                confidence=confidence
            )
            
        except Exception as e:
            print(f"RAG generation error: {e}")
            # Fallback to simple response on error
            answer = self._generate_simple_response(question, context, citations)
            return MeetingQAResponse(
                answer=answer,
                citations=citations,
                sources_count=len(citations),
                confidence=0.5
            )
    
    async def _generate_with_gemini(self, question: str, context: str) -> str:
        """Generate answer using Google Gemini (FREE)"""
        prompt = f"""You are an AI assistant that helps students find information from their class recordings.
Answer the question based ONLY on the provided transcript excerpts.
Always cite your sources using the [1], [2], etc. notation.
If the question cannot be answered from the provided context, say so clearly.
Be concise but thorough.

Based on the following excerpts from class recordings, answer this question:

Question: {question}

Transcript excerpts:
{context}

Answer the question, citing the relevant excerpts by their numbers (e.g., [1], [2])."""

        def _call_gemini():
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        
        return await asyncio.to_thread(_call_gemini)
    
    def _generate_simple_response(self, question: str, context: str, citations: List[Citation]) -> str:
        """Generate a simple extractive response without any API"""
        if not citations:
            return "No relevant information found."
        
        # Build a simple response from the citations
        response_parts = [f"Based on the class recordings, here's what was discussed:\n"]
        
        for i, citation in enumerate(citations[:3], 1):  # Top 3 most relevant
            speaker = citation.speaker_name or "The instructor"
            time = self._format_time(citation.timestamp_start)
            text = citation.text
            response_parts.append(f"[{i}] At {time}, {speaker} mentioned: \"{text}\"")
        
        if len(citations) > 3:
            response_parts.append(f"\n...and {len(citations) - 3} more relevant excerpts.")
        
        return "\n\n".join(response_parts)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as mm:ss or hh:mm:ss"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"
    
    async def get_meeting_summary(
        self,
        recording_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get summary and key topics for a recording"""
        transcript = await self.transcription_service.get_transcript(recording_id)
        
        if not transcript:
            return None
        
        return {
            'summary': transcript.summary,
            'key_topics': transcript.key_topics,
            'duration_seconds': transcript.duration_seconds,
            'word_count': transcript.word_count,
            'speaker_count': len(transcript.speakers),
            'speakers': [
                {
                    'name': s.user_name or f'Speaker {s.speaker_id + 1}',
                    'speaking_time': s.total_speaking_time_seconds
                }
                for s in transcript.speakers
            ]
        }


# Singleton instance
meeting_rag_service = MeetingRAGService()
