"""
Follow-Up Questions Generator

Uses Groq LLM to generate contextual follow-up questions
based on the question, answer, and topic.
"""
import os
import json
import logging
from typing import List, Optional
from functools import lru_cache

from groq import Groq

logger = logging.getLogger(__name__)


# ============================================================================
# Groq LLM Configuration
# ============================================================================

def _get_groq_client() -> Groq:
    """Get Groq client instance."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    return Groq(api_key=api_key)


# ============================================================================
# Follow-Up Question Generation
# ============================================================================

FOLLOWUP_PROMPT = """Based on this educational Q&A, generate exactly 3 follow-up questions that a curious student would naturally ask next.

Topic: {topic}
Question: {question}
Answer Summary: {answer}

Requirements:
- Questions should be directly related to the topic
- Questions should help deepen understanding
- Questions should be concise (under 10 words each)
- Each question should explore a different angle (cause, effect, example, application, etc.)

Respond with ONLY a JSON array of 3 strings, nothing else. Example format:
["Question 1?", "Question 2?", "Question 3?"]"""


@lru_cache(maxsize=500)
def generate_follow_up_questions(
    question: str,
    answer_short: str,
    topic: str = "",
    subject: Optional[str] = None
) -> List[str]:
    """
    Generate contextual follow-up questions using Groq LLM.
    
    Args:
        question: Original user question
        answer_short: Short answer provided
        topic: Extracted topic (optional)
        subject: Detected academic subject (optional)
        
    Returns:
        List of 2-3 follow-up questions
    """
    try:
        client = _get_groq_client()
        
        # Combine topic info
        topic_str = topic or subject or "General"
        
        # Truncate answer to avoid token limits
        answer_truncated = answer_short[:500] if len(answer_short) > 500 else answer_short
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast, cheap model
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful educational assistant. Respond with only valid JSON arrays."
                },
                {
                    "role": "user",
                    "content": FOLLOWUP_PROMPT.format(
                        topic=topic_str,
                        question=question,
                        answer=answer_truncated
                    )
                }
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON array
        try:
            questions = json.loads(result_text)
            if isinstance(questions, list) and len(questions) > 0:
                # Validate and clean questions
                clean_questions = []
                for q in questions[:3]:
                    if isinstance(q, str) and len(q) > 5:
                        # Ensure question mark
                        q = q.strip()
                        if not q.endswith('?'):
                            q += '?'
                        clean_questions.append(q)
                
                if clean_questions:
                    logger.info(f"[FOLLOWUP] Generated {len(clean_questions)} questions via LLM")
                    return clean_questions
        except json.JSONDecodeError:
            # Try to extract questions from non-JSON response
            questions = _extract_questions_from_text(result_text)
            if questions:
                logger.info(f"[FOLLOWUP] Extracted {len(questions)} questions from LLM text")
                return questions
        
        logger.warning(f"[FOLLOWUP] Failed to parse LLM response: {result_text[:100]}")
        
    except Exception as e:
        logger.warning(f"[FOLLOWUP] LLM error: {e}")
    
    # Fallback to smart defaults based on subject
    return _generate_fallback_questions(question, answer_short, subject)


def _extract_questions_from_text(text: str) -> List[str]:
    """Extract questions from non-JSON text response."""
    import re
    
    # Find all question-like strings
    questions = []
    
    # Look for lines ending with ?
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        # Remove numbering like "1.", "1)", "-", "*"
        line = re.sub(r'^[\d]+[.)]\s*|^[-*]\s*', '', line)
        line = line.strip('"\'')
        
        if line and len(line) > 5 and '?' in line:
            if not line.endswith('?'):
                line = line.split('?')[0] + '?'
            questions.append(line)
    
    return questions[:3]


def _generate_fallback_questions(question: str, answer: str, subject: Optional[str] = None) -> List[str]:
    """Generate smart fallback questions based on subject."""
    
    if subject:
        subject_lower = subject.lower()
        
        subject_map = {
            'math': ["Can you show a step-by-step example?", "What are common mistakes to avoid?", "When would I use this in real life?"],
            'biology': ["How does this process work in detail?", "What happens if this goes wrong?", "Can you provide a real-world example?"],
            'physics': ["Can you show a practical example?", "What's the mathematical formula?", "How is this measured?"],
            'chemistry': ["What are the products of this reaction?", "What conditions are needed?", "Is this reversible?"],
            'history': ["What were the main causes?", "What were the long-term effects?", "Who were the key figures?"],
            'computer': ["Can you show a code example?", "What are common bugs?", "How can I optimize this?"],
            'geography': ["How does this affect people?", "What are environmental impacts?", "How has this changed over time?"],
            'literature': ["What are the main themes?", "What literary devices are used?", "How does this relate to its era?"],
            'economics': ["How does this affect consumers?", "What are real-world examples?", "What are opposing views?"],
        }
        
        for key, questions in subject_map.items():
            if key in subject_lower:
                return questions
    
    # Generic educational fallbacks
    return [
        "Can you give me a specific example?",
        "Why is this important to understand?",
        "How is this applied in practice?"
    ]


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    test_cases = [
        ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.", "Biology"),
        ("Who was Napoleon Bonaparte?", "Napoleon was a French military leader who became Emperor of France and conquered much of Europe.", "History"),
        ("What is quantum mechanics?", "Quantum mechanics is a fundamental theory in physics that describes nature at the smallest scales.", "Physics"),
    ]
    
    print("=" * 60)
    print("FOLLOW-UP QUESTION GENERATOR TEST (GROQ LLM)")
    print("=" * 60)
    
    for q, a, topic in test_cases:
        print(f"\nQ: {q}")
        print(f"Topic: {topic}")
        questions = generate_follow_up_questions(q, a, topic)
        print("Follow-ups:")
        for fq in questions:
            print(f"  → {fq}")
    
    print("\n" + "=" * 60)
