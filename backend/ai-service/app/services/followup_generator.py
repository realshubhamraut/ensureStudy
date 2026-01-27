"""
Follow-Up Questions Generator

Uses HuggingFace text generation API to generate contextual follow-up questions
based on the topic and answer.
"""
import os
import re
import logging
from typing import List
from functools import lru_cache

import httpx

logger = logging.getLogger(__name__)


# ============================================================================
# HuggingFace API Configuration
# ============================================================================

# Use a fast text generation model for follow-up generation
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"


def _get_hf_token() -> str:
    """Get HuggingFace API token from environment."""
    return os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or ""


# ============================================================================
# Follow-Up Question Generation
# ============================================================================

@lru_cache(maxsize=500)
def generate_follow_up_questions(
    question: str,
    answer_short: str,
    topic: str = ""
) -> List[str]:
    """
    Generate contextual follow-up questions based on the Q&A.
    
    Args:
        question: Original user question
        answer_short: Short answer provided
        topic: Extracted topic (optional)
        
    Returns:
        List of 2-3 follow-up questions
    """
    hf_token = _get_hf_token()
    
    # Build prompt for generating follow-ups
    prompt = f"""Given this educational Q&A, suggest 3 follow-up questions a student might ask:

Topic: {topic or 'General'}
Question: {question}
Answer: {answer_short[:300]}

Generate exactly 3 short follow-up questions (one per line):"""
    
    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    try:
        response = httpx.post(
            HF_API_URL,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 100,
                    "temperature": 0.7,
                    "do_sample": True
                }
            },
            timeout=5.0  # Fast timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                
                # Parse the generated questions
                questions = _parse_questions(generated_text)
                
                if questions:
                    logger.debug(f"[FOLLOWUP] Generated {len(questions)} questions")
                    return questions
        
        logger.debug(f"[FOLLOWUP] HF API returned {response.status_code}, using fallback")
        
    except httpx.TimeoutException:
        logger.debug("[FOLLOWUP] HF API timeout, using fallback")
    except Exception as e:
        logger.debug(f"[FOLLOWUP] Error: {e}, using fallback")
    
    # Fallback to smart keyword-based questions
    return _generate_fallback_questions(question, answer_short)


def _parse_questions(text: str) -> List[str]:
    """Parse generated text into individual questions."""
    # Split by newlines and numbers
    lines = re.split(r'[\n\r]+|(?:\d+[.)]\s*)', text)
    
    questions = []
    for line in lines:
        line = line.strip()
        # Only keep lines that look like questions
        if line and len(line) > 10 and (line.endswith('?') or 'what' in line.lower() or 'how' in line.lower() or 'why' in line.lower()):
            # Add question mark if missing
            if not line.endswith('?'):
                line += '?'
            questions.append(line)
    
    return questions[:3]


def _generate_fallback_questions(question: str, answer: str) -> List[str]:
    """Generate smart fallback questions based on content keywords."""
    answer_lower = answer.lower()
    question_lower = question.lower()
    combined = answer_lower + " " + question_lower
    
    # Topic-based smart fallbacks
    if any(w in combined for w in ['math', 'equation', 'formula', 'calculate', 'solve']):
        return [
            "Can you show a step-by-step example?",
            "What are common mistakes to avoid?",
            "When would I use this formula?"
        ]
    
    if any(w in combined for w in ['history', 'war', 'revolution', 'century', 'era', 'king', 'empire']):
        return [
            "What were the main causes?",
            "What were the long-term effects?",
            "Who were the key figures involved?"
        ]
    
    if any(w in combined for w in ['biology', 'cell', 'organism', 'dna', 'evolution', 'species']):
        return [
            "How does this process work?",
            "What happens if this goes wrong?",
            "Can you explain with a diagram?"
        ]
    
    if any(w in combined for w in ['physics', 'force', 'energy', 'motion', 'wave', 'gravity']):
        return [
            "Can you show a real-world example?",
            "How is this measured?",
            "What's the mathematical formula?"
        ]
    
    if any(w in combined for w in ['chemistry', 'reaction', 'element', 'compound', 'molecule', 'atom']):
        return [
            "What are the products of this reaction?",
            "Is this reaction reversible?",
            "What conditions are needed?"
        ]
    
    if any(w in combined for w in ['code', 'programming', 'python', 'javascript', 'function', 'algorithm']):
        return [
            "Can you show a code example?",
            "What are common bugs to avoid?",
            "How can I optimize this?"
        ]
    
    if any(w in combined for w in ['geography', 'country', 'climate', 'population', 'continent']):
        return [
            "How does this affect people living there?",
            "What are the environmental impacts?",
            "How has this changed over time?"
        ]
    
    if any(w in combined for w in ['literature', 'book', 'poem', 'author', 'character', 'theme']):
        return [
            "What are the main themes?",
            "How does this relate to its historical context?",
            "What literary devices are used?"
        ]
    
    if any(w in combined for w in ['economics', 'market', 'supply', 'demand', 'price', 'trade']):
        return [
            "How does this affect consumers?",
            "What are real-world examples?",
            "What are the opposing views?"
        ]
    
    # Generic fallbacks
    return [
        "Can you give me an example?",
        "Why is this important to know?",
        "How is this used in real life?"
    ]


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # Test the generator
    test_cases = [
        ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight into energy."),
        ("Who was Napoleon?", "Napoleon Bonaparte was a French military leader who became Emperor of France."),
        ("How do I solve quadratic equations?", "Use the quadratic formula: x = (-b ± √(b²-4ac)) / 2a"),
    ]
    
    print("=" * 60)
    print("FOLLOW-UP QUESTION GENERATOR TEST")
    print("=" * 60)
    
    for q, a in test_cases:
        print(f"\nQ: {q}")
        print(f"A: {a[:50]}...")
        questions = generate_follow_up_questions(q, a)
        print("Follow-ups:")
        for fq in questions:
            print(f"  → {fq}")
