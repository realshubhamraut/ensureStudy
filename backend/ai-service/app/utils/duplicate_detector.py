"""
Multi-Layer Duplicate Detection for Type 5 Learning Agent

Prevents duplicate questions from being added to the database using:
1. Hash Layer - Exact text matching via normalized hash
2. Embedding Layer - Semantic similarity via sentence embeddings
3. LLM Layer - Conceptual duplicate check for edge cases
"""
import hashlib
import logging
import re
from typing import List, Dict, Tuple, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Similarity thresholds
HASH_MATCH_THRESHOLD = 1.0  # Exact match
EMBEDDING_SIMILARITY_THRESHOLD = 0.92  # High semantic similarity
LLM_DUPLICATE_THRESHOLD = 0.85  # LLM confidence


def normalize_question_text(text: str) -> str:
    """
    Normalize question text for consistent hashing.
    
    - Lowercase
    - Remove extra whitespace
    - Remove punctuation variations
    - Standardize numbers
    """
    # Lowercase
    normalized = text.lower()
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Standardize common variations
    normalized = normalized.replace("'", "'")
    normalized = normalized.replace('"', '"')
    normalized = normalized.replace('"', '"')
    
    # Remove trailing punctuation for comparison
    normalized = normalized.rstrip('?.')
    
    return normalized


def compute_question_hash(text: str) -> str:
    """
    Compute SHA256 hash of normalized question text.
    
    Used for fast exact-match duplicate detection.
    """
    normalized = normalize_question_text(text)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def check_hash_duplicate(
    question_hash: str,
    existing_hashes: List[str]
) -> bool:
    """
    Layer 1: Check for exact hash match.
    
    Returns True if duplicate found.
    """
    return question_hash in existing_hashes


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Simple text similarity using word overlap (Jaccard similarity).
    
    Used as fallback when embeddings not available.
    """
    words1 = set(normalize_question_text(text1).split())
    words2 = set(normalize_question_text(text2).split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)


async def compute_embedding_similarity(
    question_text: str,
    existing_questions: List[Dict],
    embedding_service=None
) -> Tuple[bool, Optional[str], float]:
    """
    Layer 2: Check semantic similarity using embeddings.
    
    Returns:
        Tuple of (is_duplicate, most_similar_question_id, max_similarity)
    """
    if not existing_questions:
        return False, None, 0.0
    
    max_similarity = 0.0
    most_similar_id = None
    
    try:
        if embedding_service:
            # Use sentence embeddings for semantic comparison
            from sentence_transformers import SentenceTransformer
            
            model = embedding_service or SentenceTransformer('all-MiniLM-L6-v2')
            
            new_embedding = model.encode(question_text)
            
            for q in existing_questions:
                existing_text = q.get('question_text', '')
                existing_embedding = model.encode(existing_text)
                
                # Cosine similarity
                similarity = float(
                    (new_embedding @ existing_embedding) / 
                    (np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding))
                )
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_id = q.get('id')
        else:
            # Fallback to text similarity
            for q in existing_questions:
                existing_text = q.get('question_text', '')
                similarity = compute_text_similarity(question_text, existing_text)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_id = q.get('id')
    
    except Exception as e:
        logger.warning(f"Embedding similarity check failed: {e}")
        # Fallback to text similarity
        for q in existing_questions:
            existing_text = q.get('question_text', '')
            similarity = compute_text_similarity(question_text, existing_text)
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_id = q.get('id')
    
    is_duplicate = max_similarity >= EMBEDDING_SIMILARITY_THRESHOLD
    return is_duplicate, most_similar_id, max_similarity


async def check_llm_duplicate(
    new_question: str,
    similar_question: str,
    llm_service=None
) -> Tuple[bool, float, str]:
    """
    Layer 3: Use LLM to determine if questions are conceptually duplicate.
    
    This catches edge cases where questions are phrased differently
    but test the exact same concept.
    
    Returns:
        Tuple of (is_duplicate, confidence, explanation)
    """
    prompt = f"""Compare these two questions and determine if they are duplicates.
Two questions are duplicates if they:
1. Test the exact same concept/knowledge
2. Would have the same answer
3. A student who knows the answer to one would definitely know the answer to the other

Question 1: {new_question}

Question 2: {similar_question}

Respond in JSON format:
{{
    "is_duplicate": true/false,
    "confidence": 0.0-1.0,
    "explanation": "brief reason"
}}
"""
    
    try:
        if llm_service:
            response = await llm_service.ainvoke(prompt)
            import json
            result = json.loads(response)
            return (
                result.get("is_duplicate", False),
                result.get("confidence", 0.5),
                result.get("explanation", "")
            )
    except Exception as e:
        logger.warning(f"LLM duplicate check failed: {e}")
    
    # Default to not duplicate if LLM fails
    return False, 0.0, "LLM check skipped"


async def check_duplicate(
    question_text: str,
    existing_questions: List[Dict],
    use_llm: bool = True,
    embedding_service=None,
    llm_service=None
) -> Dict:
    """
    Main duplicate detection function - runs all layers.
    
    Args:
        question_text: The new question to check
        existing_questions: List of existing questions with 'id', 'question_text', 'question_hash'
        use_llm: Whether to use LLM for final verification
        
    Returns:
        {
            "is_duplicate": bool,
            "duplicate_of": question_id or None,
            "detection_layer": "hash" | "embedding" | "llm" | None,
            "similarity_score": float,
            "explanation": str
        }
    """
    result = {
        "is_duplicate": False,
        "duplicate_of": None,
        "detection_layer": None,
        "similarity_score": 0.0,
        "explanation": ""
    }
    
    if not existing_questions:
        result["explanation"] = "No existing questions to compare"
        return result
    
    # Layer 1: Hash check
    new_hash = compute_question_hash(question_text)
    existing_hashes = {q.get('question_hash'): q.get('id') for q in existing_questions if q.get('question_hash')}
    
    if new_hash in existing_hashes:
        result["is_duplicate"] = True
        result["duplicate_of"] = existing_hashes[new_hash]
        result["detection_layer"] = "hash"
        result["similarity_score"] = 1.0
        result["explanation"] = "Exact text match (hash collision)"
        logger.info(f"Duplicate detected via hash: {new_hash[:16]}...")
        return result
    
    # Layer 2: Embedding similarity
    is_similar, similar_id, similarity = await compute_embedding_similarity(
        question_text,
        existing_questions,
        embedding_service
    )
    
    result["similarity_score"] = similarity
    
    if is_similar:
        result["is_duplicate"] = True
        result["duplicate_of"] = similar_id
        result["detection_layer"] = "embedding"
        result["explanation"] = f"High semantic similarity ({similarity:.2%})"
        logger.info(f"Duplicate detected via embedding: similarity={similarity:.2%}")
        return result
    
    # Layer 3: LLM check (only if similarity is borderline)
    if use_llm and similarity >= 0.7 and similar_id:
        similar_question = next(
            (q['question_text'] for q in existing_questions if q.get('id') == similar_id),
            None
        )
        
        if similar_question:
            is_dup, confidence, explanation = await check_llm_duplicate(
                question_text,
                similar_question,
                llm_service
            )
            
            if is_dup and confidence >= LLM_DUPLICATE_THRESHOLD:
                result["is_duplicate"] = True
                result["duplicate_of"] = similar_id
                result["detection_layer"] = "llm"
                result["similarity_score"] = confidence
                result["explanation"] = f"LLM verified duplicate: {explanation}"
                logger.info(f"Duplicate detected via LLM: confidence={confidence:.2%}")
                return result
    
    result["explanation"] = "No duplicate detected"
    return result


def batch_check_duplicates(
    new_questions: List[str],
    existing_questions: List[Dict]
) -> List[Dict]:
    """
    Check multiple questions for duplicates efficiently.
    
    First filters using hash, then uses embedding for remaining.
    """
    import asyncio
    
    async def check_all():
        results = []
        for q in new_questions:
            result = await check_duplicate(q, existing_questions, use_llm=False)
            results.append(result)
        return results
    
    return asyncio.run(check_all())
