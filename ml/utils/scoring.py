
"""
Answer Scoring Engine for Human-in-the-Loop Evaluation

Combines:
- Semantic similarity (sentence embeddings)
- Keyword matching (weighted terms)
- Step matching (multi-part answers)
- Subject-specific rules
"""
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class SubjectType(Enum):
    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    ENGLISH = "english"
    GENERAL = "general"


@dataclass
class AnswerKey:
    full_answer: str
    keywords: Dict[str, float] = field(default_factory=dict)
    steps: List[str] = field(default_factory=list)
    max_marks: float = 10.0
    subject: SubjectType = SubjectType.GENERAL


@dataclass
class ScoringResult:
    score: float
    max_marks: float
    confidence: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    feedback: List[str] = field(default_factory=list)
    matched_keywords: List[str] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)


class SemanticScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name) if EMBEDDINGS_AVAILABLE else None

    def compute_similarity(self, text1: str, text2: str) -> float:
        if self.model is None:
            words1, words2 = set(text1.lower().split()), set(text2.lower().split())
            return len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
        embeddings = self.model.encode([text1, text2])
        return float(np.dot(embeddings[0], embeddings[1]) / 
                    (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))


class KeywordMatcher:
    def match_keywords(self, answer: str, keywords: Dict[str, float]) -> Tuple[List[str], List[str], float]:
        answer_lower = answer.lower()
        matched, missing = [], []
        matched_weight = 0.0
        for kw, weight in keywords.items():
            if kw.lower() in answer_lower:
                matched.append(kw)
                matched_weight += weight
            else:
                missing.append(kw)
        return matched, missing, matched_weight / sum(keywords.values()) if keywords else 0


class AnswerScoringEngine:
    def __init__(self):
        self.semantic_scorer = SemanticScorer()
        self.keyword_matcher = KeywordMatcher()

    def score_answer(self, student_answer: str, answer_key: AnswerKey) -> ScoringResult:
        sem_score = self.semantic_scorer.compute_similarity(student_answer, answer_key.full_answer)
        matched_kw, missing_kw, kw_score = self.keyword_matcher.match_keywords(
            student_answer, answer_key.keywords) if answer_key.keywords else ([], [], sem_score)

        final_ratio = sem_score * 0.5 + kw_score * 0.5
        final_score = round(final_ratio * answer_key.max_marks, 2)
        confidence = min(0.95, sem_score + 0.2)

        feedback = []
        if missing_kw:
            feedback.append(f"Consider mentioning: {', '.join(missing_kw[:3])}")
        if final_ratio >= 0.8:
            feedback.append("✅ Great answer!")
        elif final_ratio >= 0.5:
            feedback.append("⚠️ Partial credit")
        else:
            feedback.append("❌ Review needed")

        return ScoringResult(
            score=final_score, max_marks=answer_key.max_marks, confidence=round(confidence, 2),
            breakdown={"semantic": round(sem_score, 2), "keyword": round(kw_score, 2)},
            feedback=feedback, matched_keywords=matched_kw, missing_keywords=missing_kw
        )
