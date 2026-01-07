"""
Configuration Management for Answer Evaluation System
"""

import os
from pathlib import Path

class Config:
    """Central configuration for the evaluation system"""
    
    # Project Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    MODEL_ANSWERS_DIR = DATA_DIR / "model_answers"
    KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"
    
    # NLP Models
    SPACY_MODEL = "en_core_web_sm"
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient
    # Alternative: "all-mpnet-base-v2" for better accuracy
    
    # Evaluation Weights
    SEMANTIC_WEIGHT = 0.60  # 60% weight to semantic similarity
    KEYWORD_WEIGHT = 0.25   # 25% weight to keyword matching
    CONCEPT_WEIGHT = 0.15   # 15% weight to concept coverage
    
    # Thresholds
    SEMANTIC_SIMILARITY_THRESHOLD = 0.5  # Minimum similarity for partial credit
    KEYWORD_MATCH_THRESHOLD = 0.80  # Fuzzy matching threshold
    CONCEPT_COVERAGE_THRESHOLD = 0.70  # Minimum for concept detection
    
    # Marking Scheme
    PENALTY_MISSING_CRITICAL_CONCEPT = 0.20  # Deduct 20% per missing critical concept
    PARTIAL_CREDIT_MIN = 0.30  # Minimum 30% for attempting the question
    
    # Performance Analysis
    STRONG_PERFORMANCE_THRESHOLD = 80  # Above 80% is strong
    WEAK_PERFORMANCE_THRESHOLD = 60  # Below 60% needs improvement
    
    # Output
    OUTPUT_FORMAT = "json"  # Options: json, csv, pdf
    VERBOSE_FEEDBACK = True
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.MODEL_ANSWERS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def update_weights(cls, semantic=None, keyword=None, concept=None):
        """Update evaluation weights"""
        if semantic is not None:
            cls.SEMANTIC_WEIGHT = semantic
        if keyword is not None:
            cls.KEYWORD_WEIGHT = keyword
        if concept is not None:
            cls.CONCEPT_WEIGHT = concept
        
        # Normalize to sum to 1.0
        total = cls.SEMANTIC_WEIGHT + cls.KEYWORD_WEIGHT + cls.CONCEPT_WEIGHT
        cls.SEMANTIC_WEIGHT /= total
        cls.KEYWORD_WEIGHT /= total
        cls.CONCEPT_WEIGHT /= total
