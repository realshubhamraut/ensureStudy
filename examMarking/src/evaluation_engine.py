"""
Evaluation Engine
Main evaluation logic combining keywords, semantics, and concepts
"""

from .keyword_matcher import KeywordMatcher
from .semantic_analyzer import SemanticAnalyzer
from .concept_detector import ConceptDetector
from .nlp_preprocessor import NLPPreprocessor
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Core evaluation engine combining multiple analysis methods"""
    
    def __init__(self, config=None):
        """
        Initialize evaluation engine
        
        Args:
            config: Configuration object with weights and thresholds
        """
        self.config = config
        
        # Initialize components
        self.preprocessor = NLPPreprocessor()
        self.keyword_matcher = KeywordMatcher()
        self.semantic_analyzer = SemanticAnalyzer()
        self.concept_detector = ConceptDetector()
        
        # Default weights
        self.semantic_weight = 0.60
        self.keyword_weight = 0.25
        self.concept_weight = 0.15
        
        if config:
            self.semantic_weight = getattr(config, 'SEMANTIC_WEIGHT', 0.60)
            self.keyword_weight = getattr(config, 'KEYWORD_WEIGHT', 0.25)
            self.concept_weight = getattr(config, 'CONCEPT_WEIGHT', 0.15)
        
        logger.info(f"Evaluation weights - Semantic: {self.semantic_weight}, "
                   f"Keyword: {self.keyword_weight}, Concept: {self.concept_weight}")
    
    def evaluate(self, student_answer: str, model_answer: str, 
                question_data: Dict = None, max_marks: float = 10.0) -> Dict:
        """
        Comprehensive evaluation of student answer
        
        Args:
            student_answer: Student's answer text
            model_answer: Model answer text
            question_data: Dictionary with keywords, concepts, etc.
            max_marks: Maximum marks for the question
            
        Returns:
            Detailed evaluation results
        """
        logger.info("Starting evaluation...")
        
        # Preprocess texts
        student_processed = self.preprocessor.preprocess(
            student_answer, 
            pipeline=['clean', 'lemmatize', 'keywords']
        )
        model_processed = self.preprocessor.preprocess(
            model_answer,
            pipeline=['clean', 'lemmatize', 'keywords']
        )
        
        student_tokens = student_processed["lemmas"]
        model_tokens = model_processed["lemmas"]
        
        # Extract model keywords
        model_keywords = [kw[0] for kw in model_processed["keywords"]]
        if question_data and "keywords" in question_data:
            model_keywords = question_data["keywords"]
        
        # 1. Keyword Matching
        logger.info("Performing keyword matching...")
        keyword_result = self.keyword_matcher.fuzzy_match(student_tokens, model_keywords)
        keyword_score = keyword_result["coverage_score"]
        
        # 2. Semantic Similarity
        logger.info("Calculating semantic similarity...")
        semantic_result = self.semantic_analyzer.evaluate_answer(
            student_answer, model_answer
        )
        semantic_score = semantic_result["semantic_score"]
        
        # 3. Concept Detection
        concept_score = 0.5  # Default if no concepts provided
        concept_result = {}
        
        if question_data and "concepts" in question_data:
            logger.info("Detecting concepts...")
            concept_result = self.concept_detector.detect_concepts(
                student_tokens, 
                question_data["concepts"]
            )
            concept_score = self.concept_detector.calculate_concept_score(concept_result)
        
        # Calculate weighted final score
        final_score = (
            self.semantic_weight * semantic_score +
            self.keyword_weight * keyword_score +
            self.concept_weight * concept_score
        )
        
        # Convert to marks
        marks_obtained = final_score * max_marks
        
        # Compile results
        evaluation_result = {
            "student_answer": student_answer,
            "model_answer": model_answer,
            "max_marks": max_marks,
            "marks_obtained": round(marks_obtained, 2),
            "percentage": round(final_score * 100, 2),
            "scores": {
                "semantic": {
                    "score": round(semantic_score, 3),
                    "weight": self.semantic_weight,
                    "contribution": round(semantic_score * self.semantic_weight, 3),
                    "details": semantic_result
                },
                "keyword": {
                    "score": round(keyword_score, 3),
                    "weight": self.keyword_weight,
                    "contribution": round(keyword_score * self.keyword_weight, 3),
                    "details": keyword_result
                },
                "concept": {
                    "score": round(concept_score, 3),
                    "weight": self.concept_weight,
                    "contribution": round(concept_score * self.concept_weight, 3),
                    "details": concept_result
                }
            },
            "final_score": round(final_score, 3),
            "processed_data": {
                "student_tokens_count": len(student_tokens),
                "model_tokens_count": len(model_tokens),
                "student_keywords": student_processed.get("keywords", [])[:10]
            }
        }
        
        logger.info(f"Evaluation complete - Marks: {marks_obtained}/{max_marks} ({final_score*100:.1f}%)")
        
        return evaluation_result
    
    def batch_evaluate(self, evaluations: List[Dict], max_marks: float = 10.0) -> List[Dict]:
        """
        Evaluate multiple student answers
        
        Args:
            evaluations: List of dicts with 'student_answer', 'model_answer', 'question_data'
            max_marks: Maximum marks per question
            
        Returns:
            List of evaluation results
        """
        results = []
        for i, eval_data in enumerate(evaluations):
            logger.info(f"Evaluating answer {i+1}/{len(evaluations)}")
            result = self.evaluate(
                eval_data["student_answer"],
                eval_data["model_answer"],
                eval_data.get("question_data"),
                max_marks
            )
            result["question_number"] = i + 1
            results.append(result)
        
        return results
