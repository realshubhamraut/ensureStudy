"""
Semantic Analyzer Module
Calculates semantic similarity using sentence transformers
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """Analyze semantic similarity between texts"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize semantic analyzer
        
        Args:
            model_name: Sentence transformer model name
        """
        try:
            logger.info(f"Loading sentence transformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info(f"Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        return self.model.encode(text, convert_to_numpy=True)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embedding vectors for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of embedding vectors
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        return self.model.encode(texts, convert_to_numpy=True)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        # Reshape for sklearn
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    
    def calculate_sentence_similarities(self, student_text: str, 
                                       model_text: str) -> Dict:
        """
        Calculate similarity between sentences of two texts
        
        Args:
            student_text: Student's answer
            model_text: Model answer
            
        Returns:
            Dictionary with similarity analysis
        """
        # Split into sentences (simple approach)
        student_sentences = [s.strip() for s in student_text.split('.') if s.strip()]
        model_sentences = [s.strip() for s in model_text.split('.') if s.strip()]
        
        if not student_sentences or not model_sentences:
            return {
                "overall_similarity": 0.0,
                "max_similarity": 0.0,
                "mean_similarity": 0.0,
                "sentence_matches": []
            }
        
        # Get embeddings
        student_embeddings = self.get_embeddings(student_sentences)
        model_embeddings = self.get_embeddings(model_sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(student_embeddings, model_embeddings)
        
        # Find best match for each model sentence
        sentence_matches = []
        for i, model_sent in enumerate(model_sentences):
            best_match_idx = np.argmax(similarity_matrix[:, i])
            best_similarity = similarity_matrix[best_match_idx, i]
            sentence_matches.append({
                "model_sentence": model_sent[:100],  # Truncate for display
                "student_sentence": student_sentences[best_match_idx][:100],
                "similarity": float(best_similarity)
            })
        
        # Calculate aggregate scores
        max_similarities = np.max(similarity_matrix, axis=0)
        
        return {
            "overall_similarity": float(np.mean(max_similarities)),
            "max_similarity": float(np.max(max_similarities)),
            "min_similarity": float(np.min(max_similarities)),
            "mean_similarity": float(np.mean(max_similarities)),
            "sentence_matches": sentence_matches,
            "coverage": float(np.sum(max_similarities > 0.5) / len(model_sentences))
        }
    
    def evaluate_answer(self, student_answer: str, 
                       model_answer: str,
                       threshold=0.5) -> Dict:
        """
        Comprehensive semantic evaluation
        
        Args:
            student_answer: Student's answer
            model_answer: Model answer
            threshold: Minimum similarity for partial credit
            
        Returns:
            Evaluation results
        """
        # Overall similarity
        overall_sim = self.calculate_similarity(student_answer, model_answer)
        
        # Sentence-level analysis
        sentence_analysis = self.calculate_sentence_similarities(student_answer, model_answer)
        
        # Determine score
        if overall_sim >= 0.9:
            score = 1.0
            grade = "Excellent"
        elif overall_sim >= 0.75:
            score = 0.85
            grade = "Good"
        elif overall_sim >= threshold:
            score = 0.6
            grade = "Adequate"
        elif overall_sim >= threshold * 0.7:
            score = 0.4
            grade = "Poor"
        else:
            score = 0.2
            grade = "Very Poor"
        
        return {
            "semantic_score": score,
            "overall_similarity": overall_sim,
            "grade": grade,
            "sentence_analysis": sentence_analysis,
            "passed_threshold": overall_sim >= threshold
        }
