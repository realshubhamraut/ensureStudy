"""
Keyword Matching Module
Matches keywords between student and model answers
"""

from fuzzywuzzy import fuzz
from typing import List, Dict, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeywordMatcher:
    """Match keywords with fuzzy matching support"""
    
    def __init__(self, fuzzy_threshold=80):
        """
        Initialize keyword matcher
        
        Args:
            fuzzy_threshold: Minimum similarity score (0-100) for fuzzy matching
        """
        self.fuzzy_threshold = fuzzy_threshold
    
    def exact_match(self, student_tokens: List[str], model_keywords: List[str]) -> Dict:
        """
        Exact keyword matching
        
        Args:
            student_tokens: Tokens from student answer
            model_keywords: Expected keywords from model answer
            
        Returns:
            Match statistics
        """
        student_set = set(token.lower() for token in student_tokens)
        model_set = set(kw.lower() for kw in model_keywords)
        
        matched = student_set & model_set
        missed = model_set - student_set
        
        coverage = len(matched) / len(model_set) if model_set else 0
        
        return {
            "matched_keywords": list(matched),
            "missed_keywords": list(missed),
            "total_expected": len(model_set),
            "total_matched": len(matched),
            "coverage_score": coverage
        }
    
    def fuzzy_match(self, student_tokens: List[str], model_keywords: List[str]) -> Dict:
        """
        Fuzzy keyword matching to handle variations/typos
        
        Args:
            student_tokens: Tokens from student answer
            model_keywords: Expected keywords from model answer
            
        Returns:
            Match statistics with fuzzy matching
        """
        student_tokens_lower = [t.lower() for t in student_tokens]
        model_keywords_lower = [kw.lower() for kw in model_keywords]
        
        matched = []
        missed = []
        match_details = []
        
        for model_kw in model_keywords_lower:
            best_match_score = 0
            best_match_token = None
            
            # Find best matching token
            for student_token in student_tokens_lower:
                score = fuzz.ratio(model_kw, student_token)
                if score > best_match_score:
                    best_match_score = score
                    best_match_token = student_token
            
            # Check if match is good enough
            if best_match_score >= self.fuzzy_threshold:
                matched.append(model_kw)
                match_details.append({
                    "keyword": model_kw,
                    "matched_with": best_match_token,
                    "similarity": best_match_score
                })
            else:
                missed.append(model_kw)
        
        coverage = len(matched) / len(model_keywords_lower) if model_keywords_lower else 0
        
        return {
            "matched_keywords": matched,
            "missed_keywords": missed,
            "match_details": match_details,
            "total_expected": len(model_keywords_lower),
            "total_matched": len(matched),
            "coverage_score": coverage
        }
    
    def weighted_match(self, student_tokens: List[str], 
                      weighted_keywords: Dict[str, float]) -> Dict:
        """
        Match keywords with importance weights
        
        Args:
            student_tokens: Tokens from student answer
            weighted_keywords: Dict of {keyword: weight}
            
        Returns:
            Weighted match statistics
        """
        student_tokens_lower = [t.lower() for t in student_tokens]
        
        matched_weight = 0.0
        total_weight = sum(weighted_keywords.values())
        matched_kws = []
        missed_kws = []
        
        for keyword, weight in weighted_keywords.items():
            keyword_lower = keyword.lower()
            best_score = 0
            
            # Check for fuzzy matches
            for token in student_tokens_lower:
                score = fuzz.ratio(keyword_lower, token)
                if score > best_score:
                    best_score = score
            
            if best_score >= self.fuzzy_threshold:
                matched_weight += weight
                matched_kws.append(keyword)
            else:
                missed_kws.append(keyword)
        
        weighted_coverage = matched_weight / total_weight if total_weight > 0 else 0
        
        return {
            "matched_keywords": matched_kws,
            "missed_keywords": missed_kws,
            "matched_weight": matched_weight,
            "total_weight": total_weight,
            "weighted_coverage": weighted_coverage
        }
    
    def calculate_score(self, student_tokens: List[str], 
                       model_keywords: List[str],
                       use_fuzzy=True) -> float:
        """
        Calculate keyword matching score (0-1 scale)
        
        Args:
            student_tokens: Tokens from student answer
            model_keywords: Expected keywords
            use_fuzzy: Use fuzzy matching if True
            
        Returns:
            Score between 0 and 1
        """
        if use_fuzzy:
            result = self.fuzzy_match(student_tokens, model_keywords)
        else:
            result = self.exact_match(student_tokens, model_keywords)
        
        return result["coverage_score"]
