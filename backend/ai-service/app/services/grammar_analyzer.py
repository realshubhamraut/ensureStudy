"""
Grammar Analyzer Service

Analyzes text for grammar errors, sentence structure, and language quality.
Uses rule-based analysis with optional LanguageTool integration.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class GrammarResult:
    """Result of grammar analysis."""
    score: float  # 0-100
    error_count: int
    sentence_count: int
    avg_sentence_length: float
    errors: List[dict] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class GrammarAnalyzer:
    """
    Analyzes grammar and sentence structure.
    
    Scoring Factors:
    - Error rate (errors per sentence)
    - Sentence variety (length distribution)
    - Common grammar mistakes
    """
    
    # Common grammar error patterns
    GRAMMAR_PATTERNS = [
        # Subject-verb agreement
        (r'\b(I|we|they|you)\s+(is|was|has)\b', "Subject-verb agreement error"),
        (r'\b(he|she|it)\s+(are|were|have)\b', "Subject-verb agreement error"),
        # Double negatives
        (r"\bdon't\s+\w+\s+no\b", "Double negative"),
        (r"\bcan't\s+\w+\s+nothing\b", "Double negative"),
        # Article errors
        (r'\ba\s+[aeiou]', "Article 'a' before vowel sound"),
        (r'\ban\s+[^aeiou\s]', "Article 'an' before consonant"),
        # Common mistakes
        (r'\bshould\s+of\b', "Should use 'should have'"),
        (r'\bcould\s+of\b', "Should use 'could have'"),
        (r'\bwould\s+of\b', "Should use 'would have'"),
        (r'\btheir\s+is\b', "Confusion: their/there"),
        (r'\byour\s+(welcome|the\s+best)\b', "Confusion: your/you're"),
        (r'\bits\s+(a\s+great|very|really)\b', "Confusion: its/it's"),
        # Run-on sentences (very long without punctuation)
        (r'\b(\w+\s+){25,}(?![.!?])', "Possible run-on sentence"),
        # Repeated words
        (r'\b(\w+)\s+\1\b', "Repeated word"),
        # Missing capitalization after period
        (r'\.\s+[a-z]', "Missing capitalization after period"),
    ]
    
    # Filler phrases that reduce clarity
    FILLER_PHRASES = [
        "you know", "like", "basically", "actually", "literally",
        "honestly", "to be honest", "i mean", "sort of", "kind of",
        "you see", "as a matter of fact", "at the end of the day"
    ]
    
    def __init__(self):
        """Initialize grammar analyzer."""
        self._language_tool = None
        self._init_language_tool()
    
    def _init_language_tool(self):
        """Try to initialize LanguageTool for advanced checking."""
        try:
            import language_tool_python
            self._language_tool = language_tool_python.LanguageTool('en-US')
            logger.info("LanguageTool initialized successfully")
        except ImportError:
            logger.warning("language-tool-python not installed, using rule-based analysis")
        except Exception as e:
            logger.warning(f"Could not initialize LanguageTool: {e}")
    
    def analyze(self, text: str) -> GrammarResult:
        """
        Analyze text for grammar quality.
        
        Args:
            text: The text to analyze
            
        Returns:
            GrammarResult with score and details
        """
        if not text or not text.strip():
            return GrammarResult(
                score=0.0,
                error_count=0,
                sentence_count=0,
                avg_sentence_length=0.0,
                suggestions=["No text provided for analysis"]
            )
        
        text = text.strip()
        
        # Count sentences
        sentences = self._split_sentences(text)
        sentence_count = len(sentences)
        
        if sentence_count == 0:
            return GrammarResult(
                score=50.0,
                error_count=0,
                sentence_count=0,
                avg_sentence_length=0.0,
                suggestions=["Could not detect complete sentences"]
            )
        
        # Calculate average sentence length
        word_count = len(text.split())
        avg_sentence_length = word_count / sentence_count
        
        # Find errors
        errors = []
        
        # Use LanguageTool if available
        if self._language_tool:
            try:
                lt_matches = self._language_tool.check(text)
                for match in lt_matches[:10]:  # Limit to 10 errors
                    errors.append({
                        "message": match.message,
                        "context": match.context,
                        "offset": match.offset,
                        "length": match.errorLength,
                        "suggestions": match.replacements[:3] if match.replacements else []
                    })
            except Exception as e:
                logger.warning(f"LanguageTool check failed: {e}")
                errors = self._rule_based_check(text)
        else:
            errors = self._rule_based_check(text)
        
        # Count filler phrases
        filler_count = self._count_fillers(text)
        
        # Calculate score
        score = self._calculate_score(
            error_count=len(errors),
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            filler_count=filler_count,
            word_count=word_count
        )
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            errors=errors,
            avg_sentence_length=avg_sentence_length,
            filler_count=filler_count
        )
        
        return GrammarResult(
            score=score,
            error_count=len(errors),
            sentence_count=sentence_count,
            avg_sentence_length=round(avg_sentence_length, 1),
            errors=errors,
            suggestions=suggestions
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _rule_based_check(self, text: str) -> List[dict]:
        """Check grammar using regex patterns."""
        errors = []
        text_lower = text.lower()
        
        for pattern, message in self.GRAMMAR_PATTERNS:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                errors.append({
                    "message": message,
                    "context": text[max(0, match.start()-10):match.end()+10],
                    "offset": match.start(),
                    "length": match.end() - match.start()
                })
        
        return errors[:10]  # Limit to 10 errors
    
    def _count_fillers(self, text: str) -> int:
        """Count filler phrases in text."""
        text_lower = text.lower()
        count = 0
        for filler in self.FILLER_PHRASES:
            count += text_lower.count(filler)
        return count
    
    def _calculate_score(
        self,
        error_count: int,
        sentence_count: int,
        avg_sentence_length: float,
        filler_count: int,
        word_count: int
    ) -> float:
        """
        Calculate grammar score (0-100).
        
        Scoring:
        - Base: 100
        - Error penalty: -10 per error (max -50)
        - Filler penalty: -3 per filler (max -15)
        - Sentence length: penalty if <5 or >30 words avg
        """
        score = 100.0
        
        # Error penalty
        error_penalty = min(50, error_count * 10)
        score -= error_penalty
        
        # Filler penalty (normalized by word count)
        if word_count > 0:
            filler_ratio = filler_count / (word_count / 100)
            filler_penalty = min(15, filler_ratio * 5)
            score -= filler_penalty
        
        # Sentence length penalty
        if avg_sentence_length < 5:
            score -= 10  # Too short (fragments)
        elif avg_sentence_length > 30:
            score -= 10  # Too long (run-ons)
        
        return max(0, min(100, round(score, 1)))
    
    def _generate_suggestions(
        self,
        errors: List[dict],
        avg_sentence_length: float,
        filler_count: int
    ) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        if errors:
            suggestions.append(f"Found {len(errors)} grammar issues to review")
        
        if avg_sentence_length < 5:
            suggestions.append("Try using more complete sentences")
        elif avg_sentence_length > 30:
            suggestions.append("Consider breaking up long sentences for clarity")
        
        if filler_count > 3:
            suggestions.append("Reduce filler words like 'you know', 'basically', 'like'")
        
        if not suggestions:
            suggestions.append("Good grammar structure overall!")
        
        return suggestions


# Singleton instance
_analyzer_instance: Optional[GrammarAnalyzer] = None


def get_grammar_analyzer() -> GrammarAnalyzer:
    """Get or create the grammar analyzer singleton."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = GrammarAnalyzer()
    return _analyzer_instance


def analyze_grammar(text: str) -> GrammarResult:
    """
    Convenience function to analyze grammar.
    
    Args:
        text: Text to analyze
        
    Returns:
        GrammarResult with score and details
    """
    analyzer = get_grammar_analyzer()
    return analyzer.analyze(text)
