"""
Subject Classifier - Auto-detect academic subject from query

Uses zero-shot classification to identify the subject domain
(Biology, Chemistry, Physics, Math, etc.) from student queries.
"""
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Academic subject labels
SUBJECT_LABELS = [
    "biology",
    "chemistry", 
    "physics",
    "mathematics",
    "computer science",
    "history",
    "geography",
    "english literature",
    "economics",
    "psychology",
    "general knowledge"
]

# Subject display names
SUBJECT_DISPLAY_NAMES = {
    "biology": "Biology",
    "chemistry": "Chemistry",
    "physics": "Physics",
    "mathematics": "Mathematics",
    "computer science": "Computer Science",
    "history": "History",
    "geography": "Geography",
    "english literature": "English Literature",
    "economics": "Economics",
    "psychology": "Psychology",
    "general knowledge": "General"
}

# Keyword hints for ambiguous topics - used to boost classification confidence
# These keywords strongly suggest a particular subject even when classifier is uncertain
SUBJECT_KEYWORDS = {
    "physics": [
        "thermodynamic", "thermodynamics", "entropy", "enthalpy", "heat engine",
        "newton", "force", "motion", "velocity", "acceleration", "momentum",
        "gravity", "electromagnetism", "electromagnetic", "quantum", "relativity",
        "wave", "frequency", "wavelength", "optics", "lens", "refraction",
        "energy", "work", "power", "joule", "watt", "kinetic", "potential"
    ],
    "chemistry": [
        "reaction", "compound", "element", "atom", "molecule", "ion", "bond",
        "acid", "base", "ph", "oxidation", "reduction", "catalyst", "mole",
        "periodic table", "electron configuration", "stoichiometry", "organic"
    ],
    "biology": [
        "cell", "dna", "rna", "protein", "enzyme", "photosynthesis", "mitosis",
        "meiosis", "evolution", "ecosystem", "organism", "species", "genetics",
        "chromosome", "nucleus", "bacteria", "virus", "metabolism"
    ],
    "mathematics": [
        "equation", "integral", "derivative", "calculus", "algebra", "matrix",
        "polynomial", "factorial", "trigonometry", "sine", "cosine", "logarithm",
        "probability", "statistics", "geometry", "theorem", "proof"
    ]
}


class SubjectClassifier:
    """
    Fast subject classification using zero-shot classifier
    
    Automatically detects subject from query text:
    - "What is photosynthesis?" -> Biology
    - "Solve x^2 + 5x + 6 = 0" -> Mathematics
    - "Explain Newton's laws" -> Physics
    """
    
    def __init__(self):
        self._classifier = None
        logger.info("Initialized Subject Classifier")
    
    @property
    def classifier(self):
        """Lazy load classifier"""
        if self._classifier is None:
            from app.services.llm_provider import get_classifier
            self._classifier = get_classifier()
        return self._classifier
    
    def classify_subject(self, query: str, min_confidence: float = 0.3) -> Dict[str, any]:
        """
        Classify query into academic subject
        
        Args:
            query: Student's question
            min_confidence: Minimum confidence threshold (default 0.3)
        
        Returns:
            {
                "subject": "biology",  # lowercase key
                "display_name": "Biology",
                "confidence": 0.85,
                "all_scores": {...}  # All subject scores
            }
        """
        if not query or len(query.strip()) < 3:
            return self._default_response()
        
        try:
            # Classify against all subject labels
            scores = self.classifier.classify(
                query,
                SUBJECT_LABELS,
                multi_label=False  # Single best match
            )
            
            # Get top subject from classifier
            top_subject = max(scores, key=scores.get)
            top_score = scores[top_subject]
            
            # KEYWORD BOOSTING: Check if query contains strong keywords for a subject
            # This helps correct misclassifications like "thermodynamics" -> chemistry
            query_lower = query.lower()
            for subject, keywords in SUBJECT_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in query_lower:
                        # Found a strong keyword match
                        keyword_subject_score = scores.get(subject, 0)
                        # If classifier was uncertain (low score) or picked wrong subject
                        if top_score < 0.7 or (top_subject != subject and keyword_subject_score >= 0.2):
                            logger.info(
                                f"Keyword boost: '{keyword}' suggests {subject} "
                                f"(was {top_subject}:{top_score:.2f})"
                            )
                            top_subject = subject
                            top_score = max(0.85, keyword_subject_score + 0.3)  # Boost confidence
                            break
                else:
                    continue
                break  # Exit outer loop if we found a keyword match
            
            # If confidence too low, return general
            if top_score < min_confidence:
                logger.info(
                    f"Subject confidence too low ({top_score:.2f} < {min_confidence}), "
                    f"defaulting to general"
                )
                top_subject = "general knowledge"
                top_score = 1.0
            
            logger.info(
                f"Subject classified: {top_subject} (confidence: {top_score:.2f})"
            )
            
            return {
                "subject": top_subject.replace(" ", "_"),  # "computer science" -> "computer_science"
                "display_name": SUBJECT_DISPLAY_NAMES.get(top_subject, "General"),
                "confidence": top_score,
                "all_scores": scores
            }
            
        except Exception as e:
            logger.error(f"Subject classification error: {e}", exc_info=True)
            return self._default_response()
    
    def _default_response(self) -> Dict[str, any]:
        """Default response when classification fails"""
        return {
            "subject": "general",
            "display_name": "General",
            "confidence": 1.0,
            "all_scores": {}
        }


# Singleton instance
_subject_classifier = None

def get_subject_classifier() -> SubjectClassifier:
    """Get or create singleton subject classifier"""
    global _subject_classifier
    if _subject_classifier is None:
        _subject_classifier = SubjectClassifier()
    return _subject_classifier
