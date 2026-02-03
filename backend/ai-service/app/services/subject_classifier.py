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

# LLM prompt for dynamic multi-subject extraction
SUBJECT_EXTRACTION_PROMPT = """You are an expert academic subject classifier. Given a student's question, identify the most specific subjects/topics it relates to.

RULES:
1. Return 1-3 subjects ordered from MOST SPECIFIC to MOST GENERAL
2. First should be the most specific match (e.g., "Linux", "Java", "Genetics", "Calculus")
3. Last should be the broad academic field (e.g., "Computer Science", "Biology", "Mathematics")
4. Use single words or short phrases (1-3 words max)
5. Return as comma-separated list, nothing else

EXAMPLES:
Q: "How do I write a for loop in bash?"
A: Linux, Shell Scripting, Computer Science

Q: "Explain inheritance in Java"
A: Java, Object-Oriented Programming, Computer Science

Q: "What is Mendel's law of segregation?"
A: Genetics, Biology

Q: "Solve x^2 - 5x + 6 = 0"
A: Algebra, Mathematics

Q: "What is thermodynamics?"
A: Thermodynamics, Physics

Q: "{question}"
A:"""


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
    
    def classify_subject_multi(self, query: str, max_subjects: int = 3) -> Dict[str, any]:
        """
        Classify query into multiple subjects ordered by specificity using LLM.
        
        Uses Groq API for dynamic, intelligent subject extraction.
        Example: "shell scripting loops" -> ["Linux", "Shell Scripting", "Computer Science"]
        
        Args:
            query: Student's question
            max_subjects: Maximum number of subjects to return (default 3)
        
        Returns:
            {
                "subjects": ["linux", "shell_scripting", "computer_science"],
                "display_names": ["Linux", "Shell Scripting", "Computer Science"],
                "confidences": [0.95, 0.90, 0.85],
                "primary": "linux"  # Most specific match
            }
        """
        if not query or len(query.strip()) < 3:
            return self._default_multi_response()
        
        try:
            # Try LLM-based extraction first (industry standard)
            llm_result = self._extract_subjects_llm(query)
            if llm_result:
                print(f"[SUBJECT] 🎯 LLM Multi-detect: {' → '.join(llm_result['display_names'])}")
                return llm_result
        except Exception as e:
            logger.warning(f"[SUBJECT] LLM extraction failed, falling back to classifier: {e}")
        
        # Fallback to zero-shot classifier
        broad_result = self.classify_subject(query)
        return {
            "subjects": [broad_result["subject"]],
            "display_names": [broad_result["display_name"]],
            "confidences": [broad_result["confidence"]],
            "primary": broad_result["subject"]
        }
    
    def _extract_subjects_llm(self, query: str) -> Optional[Dict[str, any]]:
        """
        Use Groq LLM to extract multiple subjects from query.
        
        This is the industry-standard approach - using AI for dynamic classification
        instead of hardcoded keyword lists.
        """
        import os
        import httpx
        
        try:
            from .api_key_manager import get_key
            api_key = get_key("GROQ_API_KEY")
        except:
            api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            logger.warning("[SUBJECT-LLM] No GROQ_API_KEY found")
            return None
        
        # Stricter prompt that emphasizes only returning subject names
        prompt = f"""Classify this academic question into 1-3 subject categories.

RULES:
- Return ONLY subject names, comma-separated
- First = most specific (e.g., Gravity, Thermodynamics, Algebra)
- Last = broad field (e.g., Physics, Chemistry, Mathematics, Biology, Computer Science, History)
- Use 1-3 words per subject
- NO question text, NO explanations

Examples:
"How do I write a for loop in bash?" → Linux, Shell Scripting, Computer Science
"Explain Newton's law of gravitation" → Gravity, Classical Mechanics, Physics
"What is photosynthesis?" → Photosynthesis, Plant Biology, Biology
"Solve x^2 - 5x + 6 = 0" → Quadratic Equations, Algebra, Mathematics

Question: "{query[:150]}"
Subjects:"""
        
        try:
            # Call Groq API (fast, <500ms)
            response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",  # Fast model for classification
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 30,  # Reduced - we only need subject names
                    "temperature": 0.0  # Zero temp for deterministic results
                },
                timeout=5.0
            )
            
            if response.status_code != 200:
                logger.warning(f"[SUBJECT-LLM] Groq API error: {response.status_code}")
                return None
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Clean up the response
            # Remove any echo of the question (LLM sometimes repeats input)
            if "→" in content:
                content = content.split("→")[-1].strip()
            if ":" in content and len(content.split(":")[0]) < 20:
                content = content.split(":")[-1].strip()
            
            # Parse comma-separated subjects
            raw_subjects = [s.strip() for s in content.split(",") if s.strip()]
            
            if not raw_subjects:
                return None
            
            # Valid academic subjects for filtering garbage
            valid_subjects = {
                "physics", "chemistry", "biology", "mathematics", "math", "maths",
                "computer_science", "history", "geography", "english", "literature",
                "economics", "psychology", "algebra", "geometry", "calculus",
                "trigonometry", "statistics", "mechanics", "thermodynamics",
                "electromagnetism", "optics", "quantum", "gravity", "gravitation",
                "genetics", "ecology", "evolution", "anatomy", "botany", "zoology",
                "organic_chemistry", "inorganic_chemistry", "biochemistry",
                "programming", "data_structures", "algorithms", "networking",
                "operating_systems", "databases", "linux", "python", "java",
                "ancient_history", "modern_history", "world_history", "geography",
                "classical_mechanics", "nuclear_physics", "particle_physics",
                "plant_biology", "cell_biology", "molecular_biology",
                "shell_scripting", "web_development", "machine_learning"
            }
            
            # Normalize to lowercase keys and proper display names
            subjects = []
            display_names = []
            confidences = []
            
            for i, subj in enumerate(raw_subjects[:3]):  # Max 3
                # Create normalized key (lowercase, underscores)
                key = subj.lower().replace(" ", "_").replace("-", "_")
                
                # Skip if it looks like query text (too long or contains question words)
                if len(key) > 30 or any(q in key for q in ["what", "how", "why", "tell", "explain", "about"]):
                    logger.warning(f"[SUBJECT-LLM] Skipping garbage subject: {key[:50]}")
                    continue
                
                subjects.append(key)
                display_names.append(subj.title())  # Proper case
                # Confidence decreases for broader subjects
                conf = 0.95 - (i * 0.05)
                confidences.append(conf)
            
            if not subjects:
                logger.warning("[SUBJECT-LLM] No valid subjects extracted, falling back")
                return None
            
            logger.info(f"[SUBJECT-LLM] Extracted: {subjects} from query: '{query[:50]}...'")
            
            return {
                "subjects": subjects,
                "display_names": display_names,
                "confidences": confidences,
                "primary": subjects[0] if subjects else "general"
            }
            
        except Exception as e:
            logger.error(f"[SUBJECT-LLM] Extraction error: {e}")
            return None
    
    def _default_multi_response(self) -> Dict[str, any]:
        """Default response for multi-subject when classification fails"""
        return {
            "subjects": ["general"],
            "display_names": ["General"],
            "confidences": [1.0],
            "primary": "general"
        }
    
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
