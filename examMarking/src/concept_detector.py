"""
Concept Detector Module
Detects presence and coverage of concepts from knowledge base
"""

from typing import List, Dict, Set
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConceptDetector:
    """Detect concepts in student answers based on knowledge base"""
    
    def __init__(self, knowledge_base_path=None):
        """
        Initialize concept detector
        
        Args:
            knowledge_base_path: Path to knowledge base JSON
        """
        self.knowledge_base = {}
        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)
    
    def load_knowledge_base(self, kb_path: str):
        """
        Load knowledge base from JSON file
        
        Args:
            kb_path: Path to knowledge base JSON
        """
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            logger.info(f"Loaded knowledge base from {kb_path}")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            self.knowledge_base = {}
    
    def get_concepts_for_question(self, question_id: str) -> Dict:
        """
        Get expected concepts for a question
        
        Args:
            question_id: Question identifier
            
        Returns:
            Dictionary of concepts and keywords
        """
        return self.knowledge_base.get("questions", {}).get(question_id, {})
    
    def detect_concepts(self, student_tokens: List[str], 
                       expected_concepts: Dict) -> Dict:
        """
        Detect which concepts are present in student answer
        
        Args:
            student_tokens: Lemmatized tokens from student answer
            expected_concepts: Expected concepts with keywords
            
        Returns:
            Concept detection results
        """
        student_tokens_set = set(token.lower() for token in student_tokens)
        
        detected = []
        missing = []
        partial = []
        
        for concept_name, concept_data in expected_concepts.items():
            keywords = concept_data.get("keywords", [])
            weight = concept_data.get("weight", 1.0)
            is_critical = concept_data.get("critical", False)
            
            # Count how many keywords are present
            keywords_found = []
            for kw in keywords:
                if kw.lower() in student_tokens_set:
                    keywords_found.append(kw)
            
            coverage = len(keywords_found) / len(keywords) if keywords else 0
            
            concept_result = {
                "concept": concept_name,
                "weight": weight,
                "critical": is_critical,
                "keywords_expected": keywords,
                "keywords_found": keywords_found,
                "coverage": coverage
            }
            
            if coverage >= 0.7:  # 70% threshold for detection
                detected.append(concept_result)
            elif coverage > 0.2:  # Partial understanding
                partial.append(concept_result)
            else:
                missing.append(concept_result)
        
        # Calculate overall concept coverage
        total_concepts = len(expected_concepts)
        detected_count = len(detected)
        partial_count = len(partial)
        
        # Weighted score
        overall_score = (detected_count + 0.5 * partial_count) / total_concepts if total_concepts else 0
        
        return {
            "detected_concepts": detected,
            "partial_concepts": partial,
            "missing_concepts": missing,
            "total_concepts": total_concepts,
            "detected_count": detected_count,
            "missing_count": len(missing),
            "overall_coverage": overall_score,
            "critical_concepts_missing": [c for c in missing if c["critical"]]
        }
    
    def detect_topics(self, detected_concepts: List[Dict]) -> Dict:
        """
        Identify topics based on detected concepts
        
        Args:
            detected_concepts: List of detected concept results
            
        Returns:
            Topic analysis
        """
        topic_coverage = {}
        
        # Map concepts to topics from knowledge base
        for concept_result in detected_concepts:
            concept_name = concept_result["concept"]
            
            # Find which topic this concept belongs to
            for topic_name, topic_data in self.knowledge_base.get("topics", {}).items():
                if concept_name in topic_data.get("concepts", {}):
                    if topic_name not in topic_coverage:
                        topic_coverage[topic_name] = []
                    topic_coverage[topic_name].append(concept_result)
        
        return topic_coverage
    
    def calculate_concept_score(self, detection_results: Dict, 
                               penalty_per_critical=0.2) -> float:
        """
        Calculate score based on concept coverage
        
        Args:
            detection_results: Results from detect_concepts
            penalty_per_critical: Penalty for each missing critical concept
            
        Returns:
            Concept score (0-1)
        """
        base_score = detection_results["overall_coverage"]
        
        # Apply penalty for missing critical concepts
        critical_missing = len(detection_results["critical_concepts_missing"])
        penalty = critical_missing * penalty_per_critical
        
        final_score = max(0.0, base_score - penalty)
        return final_score
