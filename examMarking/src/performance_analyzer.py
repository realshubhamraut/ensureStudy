"""
Performance Analyzer Module
Tracks student performance across topics and questions
"""

from typing import List, Dict
import json
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Analyze student performance across multiple evaluations"""
    
    def __init__(self, strong_threshold=80, weak_threshold=60):
        """
        Initialize performance analyzer
        
        Args:
            strong_threshold: Percentage above which performance is strong
            weak_threshold: Percentage below which performance is weak
        """
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold
    
    def analyze_single_exam(self, evaluation_results: List[Dict]) -> Dict:
        """
        Analyze performance for a single exam with multiple questions
        
        Args:
            evaluation_results: List of evaluation results from EvaluationEngine
            
        Returns:
            Performance analysis
        """
        if not evaluation_results:
            return {"error": "No evaluation results provided"}
        
        total_marks = sum(r["max_marks"] for r in evaluation_results)
        obtained_marks = sum(r["marks_obtained"] for r in evaluation_results)
        percentage = (obtained_marks / total_marks * 100) if total_marks > 0 else 0
        
        # Component-wise analysis
        semantic_scores = [r["scores"]["semantic"]["score"] for r in evaluation_results]
        keyword_scores = [r["scores"]["keyword"]["score"] for r in evaluation_results]
        concept_scores = [r["scores"]["concept"]["score"] for r in evaluation_results]
        
        # Topic analysis
        topic_performance = defaultdict(list)
        concept_coverage = defaultdict(list)
        
        for result in evaluation_results:
            concept_details = result["scores"]["concept"].get("details", {})
            
            # Collect detected concepts
            for concept in concept_details.get("detected_concepts", []):
                concept_name = concept["concept"]
                concept_coverage[concept_name].append(concept["coverage"])
            
            # Collect missing concepts
            for concept in concept_details.get("missing_concepts", []):
                concept_name = concept["concept"]
                concept_coverage[concept_name].append(0.0)
        
        # Identify weak areas
        weak_concepts = []
        strong_concepts = []
        
        for concept, coverages in concept_coverage.items():
            avg_coverage = sum(coverages) / len(coverages) * 100
            if avg_coverage < self.weak_threshold:
                weak_concepts.append({
                    "concept": concept,
                    "average_coverage": round(avg_coverage, 1)
                })
            elif avg_coverage >= self.strong_threshold:
                strong_concepts.append({
                    "concept": concept,
                    "average_coverage": round(avg_coverage, 1)
                })
        
        return {
            "overall_performance": {
                "total_marks": total_marks,
                "obtained_marks": round(obtained_marks, 2),
                "percentage": round(percentage, 1),
                "grade": self._get_grade(percentage)
            },
            "component_performance": {
                "semantic_avg": round(sum(semantic_scores) / len(semantic_scores) * 100, 1),
                "keyword_avg": round(sum(keyword_scores) / len(keyword_scores) * 100, 1),
                "concept_avg": round(sum(concept_scores) / len(concept_scores) * 100, 1)
            },
            "strong_areas": strong_concepts,
            "weak_areas": weak_concepts,
            "question_wise_performance": [
                {
                    "question": i + 1,
                    "marks": round(r["marks_obtained"], 2),
                    "max_marks": r["max_marks"],
                    "percentage": round(r["percentage"], 1)
                }
                for i, r in enumerate(evaluation_results)
            ]
        }
    
    def generate_student_profile(self, performance_analysis: Dict, 
                                student_id: str = None) -> Dict:
        """
        Generate student performance profile
        
        Args:
            performance_analysis: Results from analyze_single_exam
            student_id: Student identifier
            
        Returns:
            Student profile with recommendations
        """
        overall = performance_analysis["overall_performance"]
        components = performance_analysis["component_performance"]
        weak_areas = performance_analysis["weak_areas"]
        strong_areas = performance_analysis["strong_areas"]
        
        # Generate recommendations
        recommendations = []
        
        if components["semantic_avg"] < self.weak_threshold:
            recommendations.append({
                "area": "Conceptual Understanding",
                "priority": "High",
                "suggestion": "Focus on understanding core concepts rather than memorization"
            })
        
        if components["keyword_avg"] < self.weak_threshold:
            recommendations.append({
                "area": "Technical Terminology",
                "priority": "Medium",
                "suggestion": "Improve use of subject-specific vocabulary and key terms"
            })
        
        if components["concept_avg"] < self.weak_threshold:
            recommendations.append({
                "area": "Concept Coverage",
                "priority": "High",
                "suggestion": "Ensure all aspects of questions are addressed comprehensively"
            })
        
        # Topic-specific recommendations
        if weak_areas:
            for weak in weak_areas[:3]:  # Top 3 weak concepts
                recommendations.append({
                    "area": weak["concept"],
                    "priority": "High",
                    "suggestion": f"Dedicate extra study time to {weak['concept']}"
                })
        
        profile = {
            "student_id": student_id or "Unknown",
            "overall_grade": overall["grade"],
            "percentage": overall["percentage"],
            "performance_summary": self._get_performance_summary(overall["percentage"]),
            "strengths": [s["concept"] for s in strong_areas] if strong_areas else ["Basic understanding demonstrated"],
            "weaknesses": [w["concept"] for w in weak_areas] if weak_areas else ["No major weaknesses"],
            "recommendations": recommendations,
            "component_breakdown": components
        }
        
        return profile
    
    def _get_grade(self, percentage: float) -> str:
        """Get letter grade from percentage"""
        if percentage >= 90:
            return "A+"
        elif percentage >= 80:
            return "A"
        elif percentage >= 70:
            return "B"
        elif percentage >= 60:
            return "C"
        elif percentage >= 50:
            return "D"
        else:
            return "F"
    
    def _get_performance_summary(self, percentage: float) -> str:
        """Get performance summary text"""
        if percentage >= self.strong_threshold:
            return "Excellent performance. Strong understanding demonstrated across topics."
        elif percentage >= self.weak_threshold:
            return "Good performance. Some areas need attention."
        else:
            return "Needs improvement. Significant study required in multiple areas."
    
    def export_report(self, student_profile: Dict, output_path: str):
        """
        Export student profile to JSON
        
        Args:
            student_profile: Student profile dictionary
            output_path: Path to save JSON file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(student_profile, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported student profile to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
