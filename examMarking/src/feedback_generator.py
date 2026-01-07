"""
Feedback Generator Module
Generates constructive feedback for students
"""

from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackGenerator:
    """Generate human-readable feedback from evaluation results"""
    
    def __init__(self):
        self.grade_thresholds = {
            90: ("Excellent", "Outstanding understanding"),
            80: ("Very Good", "Strong grasp of concepts"),
            70: ("Good", "Good understanding with minor gaps"),
            60: ("Satisfactory", "Adequate understanding"),
            50: ("Pass", "Basic understanding demonstrated"),
            40: ("Weak", "Significant gaps in understanding"),
            0: ("Poor", "Major concepts missing")
        }
    
    def get_grade(self, percentage: float) -> tuple:
        """
        Get grade and description based on percentage
        
        Args:
            percentage: Score percentage (0-100)
            
        Returns:
            Tuple of (grade, description)
        """
        for threshold in sorted(self.grade_thresholds.keys(), reverse=True):
            if percentage >= threshold:
                return self.grade_thresholds[threshold]
        return ("Fail", "Needs significant improvement")
    
    def generate_feedback(self, evaluation_result: Dict, verbose=True) -> Dict:
        """
        Generate comprehensive feedback
        
        Args:
            evaluation_result: Results from EvaluationEngine
            verbose: Include detailed analysis if True
            
        Returns:
            Feedback dictionary
        """
        marks = evaluation_result["marks_obtained"]
        max_marks = evaluation_result["max_marks"]
        percentage = evaluation_result["percentage"]
        scores = evaluation_result["scores"]
        
        grade, grade_desc = self.get_grade(percentage)
        
        # Strengths and weaknesses
        strengths = []
        weaknesses = []
        suggestions = []
        
        # Analyze semantic performance
        semantic_details = scores["semantic"]["details"]
        if scores["semantic"]["score"] >= 0.75:
            strengths.append("Good semantic understanding of the topic")
        elif scores["semantic"]["score"] < 0.5:
            weaknesses.append("Conceptual understanding needs improvement")
            suggestions.append("Review the core concepts and their relationships")
        
        # Analyze keyword coverage
        keyword_details = scores["keyword"]["details"]
        matched_kws = keyword_details.get("matched_keywords", [])
        missed_kws = keyword_details.get("missed_keywords", [])
        
        if scores["keyword"]["score"] >= 0.7:
            strengths.append(f"Good coverage of key terms ({len(matched_kws)} key terms mentioned)")
        else:
            weaknesses.append(f"Missing important keywords: {', '.join(missed_kws[:5])}")
            if missed_kws:
                suggestions.append(f"Include key terms like: {', '.join(missed_kws[:3])}")
        
        # Analyze concept coverage
        concept_details = scores["concept"].get("details", {})
        if concept_details:
            detected = concept_details.get("detected_concepts", [])
            missing = concept_details.get("missing_concepts", [])
            critical_missing = concept_details.get("critical_concepts_missing", [])
            
            if detected:
                detected_names = [c["concept"] for c in detected]
                strengths.append(f"Covered concepts: {', '.join(detected_names)}")
            
            if critical_missing:
                critical_names = [c["concept"] for c in critical_missing]
                weaknesses.append(f"Missing critical concepts: {', '.join(critical_names)}")
                suggestions.append(f"Study these important topics: {', '.join(critical_names)}")
            elif missing:
                missing_names = [c["concept"] for c in missing[:3]]
                weaknesses.append(f"Some concepts not covered: {', '.join(missing_names)}")
        
        # Mark deduction explanation
        marks_lost = max_marks - marks
        deductions = []
        
        if scores["semantic"]["score"] < 1.0:
            semantic_loss = (1 - scores["semantic"]["score"]) * scores["semantic"]["weight"] * max_marks
            deductions.append({
                "reason": "Semantic similarity below expected level",
                "marks_deducted": round(semantic_loss, 2)
            })
        
        if scores["keyword"]["score"] < 1.0:
            keyword_loss = (1 - scores["keyword"]["score"]) * scores["keyword"]["weight"] * max_marks
            deductions.append({
                "reason": f"Missing key terms: {', '.join(missed_kws[:3])}",
                "marks_deducted": round(keyword_loss, 2)
            })
        
        if concept_details and concept_details.get("missing_concepts"):
            concept_loss = (1 - scores["concept"]["score"]) * scores["concept"]["weight"] * max_marks
            deductions.append({
                "reason": "Incomplete concept coverage",
                "marks_deducted": round(concept_loss, 2)
            })
        
        # Build feedback
        feedback = {
            "summary": {
                "marks_obtained": round(marks, 2),
                "max_marks": max_marks,
                "percentage": round(percentage, 1),
                "grade": grade,
                "grade_description": grade_desc
            },
            "strengths": strengths if strengths else ["Answer demonstrates basic attempt"],
            "weaknesses": weaknesses if weaknesses else ["No major weaknesses identified"],
            "suggestions": suggestions if suggestions else ["Continue with good work"],
            "deductions": deductions,
            "detailed_scores": {
                "semantic_similarity": f"{scores['semantic']['score']*100:.1f}%",
                "keyword_matching": f"{scores['keyword']['score']*100:.1f}%",
                "concept_coverage": f"{scores['concept']['score']*100:.1f}%"
            }
        }
        
        if verbose:
            feedback["verbose_analysis"] = {
                "semantic_details": semantic_details,
                "keyword_match_details": keyword_details,
                "concept_detection": concept_details
            }
        
        return feedback
    
    def format_feedback_text(self, feedback: Dict) -> str:
        """
        Format feedback as readable text
        
        Args:
            feedback: Feedback dictionary
            
        Returns:
            Formatted text string
        """
        summary = feedback["summary"]
        
        text = f"""
{'='*70}
EVALUATION FEEDBACK
{'='*70}

SCORE: {summary['marks_obtained']}/{summary['max_marks']} ({summary['percentage']}%)
GRADE: {summary['grade']} - {summary['grade_description']}

{'='*70}
STRENGTHS:
{'='*70}
"""
        for i, strength in enumerate(feedback["strengths"], 1):
            text += f"  ✓ {strength}\n"
        
        text += f"""
{'='*70}
AREAS FOR IMPROVEMENT:
{'='*70}
"""
        for i, weakness in enumerate(feedback["weaknesses"], 1):
            text += f"  ✗ {weakness}\n"
        
        text += f"""
{'='*70}
MARK DEDUCTIONS:
{'='*70}
"""
        for deduction in feedback["deductions"]:
            text += f"  - {deduction['reason']}: -{deduction['marks_deducted']} marks\n"
        
        text += f"""
{'='*70}
RECOMMENDATIONS:
{'='*70}
"""
        for i, suggestion in enumerate(feedback["suggestions"], 1):
            text += f"  {i}. {suggestion}\n"
        
        text += f"""
{'='*70}
DETAILED BREAKDOWN:
{'='*70}
  • Semantic Similarity: {feedback['detailed_scores']['semantic_similarity']}
  • Keyword Matching: {feedback['detailed_scores']['keyword_matching']}
  • Concept Coverage: {feedback['detailed_scores']['concept_coverage']}
{'='*70}
"""
        
        return text
