"""
Moderation Agent - Fast Zero-Shot Classification
Uses Hugging Face classifier instead of LLM for speed
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


# Classification labels
ACADEMIC_LABELS = ["academic", "homework", "study", "education", "learning"]
INAPPROPRIATE_LABELS = ["inappropriate", "harmful", "offensive", "spam"]


class ModerationAgent:
    """
    Fast moderation using zero-shot classification
    
    No LLM needed - uses facebook/bart-large-mnli for:
    - Academic content detection
    - Inappropriate content filtering
    - Sub-second response times
    """
    
    def __init__(self):
        self._classifier = None
        logger.info("Initialized Moderation Agent")
    
    @property
    def classifier(self):
        """Lazy load classifier"""
        if self._classifier is None:
            from app.services.llm_provider import get_classifier
            self._classifier = get_classifier()
        return self._classifier
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Moderate content
        
        Args:
            input_data: {message, user_id}
        
        Returns:
            {is_appropriate, is_academic, confidence, reason}
        """
        message = input_data.get("message", "")
        user_id = input_data.get("user_id", "")
        
        if not message:
            return self._format_output(True, True, 1.0, "empty_message")
        
        try:
            # Classify for academic content
            academic_scores = self.classifier.classify(
                message, 
                ACADEMIC_LABELS + INAPPROPRIATE_LABELS
            )
            
            # Calculate academic score
            academic_score = sum(
                academic_scores.get(label, 0) for label in ACADEMIC_LABELS
            ) / len(ACADEMIC_LABELS)
            
            # Calculate inappropriate score
            inappropriate_score = sum(
                academic_scores.get(label, 0) for label in INAPPROPRIATE_LABELS
            ) / len(INAPPROPRIATE_LABELS)
            
            # Determine if appropriate
            is_academic = academic_score > 0.2
            is_appropriate = inappropriate_score < 0.3
            
            # Overall confidence
            confidence = max(academic_score, 1 - inappropriate_score)
            
            # Reason
            if not is_appropriate:
                reason = "inappropriate_content"
            elif not is_academic:
                reason = "non_academic"
            else:
                reason = "allowed"
            
            was_blocked = not (is_academic and is_appropriate)
            
            logger.info(
                f"Moderation: academic={academic_score:.2f}, "
                f"inappropriate={inappropriate_score:.2f}, blocked={was_blocked}"
            )
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "moderation",
                "data": {
                    "is_appropriate": is_appropriate,
                    "is_academic": is_academic,
                    "was_blocked": was_blocked,
                    "confidence": confidence,
                    "reason": reason,
                    "scores": {
                        "academic": academic_score,
                        "inappropriate": inappropriate_score
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Moderation error: {e}")
            # Default to allow on error (fail open for better UX)
            return self._format_output(True, True, 0.5, f"error: {e}")
    
    def _format_output(
        self, 
        is_appropriate: bool, 
        is_academic: bool, 
        confidence: float,
        reason: str
    ) -> Dict[str, Any]:
        """Format moderation output"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": "moderation",
            "data": {
                "is_appropriate": is_appropriate,
                "is_academic": is_academic,
                "was_blocked": not (is_appropriate and is_academic),
                "confidence": confidence,
                "reason": reason
            }
        }
