"""
Assessment Producer - Publish assessment events to Kafka
"""
from datetime import datetime
from typing import Dict, Any, Optional, List

from backend.kafka.config.kafka_config import create_producer, Topics


class AssessmentProducer:
    """Produces assessment-related events to Kafka"""
    
    def __init__(self):
        self.producer = create_producer()
        self.topic = Topics.ASSESSMENT_SUBMISSIONS
    
    def publish_submission(
        self,
        user_id: str,
        assessment_id: str,
        score: float,
        time_taken_seconds: int,
        topic: str,
        subject: str,
        answers: Dict[str, Any]
    ) -> None:
        """Publish assessment submission"""
        event = {
            "event_type": "submission",
            "user_id": user_id,
            "assessment_id": assessment_id,
            "score": score,
            "time_taken_seconds": time_taken_seconds,
            "topic": topic,
            "subject": subject,
            "answers": answers,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.producer.send(
            self.topic,
            key=user_id,
            value=event
        )
        self.producer.flush()
    
    def publish_started(
        self,
        user_id: str,
        assessment_id: str,
        topic: str,
        subject: str
    ) -> None:
        """Publish assessment started event"""
        event = {
            "event_type": "started",
            "user_id": user_id,
            "assessment_id": assessment_id,
            "topic": topic,
            "subject": subject,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.producer.send(
            self.topic,
            key=user_id,
            value=event
        )
        self.producer.flush()
    
    def close(self):
        """Close the producer"""
        self.producer.close()
