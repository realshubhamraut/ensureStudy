"""
Student Event Producer - Publish student interaction events to Kafka
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from kafka import KafkaProducer

from backend.kafka.config.kafka_config import create_producer, Topics


class StudentEventProducer:
    """
    Produces student interaction events to Kafka.
    
    Event types:
    - page_view: Student viewed a page
    - study_session_start/end: Study session tracking
    - note_view: Viewed study notes
    - question_asked: Asked a question to tutor
    - assessment_started: Started an assessment
    """
    
    def __init__(self):
        self.producer = create_producer()
        self.topic = Topics.STUDENT_EVENTS
    
    def _create_event(
        self,
        event_type: str,
        user_id: str,
        data: Dict[str, Any],
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a standardized event payload"""
        return {
            "event_type": event_type,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
            "metadata": metadata or {},
            "source": "web"
        }
    
    def publish_page_view(
        self,
        user_id: str,
        page: str,
        subject: Optional[str] = None,
        topic: Optional[str] = None
    ) -> None:
        """Publish a page view event"""
        event = self._create_event(
            event_type="page_view",
            user_id=user_id,
            data={
                "page": page,
                "subject": subject,
                "topic": topic
            }
        )
        
        self.producer.send(
            self.topic,
            key=user_id,
            value=event
        )
        self.producer.flush()
    
    def publish_study_session_start(
        self,
        user_id: str,
        subject: str,
        topic: str
    ) -> str:
        """Publish study session start event"""
        session_id = f"{user_id}_{datetime.utcnow().timestamp()}"
        
        event = self._create_event(
            event_type="study_session_start",
            user_id=user_id,
            data={
                "session_id": session_id,
                "subject": subject,
                "topic": topic
            }
        )
        
        self.producer.send(self.topic, key=user_id, value=event)
        self.producer.flush()
        
        return session_id
    
    def publish_study_session_end(
        self,
        user_id: str,
        session_id: str,
        duration_seconds: int,
        pages_read: int = 0
    ) -> None:
        """Publish study session end event"""
        event = self._create_event(
            event_type="study_session_end",
            user_id=user_id,
            data={
                "session_id": session_id,
                "duration_seconds": duration_seconds,
                "pages_read": pages_read
            }
        )
        
        self.producer.send(self.topic, key=user_id, value=event)
        self.producer.flush()
    
    def publish_question_asked(
        self,
        user_id: str,
        question: str,
        subject: Optional[str] = None,
        was_answered: bool = True
    ) -> None:
        """Publish question asked event"""
        event = self._create_event(
            event_type="question_asked",
            user_id=user_id,
            data={
                "question": question[:500],  # Truncate long questions
                "subject": subject,
                "was_answered": was_answered
            }
        )
        
        self.producer.send(self.topic, key=user_id, value=event)
        self.producer.flush()
    
    def close(self):
        """Close the producer"""
        self.producer.close()


# Singleton instance
_producer: Optional[StudentEventProducer] = None


def get_student_event_producer() -> StudentEventProducer:
    """Get singleton producer instance"""
    global _producer
    if _producer is None:
        _producer = StudentEventProducer()
    return _producer
