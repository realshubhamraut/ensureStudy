"""
Chat Producer - Publish chat messages to Kafka for async processing
"""
from datetime import datetime
from typing import Dict, Any, Optional, List

from backend.kafka.config.kafka_config import create_producer, Topics


class ChatProducer:
    """
    Produces chat messages to Kafka for:
    - Async moderation
    - Analytics
    - Agent processing
    """
    
    def __init__(self):
        self.producer = create_producer()
        self.topic = Topics.CHAT_MESSAGES
    
    def publish_message(
        self,
        user_id: str,
        session_id: str,
        message: str,
        role: str = "user",
        sources: Optional[List[Dict]] = None
    ) -> None:
        """Publish a chat message"""
        event = {
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "role": role,
            "sources": sources or [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.producer.send(
            self.topic,
            key=session_id,
            value=event
        )
        self.producer.flush()
    
    def publish_conversation(
        self,
        user_id: str,
        session_id: str,
        messages: List[Dict[str, str]]
    ) -> None:
        """Publish entire conversation for analysis"""
        event = {
            "user_id": user_id,
            "session_id": session_id,
            "messages": messages,
            "message_count": len(messages),
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "conversation_complete"
        }
        
        self.producer.send(
            self.topic,
            key=session_id,
            value=event
        )
        self.producer.flush()
    
    def close(self):
        """Close the producer"""
        self.producer.close()
