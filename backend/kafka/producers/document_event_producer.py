"""
Document Event Producer - Publish document upload events to Kafka

Triggers the document processing pipeline when a document is uploaded.
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from kafka import KafkaProducer

from backend.kafka.config.kafka_config import create_producer, Topics


class DocumentEventProducer:
    """
    Produces document processing events to Kafka.
    
    Event types:
    - document_uploaded: New document needs processing
    - document_reprocess: Document needs reprocessing
    """
    
    def __init__(self):
        self.producer = create_producer()
        self.topic = Topics.DOCUMENT_PROCESSING
    
    def _create_event(
        self,
        event_type: str,
        document_id: str,
        data: Dict[str, Any],
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a standardized event payload"""
        return {
            "event_type": event_type,
            "document_id": document_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
            "metadata": metadata or {},
            "source": "core-service"
        }
    
    def publish_document_uploaded(
        self,
        document_id: str,
        student_id: str,
        classroom_id: str,
        source_url: str,
        file_type: str,
        title: Optional[str] = None
    ) -> None:
        """
        Publish document upload event to trigger processing pipeline.
        
        Args:
            document_id: Unique document/job ID
            student_id: Owner student ID
            classroom_id: Associated classroom
            source_url: S3 or local path to document
            file_type: pdf, pptx, docx, etc.
            title: Optional document title
        """
        event = self._create_event(
            event_type="document_uploaded",
            document_id=document_id,
            data={
                "student_id": student_id,
                "classroom_id": classroom_id,
                "source_url": source_url,
                "file_type": file_type,
                "title": title
            }
        )
        
        # Use document_id as key for message ordering and idempotency
        self.producer.send(
            self.topic,
            key=document_id,
            value=event
        )
        self.producer.flush()
        
        print(f"✓ Published document_uploaded event for {document_id}")
    
    def publish_reprocess(
        self,
        document_id: str,
        reason: str = "manual"
    ) -> None:
        """Trigger reprocessing of an existing document"""
        event = self._create_event(
            event_type="document_reprocess",
            document_id=document_id,
            data={"reason": reason}
        )
        
        self.producer.send(
            self.topic,
            key=document_id,
            value=event
        )
        self.producer.flush()
    
    def close(self):
        """Close the producer"""
        self.producer.close()


# Singleton instance
_producer: Optional[DocumentEventProducer] = None


def get_document_event_producer() -> DocumentEventProducer:
    """Get singleton producer instance"""
    global _producer
    if _producer is None:
        _producer = DocumentEventProducer()
    return _producer


def publish_document_for_processing(
    document_id: str,
    student_id: str,
    classroom_id: str,
    source_url: str,
    file_type: str,
    title: Optional[str] = None
) -> None:
    """
    Convenience function to publish a document for processing.
    
    Call this from the upload API after storing the document.
    """
    producer = get_document_event_producer()
    producer.publish_document_uploaded(
        document_id=document_id,
        student_id=student_id,
        classroom_id=classroom_id,
        source_url=source_url,
        file_type=file_type,
        title=title
    )
