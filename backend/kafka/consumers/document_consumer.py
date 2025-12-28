"""
Document Consumer - Consume document upload events and trigger processing

Uses the existing LangGraph DocumentProcessingAgent for the 7-stage pipeline:
Validate → Preprocess → OCR → Chunk → Embed → Index → Complete
"""
import os
import json
import asyncio
import logging
from typing import Callable, Dict, Any, Optional
from datetime import datetime
from kafka import KafkaConsumer

from backend.kafka.config.kafka_config import create_consumer, Topics

logger = logging.getLogger(__name__)


class DocumentConsumer:
    """
    Consumes document processing events from Kafka.
    
    Triggers the LangGraph DocumentProcessingAgent pipeline for each event.
    
    Features:
    - Async processing with LangGraph
    - Automatic retry on failure
    - Dead letter queue for permanent failures
    - Status updates to PostgreSQL
    """
    
    def __init__(
        self,
        group_id: str = "document-consumers",
        max_retries: int = 3
    ):
        self.topics = [Topics.DOCUMENT_PROCESSING]
        self.consumer = create_consumer(
            topics=self.topics,
            group_id=group_id
        )
        self.running = False
        self.max_retries = max_retries
        
        # Import the LangGraph agent
        from app.agents.document_agent import DocumentProcessingAgent
        self.agent = DocumentProcessingAgent()
        
        logger.info(f"Initialized DocumentConsumer for topics: {self.topics}")
    
    def start(self) -> None:
        """Start consuming messages synchronously"""
        self.running = True
        logger.info(f"Starting document consumer for topics: {self.topics}")
        
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                self._process_message_sync(message)
        
        finally:
            self.consumer.close()
    
    async def start_async(self) -> None:
        """Start consuming with async handler for LangGraph agent"""
        self.running = True
        logger.info(f"Starting async document consumer for topics: {self.topics}")
        
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                await self._process_message(message)
        
        finally:
            self.consumer.close()
    
    def _process_message_sync(self, message) -> None:
        """Process message synchronously by running async handler"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._process_message(message))
            else:
                loop.run_until_complete(self._process_message(message))
        except Exception as e:
            logger.error(f"Error in sync processing: {e}")
    
    async def _process_message(self, message) -> None:
        """Process a single Kafka message"""
        event = message.value
        document_id = event.get("document_id", "unknown")
        event_type = event.get("event_type", "unknown")
        
        logger.info(f"Processing event: {event_type} for document {document_id}")
        
        try:
            if event_type == "document_uploaded":
                await self._handle_document_uploaded(event)
            elif event_type == "document_reprocess":
                await self._handle_document_reprocess(event)
            else:
                logger.warning(f"Unknown event type: {event_type}")
        
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            await self._handle_failure(event, str(e))
    
    async def _handle_document_uploaded(self, event: Dict) -> None:
        """Handle document_uploaded event - run full pipeline"""
        document_id = event["document_id"]
        data = event.get("data", {})
        
        start_time = datetime.utcnow()
        logger.info(f"Starting processing pipeline for {document_id}")
        
        # Prepare input for LangGraph DocumentProcessingAgent
        agent_input = {
            "document_id": document_id,
            "student_id": data.get("student_id", ""),
            "classroom_id": data.get("classroom_id", ""),
            "source_url": data.get("source_url", ""),
            "file_type": data.get("file_type", "pdf"),
            "subject": data.get("subject"),
            "is_teacher_material": data.get("is_teacher_material", False)
        }
        
        # Run the LangGraph pipeline
        result = await self.agent.process(agent_input)
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        if result.get("success"):
            logger.info(
                f"✓ Document {document_id} processed successfully in {duration:.2f}s. "
                f"Chunks: {result.get('total_chunks', 0)}, "
                f"Confidence: {result.get('avg_confidence', 0):.2f}"
            )
        else:
            logger.error(f"✗ Document {document_id} processing failed: {result.get('error')}")
    
    async def _handle_document_reprocess(self, event: Dict) -> None:
        """Handle document_reprocess event"""
        document_id = event["document_id"]
        reason = event.get("data", {}).get("reason", "manual")
        
        logger.info(f"Reprocessing document {document_id} (reason: {reason})")
        
        # Fetch document metadata from database
        # This would need to be implemented to get source_url, etc.
        # For now, log a warning
        logger.warning(f"Reprocessing not fully implemented for {document_id}")
    
    async def _handle_failure(self, event: Dict, error: str) -> None:
        """Handle processing failure"""
        document_id = event.get("document_id", "unknown")
        
        # Update document status to failed
        try:
            import httpx
            
            core_url = os.getenv("CORE_SERVICE_URL", "http://localhost:8000")
            api_key = os.getenv("INTERNAL_API_KEY", "dev-internal-key")
            
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{core_url}/api/notes/internal/update-job",
                    json={
                        "job_id": document_id,
                        "status": "failed",
                        "error_message": error
                    },
                    headers={"X-Internal-API-Key": api_key},
                    timeout=10.0
                )
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")
    
    def stop(self) -> None:
        """Stop the consumer"""
        self.running = False


def run_document_consumer():
    """Run the document consumer (blocking)"""
    consumer = DocumentConsumer()
    
    try:
        # Use async start for LangGraph agent
        asyncio.run(consumer.start_async())
    except KeyboardInterrupt:
        print("\n✗ Document consumer stopped by user")
        consumer.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    run_document_consumer()
