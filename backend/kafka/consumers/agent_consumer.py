"""
Agent Consumer - Consume events for MCP agent processing
"""
import os
import json
import asyncio
from typing import Callable, Dict, Any
from kafka import KafkaConsumer

from backend.kafka.config.kafka_config import create_consumer, Topics


class AgentConsumer:
    """
    Consumes events for agent processing via MCP.
    
    Handles:
    - Chat messages for moderation
    - Student events for context building
    - Assessment submissions for adaptive learning
    """
    
    def __init__(self, group_id: str = "agent-consumers"):
        self.topics = [
            Topics.CHAT_MESSAGES,
            Topics.STUDENT_EVENTS
        ]
        self.consumer = create_consumer(
            topics=self.topics,
            group_id=group_id
        )
        self.handlers: Dict[str, Callable] = {}
        self.running = False
    
    def register_handler(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Register a handler for a specific event type"""
        self.handlers[event_type] = handler
    
    def start(self) -> None:
        """Start consuming messages"""
        self.running = True
        print(f"Starting agent consumer for topics: {self.topics}")
        
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                try:
                    event = message.value
                    event_type = event.get("event_type", "unknown")
                    
                    # Route to handler
                    handler = self.handlers.get(event_type)
                    if handler:
                        handler(event)
                    else:
                        print(f"No handler for event type: {event_type}")
                
                except Exception as e:
                    print(f"Error processing message: {e}")
        
        finally:
            self.consumer.close()
    
    def stop(self) -> None:
        """Stop the consumer"""
        self.running = False
    
    async def start_async(self) -> None:
        """Start consuming with async handlers"""
        self.running = True
        print(f"Starting async agent consumer for topics: {self.topics}")
        
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                try:
                    event = message.value
                    event_type = event.get("event_type", "unknown")
                    
                    handler = self.handlers.get(event_type)
                    if handler:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                
                except Exception as e:
                    print(f"Error processing message: {e}")
        
        finally:
            self.consumer.close()


def run_agent_consumer():
    """Run the agent consumer with default handlers"""
    consumer = AgentConsumer()
    
    # Register handlers
    def handle_chat_message(event: Dict):
        """Handle incoming chat messages"""
        user_id = event.get("user_id")
        message = event.get("message")
        print(f"Chat from {user_id}: {message[:50]}...")
        # Here you would trigger moderation agent
    
    def handle_question(event: Dict):
        """Handle question asked events"""
        user_id = event.get("user_id")
        data = event.get("data", {})
        question = data.get("question")
        print(f"Question from {user_id}: {question[:50]}...")
    
    consumer.register_handler("chat_message", handle_chat_message)
    consumer.register_handler("question_asked", handle_question)
    
    # Start consuming
    consumer.start()


if __name__ == "__main__":
    run_agent_consumer()
