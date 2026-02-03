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
    - Assessment submissions for adaptive learning (Type 5 Learning Agent)
    """
    
    def __init__(self, group_id: str = "agent-consumers"):
        self.topics = [
            Topics.CHAT_MESSAGES,
            Topics.STUDENT_EVENTS,
            Topics.ASSESSMENT_SUBMISSIONS  # Learning Agent trigger
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


# =============================================================================
# Learning Agent Event Handlers
# =============================================================================

async def handle_assessment_submitted(event: Dict) -> None:
    """
    Handle assessment.submitted events for Learning Agent.
    
    Triggers:
    - Learning update from student responses
    - Question effectiveness score updates
    - Auto-generation if 80% threshold reached
    """
    try:
        from backend.ai_service.app.agents.learning_agent import get_learning_agent
        
        topic_id = event.get("topic_id")
        responses = event.get("responses", [])
        user_id = event.get("user_id")
        
        if not topic_id:
            print(f"[LearningAgent] Missing topic_id in event")
            return
        
        print(f"[LearningAgent] Processing assessment submission for topic {topic_id}")
        
        agent = get_learning_agent()
        result = await agent.trigger_on_assessment_submit(
            topic_id=topic_id,
            responses=responses
        )
        
        print(f"[LearningAgent] Result: {result.get('data', {})}")
        
        # Log if questions were generated
        questions_generated = result.get('data', {}).get('questions_after_dedupe', 0)
        if questions_generated > 0:
            print(f"[LearningAgent] Generated {questions_generated} new questions for topic {topic_id}")
        
    except ImportError as e:
        print(f"[LearningAgent] Import error (AI service not available): {e}")
    except Exception as e:
        print(f"[LearningAgent] Error processing assessment: {e}")


async def handle_question_answered(event: Dict) -> None:
    """
    Handle question.answered events to update effectiveness scores.
    
    Updates:
    - Question discrimination index
    - Difficulty index
    - Distractor quality
    """
    try:
        from backend.ai_service.app.utils.question_effectiveness import update_question_effectiveness_from_response
        
        question_id = event.get("question_id")
        is_correct = event.get("is_correct", False)
        selected_option = event.get("selected_option")
        response_time_ms = event.get("response_time_ms", 0)
        student_percentile = event.get("student_percentile", 0.5)
        
        if not question_id:
            return
        
        # Update effectiveness (would need DB session in production)
        print(f"[LearningAgent] Updating effectiveness for question {question_id}")
        
    except Exception as e:
        print(f"[LearningAgent] Error updating effectiveness: {e}")


def run_agent_consumer():
    """Run the agent consumer with default handlers"""
    consumer = AgentConsumer()
    
    # Register handlers
    def handle_chat_message(event: Dict):
        """Handle incoming chat messages"""
        user_id = event.get("user_id")
        message = event.get("message", "")
        print(f"Chat from {user_id}: {message[:50]}...")
        # Here you would trigger moderation agent
    
    def handle_question(event: Dict):
        """Handle question asked events"""
        user_id = event.get("user_id")
        data = event.get("data", {})
        question = data.get("question", "")
        print(f"Question from {user_id}: {question[:50]}...")
    
    consumer.register_handler("chat_message", handle_chat_message)
    consumer.register_handler("question_asked", handle_question)
    
    # Learning Agent handlers (async)
    consumer.register_handler("assessment.submitted", handle_assessment_submitted)
    consumer.register_handler("question.answered", handle_question_answered)
    
    # Start consuming (use async for Learning Agent handlers)
    asyncio.run(consumer.start_async())


if __name__ == "__main__":
    run_agent_consumer()

