"""
Question Pool Agent

Monitors question usage and automatically generates more questions when the pool depletes.
Similar to Type 5 Assessment Agent pattern.
"""
import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Configuration
MIN_QUESTIONS_PER_TOPIC = 5  # Minimum questions to maintain
GENERATION_THRESHOLD = 0.8   # Trigger when 80% questions answered
GENERATE_BATCH_SIZE = 3      # Generate 3 new questions at a time


class QuestionPoolAgent:
    """
    Background agent that monitors and replenishes question pools.
    
    Triggered when:
    - Student finishes 80%+ of available questions for a topic
    - Topic has fewer than MIN_QUESTIONS_PER_TOPIC questions
    
    Actions:
    - Generates new questions using InterviewQuestionAgent
    - Stores them in the database via core-service API
    """
    
    def __init__(self):
        self._generation_in_progress: Dict[str, bool] = {}
        logger.info("[QuestionPoolAgent] Initialized")
    
    async def check_and_replenish(
        self,
        topic_id: str,
        topic_name: str,
        topic_description: str = "",
        current_count: int = 0,
        token: str = ""
    ) -> Dict[str, Any]:
        """
        Check if a topic needs more questions and generate if needed.
        
        Args:
            topic_id: The ClassroomTopic ID
            topic_name: Name of the topic
            topic_description: Description for better question generation
            current_count: Current number of active questions
            token: Auth token for API calls
            
        Returns:
            Result dict with generation status
        """
        # Prevent duplicate generation
        if self._generation_in_progress.get(topic_id):
            logger.info(f"[QuestionPoolAgent] Generation already in progress for {topic_id}")
            return {"status": "in_progress", "topic_id": topic_id}
        
        if current_count >= MIN_QUESTIONS_PER_TOPIC:
            logger.debug(f"[QuestionPoolAgent] Topic {topic_name} has enough questions ({current_count})")
            return {"status": "sufficient", "topic_id": topic_id, "count": current_count}
        
        try:
            self._generation_in_progress[topic_id] = True
            
            logger.info(f"[QuestionPoolAgent] Generating questions for '{topic_name}' (current: {current_count})")
            
            # Generate new questions
            from app.agents.interview_question_agent import get_interview_question_agent
            
            agent = get_interview_question_agent()
            result = await agent.generate({
                "topics": [{
                    "id": topic_id,
                    "name": topic_name,
                    "description": topic_description
                }],
                "questions_per_topic": GENERATE_BATCH_SIZE,
                "difficulty": "medium"  # Could be made dynamic based on student performance
            })
            
            if not result.get("success"):
                logger.warning(f"[QuestionPoolAgent] Failed to generate questions: {result.get('error')}")
                return {
                    "status": "generation_failed",
                    "topic_id": topic_id,
                    "error": result.get("error")
                }
            
            # Store in database via core-service
            stored_count = await self._store_questions(
                result.get("questions", []),
                token
            )
            
            logger.info(f"[QuestionPoolAgent] Stored {stored_count} new questions for '{topic_name}'")
            
            return {
                "status": "generated",
                "topic_id": topic_id,
                "generated": result.get("count", 0),
                "stored": stored_count
            }
            
        except Exception as e:
            logger.error(f"[QuestionPoolAgent] Error: {e}")
            return {
                "status": "error",
                "topic_id": topic_id,
                "error": str(e)
            }
        finally:
            self._generation_in_progress[topic_id] = False
    
    async def _store_questions(self, questions: List[Dict], token: str) -> int:
        """Store generated questions in the database."""
        import httpx
        import os
        
        CORE_SERVICE_URL = os.getenv("CORE_SERVICE_URL", "http://localhost:5000")
        stored = 0
        
        for q in questions:
            try:
                async with httpx.AsyncClient() as client:
                    # The core-service CRUD endpoint for creating questions
                    response = await client.post(
                        f"{CORE_SERVICE_URL}/api/interview-questions/generate",
                        json={
                            "topics": [{
                                "id": q.get("topic_id"),
                                "name": q.get("topic_name"),
                                "description": ""
                            }],
                            "questions_per_topic": 1,
                            "difficulty": q.get("difficulty", "medium")
                        },
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        stored += 1
            except Exception as e:
                logger.warning(f"[QuestionPoolAgent] Failed to store question: {e}")
        
        return stored
    
    async def check_session_completion(
        self,
        session_data: Dict[str, Any],
        token: str
    ) -> None:
        """
        Called when a student completes a session.
        Checks if question pools need replenishment.
        """
        topic_ids = session_data.get("topic_ids", [])
        topics = session_data.get("topics", [])
        
        for i, topic_id in enumerate(topic_ids):
            topic_name = topics[i].get("name", "") if i < len(topics) else ""
            
            # Check pool status
            try:
                import httpx
                import os
                
                CORE_SERVICE_URL = os.getenv("CORE_SERVICE_URL", "http://localhost:5000")
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{CORE_SERVICE_URL}/api/interview-questions/topic/{topic_id}/count",
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=10.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("needs_generation", False):
                            # Trigger background generation
                            asyncio.create_task(
                                self.check_and_replenish(
                                    topic_id=topic_id,
                                    topic_name=topic_name,
                                    current_count=data.get("active_count", 0),
                                    token=token
                                )
                            )
            except Exception as e:
                logger.warning(f"[QuestionPoolAgent] Error checking topic {topic_id}: {e}")
    
    async def bulk_replenish(
        self,
        topic_data: List[Dict[str, Any]],
        token: str
    ) -> Dict[str, Any]:
        """
        Replenish questions for multiple topics at once.
        
        Args:
            topic_data: List of {topic_id, topic_name, topic_description}
            token: Auth token
            
        Returns:
            Summary of generation results
        """
        results = []
        
        for topic in topic_data:
            result = await self.check_and_replenish(
                topic_id=topic.get("id", ""),
                topic_name=topic.get("name", ""),
                topic_description=topic.get("description", ""),
                current_count=0,  # Force generation
                token=token
            )
            results.append(result)
        
        return {
            "total_topics": len(topic_data),
            "generated": sum(1 for r in results if r.get("status") == "generated"),
            "failed": sum(1 for r in results if r.get("status") in ["error", "generation_failed"]),
            "results": results
        }


# Singleton instance
_agent_instance = None

def get_question_pool_agent() -> QuestionPoolAgent:
    """Get or create singleton agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = QuestionPoolAgent()
    return _agent_instance
