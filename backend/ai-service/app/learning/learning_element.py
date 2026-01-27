"""
Learning Element for Type 5 Tutor Agent

Fetches high-quality examples from the feedback system and injects them
into prompts as few-shot examples, enabling continuous improvement.
"""
import logging
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass
import httpx
import os

logger = logging.getLogger(__name__)


@dataclass
class LearningExample:
    """A learning example for few-shot prompting"""
    topic: str
    query: str
    good_response: str
    weight: float = 1.0


class TutorLearningElement:
    """
    Learning Element for the Tutor Agent.
    
    Fetches high-rated examples from the feedback system and uses them
    to enhance prompt quality through few-shot learning.
    """
    
    def __init__(self):
        self.core_api_url = os.getenv("CORE_SERVICE_URL", "http://localhost:8000")
        self._example_cache: Dict[str, List[LearningExample]] = {}
        self._cache_ttl_seconds = 300  # 5 minute cache
        self._last_fetch = 0
        
    async def get_examples(self, topic: str, limit: int = 3) -> List[LearningExample]:
        """
        Fetch learning examples for a topic.
        
        Returns cached examples if available, otherwise fetches from API.
        """
        import time
        
        # Check cache
        cache_key = topic.lower().strip()
        if cache_key in self._example_cache:
            if time.time() - self._last_fetch < self._cache_ttl_seconds:
                return self._example_cache[cache_key][:limit]
        
        # Fetch from API
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.core_api_url}/api/feedback/examples",
                    params={
                        "agent_type": "tutor",
                        "topic": topic,
                        "limit": limit
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    examples = [
                        LearningExample(
                            topic=ex.get("topic", ""),
                            query=ex.get("query", ""),
                            good_response=ex.get("good_response", ""),
                            weight=ex.get("weight", 1.0)
                        )
                        for ex in data.get("examples", [])
                    ]
                    
                    # Update cache
                    self._example_cache[cache_key] = examples
                    self._last_fetch = time.time()
                    
                    logger.info(f"[LEARNING] Fetched {len(examples)} examples for topic '{topic}'")
                    return examples
                    
        except Exception as e:
            logger.warning(f"[LEARNING] Failed to fetch examples: {e}")
        
        return []
    
    def build_few_shot_prompt(self, examples: List[LearningExample]) -> str:
        """
        Build a few-shot prompt section from learning examples.
        
        Returns a string to inject into the system prompt.
        """
        if not examples:
            return ""
        
        few_shot_section = "\n\nHere are examples of good responses:\n"
        
        for i, ex in enumerate(examples, 1):
            few_shot_section += f"""
---
Example {i}:
Student Question: {ex.query}
Good Response: {ex.good_response[:500]}{'...' if len(ex.good_response) > 500 else ''}
---
"""
        
        few_shot_section += "\nPlease follow a similar style and quality in your response.\n"
        
        return few_shot_section
    
    async def enhance_prompt(self, base_prompt: str, topic: str) -> str:
        """
        Enhance a system prompt with few-shot examples for the given topic.
        """
        examples = await self.get_examples(topic, limit=2)
        
        if not examples:
            return base_prompt
        
        few_shot = self.build_few_shot_prompt(examples)
        
        # Insert few-shot examples before the instructions section
        if "Instructions:" in base_prompt:
            parts = base_prompt.split("Instructions:")
            return parts[0] + few_shot + "Instructions:" + parts[1]
        else:
            return base_prompt + few_shot


class ExperienceReplay:
    """
    Simple experience replay buffer for learning.
    
    Stores interactions with reward signals for batch learning.
    """
    
    def __init__(self, max_size: int = 10000):
        self.buffer: List[Dict] = []
        self.max_size = max_size
        self.core_api_url = os.getenv("CORE_SERVICE_URL", "http://localhost:8000")
    
    async def add_experience(
        self,
        agent_type: str,
        session_id: str,
        query: str,
        response: str,
        metadata: Dict,
        reward: Optional[float] = None
    ):
        """Add an experience to the replay buffer."""
        experience = {
            "agent_type": agent_type,
            "session_id": session_id,
            "query": query,
            "response": response,
            "metadata": metadata,
            "reward": reward
        }
        
        self.buffer.append(experience)
        
        # Trim buffer if too large
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
        
        # Also log to API for persistence
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{self.core_api_url}/api/feedback/interactions",
                    json={
                        "agent_type": agent_type,
                        "session_id": session_id,
                        "query": query,
                        "response": response,
                        "metadata": metadata,
                        "topic": metadata.get("topic")
                    },
                    headers={"Content-Type": "application/json"}
                )
        except Exception as e:
            logger.warning(f"[EXPERIENCE] Failed to log interaction: {e}")
    
    def get_positive_examples(self, min_reward: float = 0.5) -> List[Dict]:
        """Get experiences with positive reward for learning."""
        return [e for e in self.buffer if e.get("reward", 0) >= min_reward]


# Singleton instances
_learning_element: Optional[TutorLearningElement] = None
_experience_replay: Optional[ExperienceReplay] = None


def get_learning_element() -> TutorLearningElement:
    """Get or create the learning element singleton."""
    global _learning_element
    if _learning_element is None:
        _learning_element = TutorLearningElement()
    return _learning_element


def get_experience_replay() -> ExperienceReplay:
    """Get or create the experience replay singleton."""
    global _experience_replay
    if _experience_replay is None:
        _experience_replay = ExperienceReplay()
    return _experience_replay
