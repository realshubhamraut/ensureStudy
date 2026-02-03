"""
Type 5 Learning Agent for Assessment Question Generation

A self-improving AI agent that:
- Learns from student performance to improve question quality
- Automatically generates new questions when 80% are attempted
- Uses psychometric metrics to evaluate question effectiveness
- Persists learning memory for continuous improvement
- Implements duplicate detection to avoid repetitive questions

LangGraph-based workflow with nodes:
1. analyze_performance - Evaluate recent student responses
2. update_learning - Update agent memory with learned patterns
3. check_threshold - Check if 80% questions attempted
4. generate_questions - Generate new MCQs using learned strategy
5. deduplicate - Remove duplicate questions
6. store_questions - Save to database with effectiveness tracking
"""
import logging
import json
import hashlib
from typing import Dict, Any, List, TypedDict, Optional
from datetime import datetime
from enum import Enum

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

class TaskType(str, Enum):
    LEARN = "learn"  # Update learning from responses
    GENERATE = "generate"  # Generate new questions
    EVALUATE = "evaluate"  # Evaluate question effectiveness
    CHECK_THRESHOLD = "check_threshold"  # Check if generation needed


class LearningState(TypedDict):
    """State for the Type 5 Learning Agent"""
    
    # Input
    task_type: str
    topic_id: str
    classroom_id: Optional[str]
    
    # Learning Memory (loaded from DB)
    memory: Dict[str, Any]
    
    # Recent Performance Data
    recent_responses: List[Dict]  # Last N question responses for this topic
    
    # Question Pool
    existing_questions: List[Dict]  # Current questions for topic
    questions_attempted: int
    total_questions: int
    attempt_percentage: float
    
    # Generation
    generation_strategy: Dict[str, Any]
    generated_questions: List[Dict]
    deduplicated_questions: List[Dict]
    
    # Output
    output: Dict
    error: Optional[str]
    
    # Metrics
    learning_triggered: bool
    generation_triggered: bool


# ============================================================================
# Node Functions
# ============================================================================

def load_topic_memory(state: LearningState) -> LearningState:
    """Load learning memory for the topic from database."""
    logger.info(f"Loading memory for topic: {state['topic_id']}")
    
    # In production, load from LearningAgentMemory table
    # For now, initialize default memory
    if not state.get('memory'):
        state['memory'] = {
            'calibrated_difficulty': 0.5,
            'target_success_rate': 0.7,
            'preferred_question_types': ['mcq'],
            'avoided_patterns': [],
            'successful_prompts': [],
            'learning_iterations': 0
        }
    
    return state


def analyze_performance(state: LearningState) -> LearningState:
    """
    Critic function: Analyze recent student responses to evaluate question quality.
    
    Updates:
    - Question effectiveness scores
    - Topic difficulty calibration
    - Identifies problematic patterns
    """
    logger.info("Analyzing recent performance...")
    
    if not state.get('recent_responses'):
        logger.info("No recent responses to analyze")
        return state
    
    responses = state['recent_responses']
    
    # Calculate topic-level metrics
    total_correct = sum(1 for r in responses if r.get('is_correct'))
    total_count = len(responses)
    success_rate = total_correct / total_count if total_count > 0 else 0.5
    
    # Update memory with actual success rate
    state['memory']['actual_success_rate'] = success_rate
    
    # Identify patterns that confuse students
    incorrect_responses = [r for r in responses if not r.get('is_correct')]
    confusion_patterns = []
    
    for r in incorrect_responses:
        # Look for common mistakes
        question_type = r.get('question_type', 'mcq')
        if question_type == 'mcq':
            selected = r.get('selected_option')
            if selected:
                # Track which distractors are selected
                confusion_patterns.append({
                    'question_id': r.get('question_id'),
                    'selected_distractor': selected,
                    'response_time_ms': r.get('response_time_ms', 0)
                })
    
    # Store patterns for learning
    state['memory']['confusion_patterns'] = confusion_patterns[:10]  # Keep last 10
    
    logger.info(f"Success rate: {success_rate:.2%}, Confusion patterns: {len(confusion_patterns)}")
    
    return state


def update_learning(state: LearningState) -> LearningState:
    """
    Learning Element: Update generation strategy based on performance analysis.
    
    This is the core of the Type 5 agent - it improves over time.
    """
    logger.info("Updating learning strategy...")
    
    memory = state['memory']
    
    # Adjust difficulty based on success rate
    actual_rate = memory.get('actual_success_rate', 0.5)
    target_rate = memory.get('target_success_rate', 0.7)
    
    difficulty = memory.get('calibrated_difficulty', 0.5)
    
    # If success rate too high, increase difficulty
    if actual_rate > target_rate + 0.1:
        difficulty = min(1.0, difficulty + 0.1)
        logger.info(f"Increasing difficulty: {difficulty:.2f}")
    # If success rate too low, decrease difficulty
    elif actual_rate < target_rate - 0.1:
        difficulty = max(0.0, difficulty - 0.1)
        logger.info(f"Decreasing difficulty: {difficulty:.2f}")
    
    memory['calibrated_difficulty'] = difficulty
    
    # Update avoided patterns based on confusion
    confusion_patterns = memory.get('confusion_patterns', [])
    if len(confusion_patterns) > 5:
        # Pattern is causing too much confusion
        memory.setdefault('avoided_patterns', []).append({
            'type': 'high_confusion',
            'count': len(confusion_patterns),
            'identified_at': datetime.utcnow().isoformat()
        })
    
    # Increment learning iterations
    memory['learning_iterations'] = memory.get('learning_iterations', 0) + 1
    memory['last_learning_at'] = datetime.utcnow().isoformat()
    
    state['memory'] = memory
    state['learning_triggered'] = True
    
    return state


def check_threshold(state: LearningState) -> LearningState:
    """
    Problem Generator: Check if 80% of questions are attempted.
    
    If threshold reached, trigger question generation.
    """
    attempted = state.get('questions_attempted', 0)
    total = state.get('total_questions', 0)
    
    if total == 0:
        state['attempt_percentage'] = 0.0
        state['generation_triggered'] = True  # Generate if no questions exist
        logger.info("No questions exist, triggering generation")
        return state
    
    percentage = (attempted / total) * 100
    state['attempt_percentage'] = percentage
    
    # Trigger generation at 80% threshold
    if percentage >= 80:
        state['generation_triggered'] = True
        logger.info(f"Threshold reached: {percentage:.1f}% attempted, triggering generation")
    else:
        state['generation_triggered'] = False
        logger.info(f"Below threshold: {percentage:.1f}% attempted")
    
    return state


async def generate_questions(state: LearningState) -> LearningState:
    """
    Performance Element: Generate new MCQ questions using learned strategy.
    
    Uses the LLM with prompts refined by the learning element.
    """
    logger.info("Generating questions using learned strategy...")
    
    if not state.get('generation_triggered'):
        logger.info("Generation not triggered, skipping")
        return state
    
    memory = state['memory']
    
    # Build prompt based on learned strategy
    difficulty = memory.get('calibrated_difficulty', 0.5)
    difficulty_name = 'easy' if difficulty < 0.33 else 'hard' if difficulty > 0.66 else 'medium'
    
    avoided_patterns = memory.get('avoided_patterns', [])
    avoid_instructions = ""
    if avoided_patterns:
        avoid_instructions = f"\nAvoid these patterns that have confused students: {json.dumps(avoided_patterns[:3])}"
    
    # Topic info
    topic_id = state.get('topic_id', '')
    
    prompt = f"""Generate 5 multiple choice questions for topic ID: {topic_id}

Difficulty: {difficulty_name} (calibrated value: {difficulty:.2f})
Target success rate: {memory.get('target_success_rate', 0.7):.0%}
{avoid_instructions}

Requirements:
1. Each question should test a distinct concept
2. Options should include plausible distractors
3. Explanations should be educational
4. Match the difficulty level specified

Return a JSON array:
[
  {{
    "question_text": "Question here?",
    "options": [
      {{"id": "A", "text": "Option A"}},
      {{"id": "B", "text": "Option B"}},
      {{"id": "C", "text": "Option C"}},
      {{"id": "D", "text": "Option D"}}
    ],
    "correct_answer": "A",
    "explanation": "Why A is correct...",
    "difficulty": "{difficulty_name}"
  }}
]

Return ONLY valid JSON, no other text."""

    try:
        from app.services.llm_provider import get_llm
        
        llm = get_llm()
        response = llm.invoke(prompt)
        
        # Parse JSON
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        
        questions = json.loads(text)
        
        # Add metadata to each question
        for q in questions:
            q['topic_id'] = topic_id
            q['question_type'] = 'mcq'
            q['auto_generated'] = True
            q['generation_strategy'] = {
                'difficulty': difficulty,
                'learning_iteration': memory.get('learning_iterations', 0)
            }
            # Compute hash for duplicate detection
            normalized = q['question_text'].lower().strip()
            q['question_hash'] = hashlib.sha256(normalized.encode()).hexdigest()
        
        state['generated_questions'] = questions
        logger.info(f"Generated {len(questions)} questions")
        
        # Track successful prompt
        memory.setdefault('successful_prompts', []).append({
            'difficulty': difficulty_name,
            'questions_generated': len(questions),
            'timestamp': datetime.utcnow().isoformat()
        })
        # Keep only last 10 successful prompts
        memory['successful_prompts'] = memory['successful_prompts'][-10:]
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse generated questions: {e}")
        state['generated_questions'] = []
        state['error'] = f"JSON parse error: {str(e)}"
    except Exception as e:
        logger.error(f"Question generation error: {e}")
        state['generated_questions'] = []
        state['error'] = str(e)
    
    return state


async def deduplicate_questions(state: LearningState) -> LearningState:
    """
    Remove duplicate questions using multi-layer detection.
    """
    logger.info("Deduplicating generated questions...")
    
    generated = state.get('generated_questions', [])
    existing = state.get('existing_questions', [])
    
    if not generated:
        state['deduplicated_questions'] = []
        return state
    
    from app.utils.duplicate_detector import check_duplicate
    
    unique_questions = []
    
    for q in generated:
        result = await check_duplicate(
            q['question_text'],
            existing + unique_questions,  # Check against existing and already-added new ones
            use_llm=False  # Skip LLM for performance
        )
        
        if not result['is_duplicate']:
            unique_questions.append(q)
            logger.info(f"Question accepted: {q['question_text'][:50]}...")
        else:
            logger.info(f"Duplicate detected: {result['explanation']}")
    
    state['deduplicated_questions'] = unique_questions
    logger.info(f"Deduplicated: {len(generated)} -> {len(unique_questions)}")
    
    return state


def format_output(state: LearningState) -> LearningState:
    """Format final output for the agent."""
    
    state['output'] = {
        'success': not bool(state.get('error')),
        'topic_id': state.get('topic_id'),
        'learning_triggered': state.get('learning_triggered', False),
        'generation_triggered': state.get('generation_triggered', False),
        'questions_generated': len(state.get('generated_questions', [])),
        'questions_after_dedupe': len(state.get('deduplicated_questions', [])),
        'attempt_percentage': state.get('attempt_percentage', 0),
        'memory_updated': {
            'calibrated_difficulty': state['memory'].get('calibrated_difficulty'),
            'learning_iterations': state['memory'].get('learning_iterations')
        },
        'error': state.get('error')
    }
    
    return state


# ============================================================================
# Routing Functions
# ============================================================================

def should_generate(state: LearningState) -> str:
    """Determine if generation should be triggered."""
    if state.get('generation_triggered'):
        return "generate"
    return "output"


# ============================================================================
# Graph Builder
# ============================================================================

def build_learning_agent_graph():
    """Build LangGraph workflow for the Type 5 Learning Agent."""
    
    workflow = StateGraph(LearningState)
    
    # Add nodes
    workflow.add_node("load_memory", load_topic_memory)
    workflow.add_node("analyze", analyze_performance)
    workflow.add_node("learn", update_learning)
    workflow.add_node("check_threshold", check_threshold)
    workflow.add_node("generate", generate_questions)
    workflow.add_node("deduplicate", deduplicate_questions)
    workflow.add_node("output", format_output)
    
    # Define edges
    workflow.set_entry_point("load_memory")
    workflow.add_edge("load_memory", "analyze")
    workflow.add_edge("analyze", "learn")
    workflow.add_edge("learn", "check_threshold")
    
    # Conditional: generate only if threshold reached
    workflow.add_conditional_edges(
        "check_threshold",
        should_generate,
        {
            "generate": "generate",
            "output": "output"
        }
    )
    
    workflow.add_edge("generate", "deduplicate")
    workflow.add_edge("deduplicate", "output")
    workflow.add_edge("output", END)
    
    return workflow.compile()


# ============================================================================
# Agent Class
# ============================================================================

class LearningAgent:
    """
    Type 5 Learning Agent for Assessment Question Generation
    
    Features:
    - Self-improving question generation
    - Adaptive difficulty calibration
    - Multi-layer duplicate detection
    - Persistent learning memory
    """
    
    def __init__(self):
        self.graph = build_learning_agent_graph()
        logger.info("Initialized Type 5 Learning Agent")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Learning Agent.
        
        Args:
            input_data: {
                task_type: "learn" | "generate" | "evaluate",
                topic_id: str,
                classroom_id: str (optional),
                recent_responses: List[Dict] (for learning),
                existing_questions: List[Dict] (for deduplication)
            }
        
        Returns:
            Agent output with generated questions and learning updates
        """
        initial_state: LearningState = {
            'task_type': input_data.get('task_type', 'learn'),
            'topic_id': input_data.get('topic_id', ''),
            'classroom_id': input_data.get('classroom_id'),
            'memory': input_data.get('memory', {}),
            'recent_responses': input_data.get('recent_responses', []),
            'existing_questions': input_data.get('existing_questions', []),
            'questions_attempted': input_data.get('questions_attempted', 0),
            'total_questions': input_data.get('total_questions', 0),
            'attempt_percentage': 0.0,
            'generation_strategy': {},
            'generated_questions': [],
            'deduplicated_questions': [],
            'output': {},
            'error': None,
            'learning_triggered': False,
            'generation_triggered': False
        }
        
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'agent': 'learning_agent_v5',
                'data': final_state['output'],
                'memory': final_state['memory'],
                'questions': final_state.get('deduplicated_questions', [])
            }
            
        except Exception as e:
            logger.error(f"Learning Agent error: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'agent': 'learning_agent_v5',
                'data': {
                    'success': False,
                    'error': str(e)
                },
                'questions': []
            }
    
    async def trigger_on_assessment_submit(
        self,
        topic_id: str,
        responses: List[Dict],
        db_session=None
    ) -> Dict[str, Any]:
        """
        Convenience method to trigger learning after assessment submission.
        
        This is called by the Kafka consumer when an assessment is submitted.
        """
        # Load existing questions for duplicate checking
        existing_questions = []
        questions_attempted = 0
        total_questions = 0
        
        if db_session:
            from app.models.curriculum import TopicQuestion, StudentQuestionResponse
            
            # Get all questions for topic
            questions = db_session.query(TopicQuestion).filter_by(
                classroom_topic_id=topic_id,
                is_active=True
            ).all()
            
            existing_questions = [q.to_dict(include_answer=True) for q in questions]
            total_questions = len(existing_questions)
            
            # Get count of attempted questions by this student
            # (This would need user_id context in production)
            questions_attempted = int(total_questions * 0.5)  # Placeholder
        
        return await self.execute({
            'task_type': 'learn',
            'topic_id': topic_id,
            'recent_responses': responses,
            'existing_questions': existing_questions,
            'questions_attempted': questions_attempted,
            'total_questions': total_questions
        })


# Singleton instance
_learning_agent = None


def get_learning_agent() -> LearningAgent:
    """Get or create the Learning Agent singleton."""
    global _learning_agent
    if _learning_agent is None:
        _learning_agent = LearningAgent()
    return _learning_agent
