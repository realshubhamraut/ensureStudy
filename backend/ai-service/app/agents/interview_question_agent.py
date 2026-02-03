"""
Type 5 Interview Question Learning Agent

A self-improving AI agent for interview question generation that:
- Learns from student interview scores to improve question quality
- Automatically generates new questions when 80% are attempted
- Uses psychometric metrics to evaluate question effectiveness
- Persists learning memory for continuous improvement
- Implements duplicate detection to avoid repetitive questions

LangGraph-based workflow with nodes:
1. load_memory - Load learning memory for the topic
2. analyze_performance - Evaluate recent interview responses
3. update_learning - Update agent memory with learned patterns
4. check_threshold - Check if 80% questions attempted
5. generate_questions - Generate new descriptive Q&A using learned strategy
6. deduplicate - Remove duplicate questions
7. format_output - Format final output
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


class InterviewLearningState(TypedDict):
    """State for the Type 5 Interview Question Agent"""
    
    # Input
    task_type: str
    topic_id: str
    topic_name: str
    topic_description: str
    classroom_id: Optional[str]
    
    # Learning Memory (loaded from DB)
    memory: Dict[str, Any]
    
    # Recent Performance Data
    recent_responses: List[Dict]  # Last N interview evaluations for this topic
    
    # Question Pool
    existing_questions: List[Dict]  # Current questions for topic
    questions_attempted: int
    total_questions: int
    attempt_percentage: float
    
    # Generation Config
    questions_per_topic: int
    
    # Processing
    generation_strategy: Dict[str, Any]
    generated_questions: List[Dict]
    deduplicated_questions: List[Dict]
    
    # Output
    questions: List[Dict]
    output: Dict
    error: Optional[str]
    
    # Metrics
    learning_triggered: bool
    generation_triggered: bool


# ============================================================================
# Node Functions
# ============================================================================

def load_interview_memory(state: InterviewLearningState) -> InterviewLearningState:
    """Load learning memory for the topic from database."""
    logger.info(f"[InterviewQ-T5] Loading memory for topic: {state.get('topic_id', 'unknown')}")
    
    # In production, load from LearningAgentMemory table
    # For now, initialize default memory
    if not state.get('memory'):
        state['memory'] = {
            'calibrated_difficulty': 0.5,  # 0=easy, 1=hard
            'target_avg_score': 70,  # Target average interview score
            'preferred_question_types': ['conceptual', 'application', 'analysis'],
            'avoided_patterns': [],
            'successful_prompts': [],
            'learning_iterations': 0,
            'avg_interview_score': 0,
            'total_evaluations': 0
        }
    
    return state


def analyze_interview_performance(state: InterviewLearningState) -> InterviewLearningState:
    """
    Critic function: Analyze recent interview responses to evaluate question quality.
    
    Updates:
    - Question effectiveness scores based on interview scores
    - Topic difficulty calibration
    - Identifies concept gaps
    """
    logger.info("[InterviewQ-T5] Analyzing recent interview performance...")
    
    if not state.get('recent_responses'):
        logger.info("[InterviewQ-T5] No recent responses to analyze")
        return state
    
    responses = state['recent_responses']
    
    # Calculate topic-level metrics from interview scores
    scores = [r.get('score', 0) for r in responses]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # Identify weak concepts (concepts that were missed frequently)
    missed_concepts = []
    for r in responses:
        missed_concepts.extend(r.get('missed_concepts', []))
    
    # Count concept frequency
    concept_counts = {}
    for concept in missed_concepts:
        concept_counts[concept] = concept_counts.get(concept, 0) + 1
    
    # Top 5 weak concepts
    weak_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Safe memory access - initialize if needed
    if not state.get('memory'):
        state['memory'] = {}
    
    # Update memory with actual performance
    state['memory']['avg_interview_score'] = avg_score
    state['memory']['weak_concepts'] = [c[0] for c in weak_concepts]
    state['memory']['total_evaluations'] = len(responses)
    
    logger.info(f"[InterviewQ-T5] Avg score: {avg_score:.1f}, Weak concepts: {len(weak_concepts)}")
    
    return state


def update_interview_learning(state: InterviewLearningState) -> InterviewLearningState:
    """
    Learning Element: Update generation strategy based on performance analysis.
    
    This is the core of the Type 5 agent - it improves over time.
    """
    logger.info("[InterviewQ-T5] Updating learning strategy...")
    
    # Safe memory access
    memory = state.get('memory') or {}
    
    # Adjust difficulty based on average interview score
    avg_score = memory.get('avg_interview_score', 70)
    target_score = memory.get('target_avg_score', 70)
    
    difficulty = memory.get('calibrated_difficulty', 0.5)
    
    # If students scoring too high, increase difficulty
    if avg_score > target_score + 10:
        difficulty = min(1.0, difficulty + 0.1)
        logger.info(f"[InterviewQ-T5] Increasing difficulty: {difficulty:.2f}")
    # If scoring too low, decrease difficulty
    elif avg_score < target_score - 10:
        difficulty = max(0.0, difficulty - 0.1)
        logger.info(f"[InterviewQ-T5] Decreasing difficulty: {difficulty:.2f}")
    
    memory['calibrated_difficulty'] = difficulty
    
    # Build generation strategy based on weak concepts
    weak_concepts = memory.get('weak_concepts', [])
    if weak_concepts:
        # Focus on weak areas
        memory['focus_concepts'] = weak_concepts
        logger.info(f"[InterviewQ-T5] Focusing on weak concepts: {weak_concepts}")
    
    # Increment learning iterations
    memory['learning_iterations'] = memory.get('learning_iterations', 0) + 1
    memory['last_learning_at'] = datetime.utcnow().isoformat()
    
    state['memory'] = memory
    state['learning_triggered'] = True
    
    return state


def check_interview_threshold(state: InterviewLearningState) -> InterviewLearningState:
    """
    Problem Generator: Check if 80% of questions are attempted.
    
    If threshold reached, trigger question generation.
    """
    attempted = state.get('questions_attempted', 0)
    total = state.get('total_questions', 0)
    
    if total == 0:
        state['attempt_percentage'] = 0.0
        state['generation_triggered'] = True  # Generate if no questions exist
        logger.info("[InterviewQ-T5] No questions exist, triggering generation")
        return state
    
    percentage = (attempted / total) * 100
    state['attempt_percentage'] = percentage
    
    # Trigger generation at 80% threshold
    if percentage >= 80:
        state['generation_triggered'] = True
        logger.info(f"[InterviewQ-T5] Threshold reached: {percentage:.1f}% attempted, triggering generation")
    else:
        state['generation_triggered'] = False
        logger.info(f"[InterviewQ-T5] Below threshold: {percentage:.1f}% attempted")
    
    return state


async def generate_interview_questions(state: InterviewLearningState) -> InterviewLearningState:
    """
    Performance Element: Generate new descriptive questions using learned strategy.
    
    Uses the LLM with prompts refined by the learning element.
    """
    logger.info("[InterviewQ-T5] Generating questions using learned strategy...")
    
    if not state.get('generation_triggered'):
        logger.info("[InterviewQ-T5] Generation not triggered, skipping")
        return state
    
    # Safe memory access
    memory = state.get('memory') or {}
    
    # Build prompt based on learned strategy
    difficulty = memory.get('calibrated_difficulty', 0.5)
    difficulty_name = 'easy' if difficulty < 0.33 else 'hard' if difficulty > 0.66 else 'medium'
    
    # Focus on weak concepts if available
    focus_concepts = memory.get('focus_concepts', [])
    focus_instruction = ""
    if focus_concepts:
        focus_instruction = f"\nFocus especially on these concepts students struggle with: {', '.join(focus_concepts)}"
    
    # Avoided patterns
    avoided_patterns = memory.get('avoided_patterns', [])
    avoid_instruction = ""
    if avoided_patterns:
        avoid_instruction = f"\nAvoid these question patterns: {json.dumps(avoided_patterns[:3])}"
    
    # Topic info
    topic_id = state.get('topic_id', '')
    topic_name = state.get('topic_name', 'General Topic')
    topic_description = state.get('topic_description', '')
    questions_per_topic = state.get('questions_per_topic', 5)
    
    difficulty_guidance = {
        "easy": "Basic understanding and recall. Student should explain fundamental concepts in simple terms.",
        "medium": "Application and analysis. Student should explain processes, compare concepts, or solve problems.",
        "hard": "Deep synthesis and evaluation. Student should analyze complex scenarios, critique ideas, or make connections."
    }
    
    prompt = f"""Generate {questions_per_topic} descriptive interview questions for the topic: "{topic_name}"

Topic Description: {topic_description or 'A standard academic topic.'}

Difficulty: {difficulty_name} (calibrated value: {difficulty:.2f})
Target average score: {memory.get('target_avg_score', 70)}%
{difficulty_guidance.get(difficulty_name, difficulty_guidance['medium'])}
{focus_instruction}
{avoid_instruction}

Requirements:
1. Each question should test a distinct concept
2. Questions should be open-ended, requiring explanation (not yes/no)
3. Expected answers should be comprehensive (100-300 words)
4. Key concepts must be specific and testable
5. Match the difficulty level specified

Return a JSON array:
[
  {{
    "question": "Explain the concept of X and its significance in Y.",
    "expected_answer": "A complete, well-structured answer that covers...",
    "key_concepts": ["Concept 1", "Concept 2", "Concept 3"],
    "difficulty": "{difficulty_name}"
  }}
]

Return ONLY the JSON array, no other text."""

    try:
        import os
        from groq import Groq
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.error("[InterviewQ-T5] No GROQ_API_KEY found in graph node!")
            state['generated_questions'] = []
            state['error'] = "No GROQ_API_KEY configured"
            return state
        
        client = Groq(api_key=groq_api_key)
        logger.info(f"[InterviewQ-T5] Graph node calling Groq API for {topic_name}...")
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        text = response.choices[0].message.content.strip()
        logger.info(f"[InterviewQ-T5] Graph node received response, length: {len(text)}")
        
        # Parse JSON - handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            text = text.strip()
        
        questions = json.loads(text)
        
        # Add metadata to each question
        for q in questions:
            q['topic_id'] = topic_id
            q['topic_name'] = topic_name
            q['auto_generated'] = True
            q['generation_strategy'] = {
                'difficulty': difficulty,
                'learning_iteration': memory.get('learning_iterations', 0),
                'focus_concepts': focus_concepts
            }
            # Compute hash for duplicate detection
            normalized = q['question'].lower().strip()
            q['question_hash'] = hashlib.sha256(normalized.encode()).hexdigest()
        
        state['generated_questions'] = questions
        logger.info(f"[InterviewQ-T5] Generated {len(questions)} questions")
        
        # Track successful prompt
        memory.setdefault('successful_prompts', []).append({
            'difficulty': difficulty_name,
            'questions_generated': len(questions),
            'timestamp': datetime.utcnow().isoformat()
        })
        # Keep only last 10 successful prompts
        memory['successful_prompts'] = memory['successful_prompts'][-10:]
        
    except json.JSONDecodeError as e:
        logger.error(f"[InterviewQ-T5] Failed to parse generated questions: {e}")
        state['generated_questions'] = []
        state['error'] = f"JSON parse error: {str(e)}"
    except Exception as e:
        logger.error(f"[InterviewQ-T5] Question generation error: {e}")
        import traceback
        logger.error(f"[InterviewQ-T5] Traceback: {traceback.format_exc()}")
        state['generated_questions'] = []
        state['error'] = str(e)
    
    return state


async def deduplicate_interview_questions(state: InterviewLearningState) -> InterviewLearningState:
    """
    Remove duplicate questions using multi-layer detection:
    1. Hash-based exact match
    2. Text similarity
    """
    logger.info("[InterviewQ-T5] Deduplicating generated questions...")
    
    generated = state.get('generated_questions', [])
    existing = state.get('existing_questions', [])
    
    if not generated:
        state['deduplicated_questions'] = []
        return state
    
    # Build set of existing hashes
    existing_hashes = set()
    for q in existing:
        text = q.get('question', q.get('question_text', '')).lower().strip()
        existing_hashes.add(hashlib.sha256(text.encode()).hexdigest())
    
    unique_questions = []
    new_hashes = set()
    
    for q in generated:
        q_hash = q.get('question_hash', '')
        
        # Check hash-based duplicate
        if q_hash in existing_hashes or q_hash in new_hashes:
            logger.info(f"[InterviewQ-T5] Duplicate detected (hash): {q['question'][:50]}...")
            continue
        
        # Simple text similarity check (could be enhanced with embeddings)
        is_similar = False
        new_text = q['question'].lower()
        for existing_q in existing + unique_questions:
            existing_text = existing_q.get('question', existing_q.get('question_text', '')).lower()
            # Simple word overlap check
            new_words = set(new_text.split())
            existing_words = set(existing_text.split())
            overlap = len(new_words & existing_words) / max(len(new_words), 1)
            if overlap > 0.7:
                is_similar = True
                logger.info(f"[InterviewQ-T5] Duplicate detected (similarity): {q['question'][:50]}...")
                break
        
        if not is_similar:
            unique_questions.append(q)
            new_hashes.add(q_hash)
    
    state['deduplicated_questions'] = unique_questions
    logger.info(f"[InterviewQ-T5] Deduplicated: {len(generated)} -> {len(unique_questions)}")
    
    return state


def format_interview_output(state: InterviewLearningState) -> InterviewLearningState:
    """Format final output for the agent."""
    
    # Use deduplicated questions if generated, otherwise use existing
    questions = state.get('deduplicated_questions', [])
    
    # Safe memory access
    memory = state.get('memory') or {}
    
    state['questions'] = questions
    state['output'] = {
        'success': not bool(state.get('error')),
        'topic_id': state.get('topic_id'),
        'topic_name': state.get('topic_name'),
        'learning_triggered': state.get('learning_triggered', False),
        'generation_triggered': state.get('generation_triggered', False),
        'questions_generated': len(state.get('generated_questions', [])),
        'questions_after_dedupe': len(questions),
        'attempt_percentage': state.get('attempt_percentage', 0),
        'memory_updated': {
            'calibrated_difficulty': memory.get('calibrated_difficulty'),
            'learning_iterations': memory.get('learning_iterations'),
            'avg_interview_score': memory.get('avg_interview_score')
        },
        'error': state.get('error')
    }
    
    return state


# ============================================================================
# Routing Functions
# ============================================================================

def should_generate_interview(state: InterviewLearningState) -> str:
    """Determine if generation should be triggered."""
    if state.get('generation_triggered'):
        return "generate"
    return "output"


# ============================================================================
# Graph Builder
# ============================================================================

def build_interview_learning_graph():
    """Build LangGraph workflow for the Type 5 Interview Question Agent."""
    
    workflow = StateGraph(InterviewLearningState)
    
    # Add nodes
    workflow.add_node("load_memory", load_interview_memory)
    workflow.add_node("analyze", analyze_interview_performance)
    workflow.add_node("learn", update_interview_learning)
    workflow.add_node("check_threshold", check_interview_threshold)
    workflow.add_node("generate", generate_interview_questions)
    workflow.add_node("deduplicate", deduplicate_interview_questions)
    workflow.add_node("output", format_interview_output)
    
    # Define edges
    workflow.set_entry_point("load_memory")
    workflow.add_edge("load_memory", "analyze")
    workflow.add_edge("analyze", "learn")
    workflow.add_edge("learn", "check_threshold")
    
    # Conditional: generate only if threshold reached
    workflow.add_conditional_edges(
        "check_threshold",
        should_generate_interview,
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

class InterviewQuestionAgent:
    """
    Type 5 Learning Agent for Interview Question Generation
    
    Features:
    - Self-improving question generation based on interview scores
    - Adaptive difficulty calibration
    - Multi-layer duplicate detection
    - Persistent learning memory
    - Focus on weak concepts automatically
    
    Usage:
        agent = InterviewQuestionAgent()
        result = await agent.generate({
            "topic_id": "...",
            "topic_name": "Newton's Laws",
            "topic_description": "...",
            "questions_per_topic": 5
        })
    """
    
    def __init__(self):
        self.graph = build_interview_learning_graph()
        logger.info("[InterviewQ-T5] Initialized Type 5 Interview Question Agent")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Interview Question Agent with full learning pipeline.
        
        Args:
            input_data: {
                task_type: "learn" | "generate" | "evaluate",
                topic_id: str,
                topic_name: str,
                topic_description: str (optional),
                classroom_id: str (optional),
                questions_per_topic: int (default 5),
                recent_responses: List[Dict] (for learning),
                existing_questions: List[Dict] (for deduplication),
                questions_attempted: int,
                total_questions: int
            }
        
        Returns:
            Agent output with generated questions and learning updates
        """
        initial_state: InterviewLearningState = {
            'task_type': input_data.get('task_type', 'generate'),
            'topic_id': input_data.get('topic_id', ''),
            'topic_name': input_data.get('topic_name', 'General Topic'),
            'topic_description': input_data.get('topic_description', ''),
            'classroom_id': input_data.get('classroom_id'),
            'memory': input_data.get('memory', {}),
            'recent_responses': input_data.get('recent_responses', []),
            'existing_questions': input_data.get('existing_questions', []),
            'questions_attempted': input_data.get('questions_attempted', 0),
            'total_questions': input_data.get('total_questions', 0),
            'attempt_percentage': 0.0,
            'questions_per_topic': input_data.get('questions_per_topic', 5),
            'generation_strategy': {},
            'generated_questions': [],
            'deduplicated_questions': [],
            'questions': [],
            'output': {},
            'error': None,
            'learning_triggered': False,
            'generation_triggered': False
        }
        
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'agent': 'interview_question_agent_v5',
                'success': not bool(final_state.get('error')),
                'data': final_state['output'],
                'memory': final_state['memory'],
                'questions': final_state.get('questions', []),
                'count': len(final_state.get('questions', []))
            }
            
        except Exception as e:
            logger.error(f"[InterviewQ-T5] Agent error: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'agent': 'interview_question_agent_v5',
                'success': False,
                'data': {
                    'success': False,
                    'error': str(e)
                },
                'questions': [],
                'count': 0
            }
    
    async def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simplified method to generate questions (for compatibility).
        
        Args:
            input_data: {
                topics: [{id, name, description}],
                questions_per_topic: int (default 5),
                difficulty: "easy" | "medium" | "hard" (default "medium")
            }
        """
        topics = input_data.get('topics', [])
        if not topics:
            return {'success': False, 'questions': [], 'count': 0, 'error': 'No topics provided'}
        
        all_questions = []
        
        for topic in topics:
            result = await self.execute({
                'task_type': 'generate',
                'topic_id': topic.get('id', ''),
                'topic_name': topic.get('name', 'Unknown Topic'),
                'topic_description': topic.get('description', ''),
                'questions_per_topic': input_data.get('questions_per_topic', 5),
                'questions_attempted': 0,  # Force generation
                'total_questions': 0
            })
            
            all_questions.extend(result.get('questions', []))
        
        return {
            'success': True,
            'questions': all_questions,
            'count': len(all_questions),
            'topics_covered': [t.get('name') for t in topics],
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def generate_for_single_topic(
        self, 
        topic_id: str, 
        topic_name: str, 
        description: str = "",
        count: int = 5,
        difficulty: str = "medium"
    ) -> List[Dict]:
        """
        Convenience method to generate questions for a single topic.
        
        Uses direct LLM call for reliability (bypasses complex graph).
        Returns list of questions or empty list on error.
        """
        import json
        
        logger.info(f"[InterviewQ-T5] Direct generation for topic: {topic_name}")
        
        difficulty_guidance = {
            "easy": "Basic understanding and recall. Student should explain fundamental concepts in simple terms.",
            "medium": "Application and analysis. Student should explain processes, compare concepts, or solve problems.",
            "hard": "Deep synthesis and evaluation. Student should analyze complex scenarios, critique ideas, or make connections."
        }
        
        prompt = f"""Generate {count} descriptive interview questions for the topic: "{topic_name}"

Topic Description: {description or 'A standard academic topic.'}

Difficulty: {difficulty}
{difficulty_guidance.get(difficulty, difficulty_guidance['medium'])}

Requirements:
1. Each question should test a distinct concept
2. Questions should be open-ended, requiring explanation (not yes/no)
3. Expected answers should be comprehensive (100-300 words)
4. Key concepts must be specific and testable

Return a JSON array:
[
  {{
    "question": "Explain the concept of X and its significance in Y.",
    "expected_answer": "A complete, well-structured answer that covers...",
    "key_concepts": ["Concept 1", "Concept 2", "Concept 3"],
    "difficulty": "{difficulty}"
  }}
]

Return ONLY the JSON array, no other text."""

        try:
            import os
            from groq import Groq
            
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                logger.error("[InterviewQ-T5] No GROQ_API_KEY found!")
                return []
            
            client = Groq(api_key=groq_api_key)
            logger.info(f"[InterviewQ-T5] Calling Groq API for {topic_name}...")
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            
            text = response.choices[0].message.content.strip()
            logger.info(f"[InterviewQ-T5] Groq response received, length: {len(text)}")
            
            # Parse JSON - handle markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else text
                text = text.strip()
            
            questions = json.loads(text)
            
            # Add metadata to each question
            for q in questions:
                q['topic_id'] = topic_id
                q['topic_name'] = topic_name
                q['auto_generated'] = True
            
            logger.info(f"[InterviewQ-T5] Generated {len(questions)} questions for {topic_name}")
            return questions
            
        except json.JSONDecodeError as e:
            logger.error(f"[InterviewQ-T5] JSON parse error: {e}")
            logger.error(f"[InterviewQ-T5] Raw response: {text[:500] if text else 'None'}")
            return []
        except Exception as e:
            logger.error(f"[InterviewQ-T5] Generation error: {e}")
            import traceback
            logger.error(f"[InterviewQ-T5] Traceback: {traceback.format_exc()}")
            return []
    
    async def trigger_on_interview_complete(
        self,
        topic_id: str,
        topic_name: str,
        evaluations: List[Dict],
        existing_questions: List[Dict] = None,
        questions_attempted: int = 0,
        total_questions: int = 0
    ) -> Dict[str, Any]:
        """
        Trigger learning after an interview session is completed.
        
        This should be called after a mock interview to:
        1. Learn from the evaluation scores
        2. Identify weak concepts
        3. Auto-generate new questions if 80% threshold reached
        
        Args:
            topic_id: Topic that was interviewed
            topic_name: Name of the topic
            evaluations: List of evaluation results [{score, missed_concepts, ...}]
            existing_questions: Current questions for deduplication
            questions_attempted: Total unique questions attempted by user
            total_questions: Total questions available for topic
        """
        return await self.execute({
            'task_type': 'learn',
            'topic_id': topic_id,
            'topic_name': topic_name,
            'recent_responses': evaluations,
            'existing_questions': existing_questions or [],
            'questions_attempted': questions_attempted,
            'total_questions': total_questions
        })


# Singleton instance
_agent_instance = None


def get_interview_question_agent() -> InterviewQuestionAgent:
    """Get or create singleton agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = InterviewQuestionAgent()
    return _agent_instance
