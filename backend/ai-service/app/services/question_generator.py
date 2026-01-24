"""
Question Generator Service
==========================

Generates MCQ and descriptive questions from classroom materials using LLM.
Integrates with Qdrant for context retrieval and supports topic-based generation.

Technologies:
- HuggingFace LLM for question generation
- Qdrant for content retrieval
- Sentence Transformers for embeddings
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class GeneratedQuestion:
    """Structured question object"""
    question_type: str  # mcq, descriptive, short_answer
    question_text: str
    options: List[Dict[str, str]]  # For MCQ: [{"id": "A", "text": "..."}]
    correct_answer: str
    explanation: str
    key_points: List[str]
    difficulty: str
    source_content: str
    marks: int = 1
    time_estimate_seconds: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QuestionGeneratorService:
    """
    Service for generating questions from classroom materials.
    
    Features:
    - MCQ generation with 4 options
    - Descriptive question generation with key points
    - Short answer question generation
    - Topic-aware generation from Qdrant chunks
    - Difficulty-based generation (easy, medium, hard)
    """
    
    # Question type configs
    QUESTION_CONFIGS = {
        "mcq": {
            "marks": 1,
            "time_seconds": 45,
            "num_options": 4
        },
        "short_answer": {
            "marks": 2,
            "time_seconds": 90,
        },
        "descriptive": {
            "marks": 5,
            "time_seconds": 300,
        }
    }
    
    def __init__(self):
        """Initialize the question generator"""
        self.api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        self.model_name = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        
        # Lazy load clients
        self._llm_client = None
        self._embedder = None
        self._qdrant = None
        
        logger.info(f"[QuestionGenerator] Initialized with model: {self.model_name}")
    
    @property
    def llm_client(self):
        """Lazy load HuggingFace inference client"""
        if self._llm_client is None:
            try:
                from huggingface_hub import InferenceClient
                self._llm_client = InferenceClient(token=self.api_key)
                logger.info("[QuestionGenerator] LLM client initialized")
            except Exception as e:
                logger.error(f"[QuestionGenerator] Failed to init LLM client: {e}")
                raise
        return self._llm_client
    
    @property
    def embedder(self):
        """Lazy load sentence transformer"""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
                self._embedder = SentenceTransformer(model_name)
                logger.info(f"[QuestionGenerator] Embedder loaded: {model_name}")
            except Exception as e:
                logger.error(f"[QuestionGenerator] Failed to load embedder: {e}")
                raise
        return self._embedder
    
    @property
    def qdrant(self):
        """Lazy load Qdrant client"""
        if self._qdrant is None:
            try:
                from qdrant_client import QdrantClient
                host = os.getenv("QDRANT_HOST", "localhost")
                port = int(os.getenv("QDRANT_PORT", "6333"))
                self._qdrant = QdrantClient(host=host, port=port)
                logger.info(f"[QuestionGenerator] Qdrant connected: {host}:{port}")
            except Exception as e:
                logger.error(f"[QuestionGenerator] Failed to connect Qdrant: {e}")
                raise
        return self._qdrant
    
    def _create_mcq_prompt(self, content: str, num_questions: int, difficulty: str) -> str:
        """Create prompt for MCQ generation"""
        difficulty_guidance = {
            "easy": "Focus on basic definitions, simple facts, and straightforward concepts.",
            "medium": "Include application-based questions and concepts that require understanding.",
            "hard": "Ask analytical questions, compare/contrast concepts, and test deep understanding."
        }
        
        return f"""You are an expert educator creating multiple choice questions for assessments.

CONTENT TO GENERATE QUESTIONS FROM:
{content[:4000]}

INSTRUCTIONS:
1. Generate exactly {num_questions} multiple choice questions
2. Difficulty level: {difficulty.upper()} - {difficulty_guidance.get(difficulty, difficulty_guidance['medium'])}
3. Each question MUST have exactly 4 options (A, B, C, D)
4. Questions must be directly answerable from the content
5. DO NOT use phrases like "According to the passage" or "Based on the text"
6. Start questions directly (What, Which, How, Why, etc.)

OUTPUT FORMAT (use this EXACT format for each question):
QUESTION_1:
Text: [Your question here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct: [A/B/C/D]
Explanation: [Why this answer is correct]

Generate {num_questions} questions now:"""
    
    def _create_descriptive_prompt(self, content: str, num_questions: int, difficulty: str) -> str:
        """Create prompt for descriptive question generation"""
        difficulty_guidance = {
            "easy": "Ask about definitions, basic explanations, and simple descriptions.",
            "medium": "Ask about processes, causes, effects, and require moderate explanation.",
            "hard": "Ask about analysis, evaluation, comparison, and require detailed explanation."
        }
        
        return f"""You are an expert educator creating descriptive questions for assessments.

CONTENT TO GENERATE QUESTIONS FROM:
{content[:4000]}

INSTRUCTIONS:
1. Generate exactly {num_questions} descriptive/essay-type questions
2. Difficulty level: {difficulty.upper()} - {difficulty_guidance.get(difficulty, difficulty_guidance['medium'])}
3. Use question starters: Explain, Describe, Compare, Analyze, Discuss, How, Why
4. DO NOT use phrases like "According to the passage" or "Based on the text"
5. Each question should require 3-5 sentences to answer properly

OUTPUT FORMAT (use this EXACT format for each question):
QUESTION_1:
Text: [Your descriptive question here]
Key Points:
- [Key point 1 that should be in the answer]
- [Key point 2 that should be in the answer]
- [Key point 3 that should be in the answer]
Model Answer: [A brief model answer covering all key points]

Generate {num_questions} questions now:"""
    
    def _create_short_answer_prompt(self, content: str, num_questions: int, difficulty: str) -> str:
        """Create prompt for short answer question generation"""
        return f"""You are an expert educator creating short answer questions for assessments.

CONTENT TO GENERATE QUESTIONS FROM:
{content[:4000]}

INSTRUCTIONS:
1. Generate exactly {num_questions} short answer questions
2. Difficulty level: {difficulty.upper()}
3. Questions should be answerable in 1-2 sentences
4. Focus on definitions, key terms, and specific facts
5. DO NOT use phrases like "According to the passage"

OUTPUT FORMAT (use this EXACT format for each question):
QUESTION_1:
Text: [Your short answer question here]
Expected Answer: [The expected 1-2 sentence answer]

Generate {num_questions} questions now:"""
    
    def _parse_mcq_response(self, response: str) -> List[GeneratedQuestion]:
        """Parse LLM response into MCQ objects"""
        questions = []
        
        # Split by QUESTION_ markers
        question_blocks = re.split(r'QUESTION_\d+:', response)
        
        for block in question_blocks[1:]:  # Skip first empty split
            if not block.strip():
                continue
            
            try:
                # Extract question text
                text_match = re.search(r'Text:\s*(.+?)(?=\n[A-D]\))', block, re.DOTALL)
                question_text = text_match.group(1).strip() if text_match else ""
                
                # Extract options
                options = []
                for opt_id in ['A', 'B', 'C', 'D']:
                    opt_match = re.search(rf'{opt_id}\)\s*(.+?)(?=\n[A-D]\)|Correct:|$)', block, re.DOTALL)
                    if opt_match:
                        options.append({
                            "id": opt_id,
                            "text": opt_match.group(1).strip()
                        })
                
                # Extract correct answer
                correct_match = re.search(r'Correct:\s*([A-D])', block, re.IGNORECASE)
                correct_answer = correct_match.group(1).upper() if correct_match else "A"
                
                # Extract explanation
                exp_match = re.search(r'Explanation:\s*(.+?)(?=QUESTION_|$)', block, re.DOTALL)
                explanation = exp_match.group(1).strip() if exp_match else ""
                
                if question_text and len(options) >= 4:
                    questions.append(GeneratedQuestion(
                        question_type="mcq",
                        question_text=question_text,
                        options=options[:4],
                        correct_answer=correct_answer,
                        explanation=explanation,
                        key_points=[],
                        difficulty="medium",
                        source_content="",
                        marks=self.QUESTION_CONFIGS["mcq"]["marks"],
                        time_estimate_seconds=self.QUESTION_CONFIGS["mcq"]["time_seconds"]
                    ))
            except Exception as e:
                logger.warning(f"[QuestionGenerator] Failed to parse MCQ block: {e}")
                continue
        
        return questions
    
    def _parse_descriptive_response(self, response: str) -> List[GeneratedQuestion]:
        """Parse LLM response into descriptive question objects"""
        questions = []
        
        # Split by QUESTION_ markers
        question_blocks = re.split(r'QUESTION_\d+:', response)
        
        for block in question_blocks[1:]:
            if not block.strip():
                continue
            
            try:
                # Extract question text
                text_match = re.search(r'Text:\s*(.+?)(?=Key Points:|$)', block, re.DOTALL)
                question_text = text_match.group(1).strip() if text_match else ""
                
                # Extract key points
                key_points = []
                kp_match = re.search(r'Key Points:\s*(.+?)(?=Model Answer:|$)', block, re.DOTALL)
                if kp_match:
                    kp_text = kp_match.group(1)
                    key_points = [
                        point.strip().lstrip('-').strip()
                        for point in kp_text.split('\n')
                        if point.strip() and point.strip() != '-'
                    ]
                
                # Extract model answer
                answer_match = re.search(r'Model Answer:\s*(.+?)(?=QUESTION_|$)', block, re.DOTALL)
                model_answer = answer_match.group(1).strip() if answer_match else ""
                
                if question_text:
                    questions.append(GeneratedQuestion(
                        question_type="descriptive",
                        question_text=question_text,
                        options=[],
                        correct_answer=model_answer,
                        explanation=model_answer,
                        key_points=key_points,
                        difficulty="medium",
                        source_content="",
                        marks=self.QUESTION_CONFIGS["descriptive"]["marks"],
                        time_estimate_seconds=self.QUESTION_CONFIGS["descriptive"]["time_seconds"]
                    ))
            except Exception as e:
                logger.warning(f"[QuestionGenerator] Failed to parse descriptive block: {e}")
                continue
        
        return questions
    
    def _parse_short_answer_response(self, response: str) -> List[GeneratedQuestion]:
        """Parse LLM response into short answer question objects"""
        questions = []
        
        # Split by QUESTION_ markers
        question_blocks = re.split(r'QUESTION_\d+:', response)
        
        for block in question_blocks[1:]:
            if not block.strip():
                continue
            
            try:
                # Extract question text
                text_match = re.search(r'Text:\s*(.+?)(?=Expected Answer:|$)', block, re.DOTALL)
                question_text = text_match.group(1).strip() if text_match else ""
                
                # Extract expected answer
                answer_match = re.search(r'Expected Answer:\s*(.+?)(?=QUESTION_|$)', block, re.DOTALL)
                expected_answer = answer_match.group(1).strip() if answer_match else ""
                
                if question_text:
                    questions.append(GeneratedQuestion(
                        question_type="short_answer",
                        question_text=question_text,
                        options=[],
                        correct_answer=expected_answer,
                        explanation=expected_answer,
                        key_points=[expected_answer] if expected_answer else [],
                        difficulty="medium",
                        source_content="",
                        marks=self.QUESTION_CONFIGS["short_answer"]["marks"],
                        time_estimate_seconds=self.QUESTION_CONFIGS["short_answer"]["time_seconds"]
                    ))
            except Exception as e:
                logger.warning(f"[QuestionGenerator] Failed to parse short answer block: {e}")
                continue
        
        return questions
    
    async def generate_questions(
        self,
        content: str,
        question_type: str = "mcq",
        num_questions: int = 5,
        difficulty: str = "medium"
    ) -> List[GeneratedQuestion]:
        """
        Generate questions from provided content.
        
        Args:
            content: Text content to generate questions from
            question_type: 'mcq', 'descriptive', or 'short_answer'
            num_questions: Number of questions to generate
            difficulty: 'easy', 'medium', or 'hard'
            
        Returns:
            List of GeneratedQuestion objects
        """
        if not content or len(content) < 50:
            logger.warning("[QuestionGenerator] Content too short for question generation")
            return []
        
        # Create appropriate prompt
        if question_type == "mcq":
            prompt = self._create_mcq_prompt(content, num_questions, difficulty)
        elif question_type == "descriptive":
            prompt = self._create_descriptive_prompt(content, num_questions, difficulty)
        else:  # short_answer
            prompt = self._create_short_answer_prompt(content, num_questions, difficulty)
        
        try:
            logger.info(f"[QuestionGenerator] Generating {num_questions} {question_type} questions ({difficulty})")
            
            # Call LLM
            messages = [
                {"role": "system", "content": "You are an expert educator who creates high-quality assessment questions. Follow the exact format specified."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=2000,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content or ""
            logger.info(f"[QuestionGenerator] Received response ({len(response_text)} chars)")
            
            # Parse response based on question type
            if question_type == "mcq":
                questions = self._parse_mcq_response(response_text)
            elif question_type == "descriptive":
                questions = self._parse_descriptive_response(response_text)
            else:
                questions = self._parse_short_answer_response(response_text)
            
            # Update difficulty and source content
            for q in questions:
                q.difficulty = difficulty
                q.source_content = content[:500]  # Store preview
            
            logger.info(f"[QuestionGenerator] Successfully parsed {len(questions)} questions")
            return questions
            
        except Exception as e:
            logger.error(f"[QuestionGenerator] Error generating questions: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def generate_from_classroom_materials(
        self,
        classroom_id: str,
        topic_id: Optional[str] = None,
        question_type: str = "mcq",
        num_questions: int = 5,
        difficulty: str = "medium"
    ) -> List[GeneratedQuestion]:
        """
        Generate questions from classroom materials stored in Qdrant.
        
        Args:
            classroom_id: Classroom ID to search materials in
            topic_id: Optional topic ID to filter content
            question_type: 'mcq', 'descriptive', or 'short_answer'
            num_questions: Number of questions to generate
            difficulty: 'easy', 'medium', or 'hard'
            
        Returns:
            List of GeneratedQuestion objects
        """
        try:
            # Search for relevant content in Qdrant
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Build filter
            must_conditions = [
                FieldCondition(
                    key="classroom_id",
                    match=MatchValue(value=classroom_id)
                )
            ]
            
            # Get a sample query to search for relevant chunks
            sample_query = f"educational content for assessment questions"
            if topic_id:
                sample_query = f"topic {topic_id} content"
            
            query_embedding = self.embedder.encode(sample_query).tolist()
            
            # Search Qdrant
            results = self.qdrant.search(
                collection_name="classroom_materials",
                query_vector=query_embedding,
                query_filter=Filter(must=must_conditions),
                limit=10,
                score_threshold=0.3
            )
            
            if not results:
                logger.warning(f"[QuestionGenerator] No materials found for classroom {classroom_id}")
                return []
            
            # Combine content from chunks
            combined_content = "\n\n".join([
                hit.payload.get("content", "") or hit.payload.get("text", "")
                for hit in results
            ])
            
            logger.info(f"[QuestionGenerator] Retrieved {len(results)} chunks ({len(combined_content)} chars)")
            
            # Generate questions from combined content
            questions = await self.generate_questions(
                content=combined_content,
                question_type=question_type,
                num_questions=num_questions,
                difficulty=difficulty
            )
            
            # Add source chunk IDs
            for i, q in enumerate(questions):
                if i < len(results):
                    q.source_chunk_id = str(results[i].id)
            
            return questions
            
        except Exception as e:
            logger.error(f"[QuestionGenerator] Error generating from materials: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def generate_mixed_assessment(
        self,
        content: str,
        mcq_count: int = 5,
        descriptive_count: int = 2,
        short_answer_count: int = 3,
        difficulty: str = "medium"
    ) -> Dict[str, List[GeneratedQuestion]]:
        """
        Generate a mixed assessment with different question types.
        
        Returns:
            Dictionary with 'mcq', 'descriptive', 'short_answer' keys
        """
        result = {
            "mcq": [],
            "descriptive": [],
            "short_answer": []
        }
        
        if mcq_count > 0:
            result["mcq"] = await self.generate_questions(
                content, "mcq", mcq_count, difficulty
            )
        
        if descriptive_count > 0:
            result["descriptive"] = await self.generate_questions(
                content, "descriptive", descriptive_count, difficulty
            )
        
        if short_answer_count > 0:
            result["short_answer"] = await self.generate_questions(
                content, "short_answer", short_answer_count, difficulty
            )
        
        total = sum(len(qs) for qs in result.values())
        logger.info(f"[QuestionGenerator] Generated mixed assessment: {total} total questions")
        
        return result


# Singleton instance
_generator_instance = None


def get_question_generator() -> QuestionGeneratorService:
    """Get singleton instance of QuestionGeneratorService"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = QuestionGeneratorService()
    return _generator_instance
