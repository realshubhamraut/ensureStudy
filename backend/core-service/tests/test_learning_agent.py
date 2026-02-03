"""
Test Suite: Type 5 Learning Agent

Tests for:
- Question effectiveness scoring
- Duplicate detection
- Learning Agent workflow
- 80% threshold auto-generation
- Memory persistence

Run: pytest backend/core-service/tests/test_learning_agent.py -v
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import hashlib
import json


# =============================================================================
# Test Question Effectiveness Scoring
# =============================================================================

class TestQuestionEffectiveness:
    """Test psychometric scoring functions"""
    
    def test_calculate_discrimination_index_good_question(self):
        """Good questions discriminate between high and low performers"""
        from backend.ai_service.app.utils.question_effectiveness import calculate_discrimination_index
        
        # High performers: 90% correct, Low performers: 30% correct
        d_index = calculate_discrimination_index(
            top_27_correct=9,
            top_27_total=10,
            bottom_27_correct=3,
            bottom_27_total=10
        )
        
        assert d_index > 0.4, "Good discrimination should be > 0.4"
        assert d_index == pytest.approx(0.6, abs=0.05)
    
    def test_calculate_discrimination_index_poor_question(self):
        """Poor questions don't discriminate"""
        from backend.ai_service.app.utils.question_effectiveness import calculate_discrimination_index
        
        # Both groups perform similarly
        d_index = calculate_discrimination_index(
            top_27_correct=5,
            top_27_total=10,
            bottom_27_correct=5,
            bottom_27_total=10
        )
        
        assert d_index < 0.2, "Poor discrimination should be < 0.2"
    
    def test_calculate_difficulty_index_easy_question(self):
        """Easy questions have high difficulty index (counterintuitive naming)"""
        from backend.ai_service.app.utils.question_effectiveness import calculate_difficulty_index
        
        # 90% correct = easy question
        diff_index = calculate_difficulty_index(correct=90, total=100)
        
        assert diff_index > 0.8, "Easy question should have index > 0.8"
    
    def test_calculate_difficulty_index_hard_question(self):
        """Hard questions have low difficulty index"""
        from backend.ai_service.app.utils.question_effectiveness import calculate_difficulty_index
        
        # 20% correct = hard question
        diff_index = calculate_difficulty_index(correct=20, total=100)
        
        assert diff_index < 0.3, "Hard question should have index < 0.3"
    
    def test_compute_effectiveness_score(self):
        """Overall effectiveness combines all metrics"""
        from backend.ai_service.app.utils.question_effectiveness import compute_effectiveness_score
        
        # Good discrimination, moderate difficulty, good distractors
        score = compute_effectiveness_score(
            discrimination_index=0.6,
            difficulty_index=0.5,
            distractor_quality=0.7,
            sample_size=100
        )
        
        assert 0 <= score <= 1, "Score should be normalized"
        assert score > 0.5, "Good question should have score > 0.5"
    
    def test_should_regenerate_question_low_effectiveness(self):
        """Questions with low effectiveness should be regenerated"""
        from backend.ai_service.app.utils.question_effectiveness import should_regenerate_question
        
        effectiveness = {
            'effectiveness_score': 0.2,
            'discrimination_index': 0.1,
            'difficulty_index': 0.9,  # Too easy
            'sample_size': 50
        }
        
        should_regen, reasons = should_regenerate_question(effectiveness)
        
        assert should_regen is True
        assert len(reasons) > 0
    
    def test_should_not_regenerate_good_question(self):
        """Good questions should not be regenerated"""
        from backend.ai_service.app.utils.question_effectiveness import should_regenerate_question
        
        effectiveness = {
            'effectiveness_score': 0.8,
            'discrimination_index': 0.5,
            'difficulty_index': 0.6,
            'sample_size': 100
        }
        
        should_regen, reasons = should_regenerate_question(effectiveness)
        
        assert should_regen is False


# =============================================================================
# Test Duplicate Detection
# =============================================================================

class TestDuplicateDetection:
    """Test multi-layer duplicate detection"""
    
    def test_normalize_question_text(self):
        """Text normalization for hashing"""
        from backend.ai_service.app.utils.duplicate_detector import normalize_question_text
        
        q1 = "What is   the CAPITAL of France?"
        q2 = "what is the capital of france?"
        
        assert normalize_question_text(q1) == normalize_question_text(q2)
    
    def test_compute_question_hash(self):
        """Hash generation is consistent"""
        from backend.ai_service.app.utils.duplicate_detector import compute_question_hash
        
        question = "What is the capital of France?"
        
        hash1 = compute_question_hash(question)
        hash2 = compute_question_hash(question)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
    
    def test_hash_detects_exact_duplicate(self):
        """Hash layer detects exact duplicates"""
        from backend.ai_service.app.utils.duplicate_detector import check_hash_duplicate
        
        question = "What is the capital of France?"
        existing = [
            {"question_text": "What is the capital of France?", "question_hash": "abc123"}
        ]
        
        # Compute actual hash
        from backend.ai_service.app.utils.duplicate_detector import compute_question_hash
        existing[0]["question_hash"] = compute_question_hash(existing[0]["question_text"])
        
        is_dup, matching_id = check_hash_duplicate(question, existing)
        
        assert is_dup is True
    
    def test_hash_allows_different_questions(self):
        """Hash layer allows different questions"""
        from backend.ai_service.app.utils.duplicate_detector import check_hash_duplicate, compute_question_hash
        
        question = "What is the capital of Germany?"
        existing = [
            {
                "question_text": "What is the capital of France?",
                "question_hash": compute_question_hash("What is the capital of France?")
            }
        ]
        
        is_dup, matching_id = check_hash_duplicate(question, existing)
        
        assert is_dup is False
    
    @pytest.mark.asyncio
    async def test_embedding_similarity_detection(self):
        """Embedding layer detects semantic duplicates"""
        from backend.ai_service.app.utils.duplicate_detector import compute_embedding_similarity
        
        question = "What is France's capital city?"
        existing = [
            {"question_text": "What is the capital of France?", "id": "q1"}
        ]
        
        # Mock embedding service
        mock_embedding = Mock()
        mock_embedding.encode = Mock(return_value=[
            [0.9, 0.1, 0.1],  # Question embedding
            [0.9, 0.1, 0.1]   # Existing question embedding (very similar)
        ])
        
        is_dup, matching_id, similarity = await compute_embedding_similarity(
            question, existing, mock_embedding
        )
        
        # Should detect as duplicate due to high similarity
        assert similarity > 0.8


# =============================================================================
# Test Learning Agent Workflow
# =============================================================================

class TestLearningAgentWorkflow:
    """Test the Type 5 Learning Agent LangGraph workflow"""
    
    @pytest.fixture
    def sample_state(self):
        """Sample initial state for agent"""
        return {
            'task_type': 'learn',
            'topic_id': 'topic-123',
            'classroom_id': 'class-456',
            'existing_questions': [
                {"id": "q1", "question_text": "What is 2+2?"},
                {"id": "q2", "question_text": "What is 3+3?"}
            ],
            'questions_attempted': 8,
            'total_questions': 10,
            'student_responses': [
                {"question_id": "q1", "is_correct": True, "response_time_ms": 5000},
                {"question_id": "q2", "is_correct": False, "response_time_ms": 8000}
            ]
        }
    
    def test_should_generate_at_80_percent(self, sample_state):
        """Agent should trigger generation at 80% threshold"""
        from backend.ai_service.app.agents.learning_agent import should_generate_questions
        
        # 8/10 = 80% threshold
        should_gen, reason = should_generate_questions(
            questions_attempted=8,
            total_questions=10,
            threshold=0.8
        )
        
        assert should_gen is True
        assert "80%" in reason or "threshold" in reason.lower()
    
    def test_should_not_generate_below_threshold(self, sample_state):
        """Agent should not generate below threshold"""
        from backend.ai_service.app.agents.learning_agent import should_generate_questions
        
        # 5/10 = 50% < 80% threshold
        should_gen, reason = should_generate_questions(
            questions_attempted=5,
            total_questions=10,
            threshold=0.8
        )
        
        assert should_gen is False
    
    @pytest.mark.asyncio
    async def test_learning_agent_execute(self, sample_state):
        """Test full agent execution"""
        from backend.ai_service.app.agents.learning_agent import LearningAgent
        
        with patch('backend.ai_service.app.agents.learning_agent.build_learning_agent_graph') as mock_graph:
            # Mock the graph compilation
            mock_compiled = AsyncMock()
            mock_compiled.ainvoke = AsyncMock(return_value={
                **sample_state,
                'generated_questions': [
                    {"question_text": "What is 4+4?", "options": ["6", "7", "8", "9"], "correct_answer": "8"}
                ],
                'should_generate': True
            })
            mock_graph.return_value = mock_compiled
            
            agent = LearningAgent()
            result = await agent.execute(sample_state)
            
            assert result['success'] is True or 'questions' in result


# =============================================================================
# Test Memory Persistence
# =============================================================================

class TestLearningMemory:
    """Test Learning Agent memory persistence"""
    
    def test_memory_model_creation(self):
        """LearningAgentMemory model can be created"""
        from backend.core_service.app.models.curriculum import LearningAgentMemory
        
        memory = LearningAgentMemory(
            topic_id='topic-123',
            calibrated_difficulty=0.6,
            target_success_rate=0.7,
            learning_iterations=5
        )
        
        assert memory.topic_id == 'topic-123'
        assert memory.calibrated_difficulty == 0.6
    
    def test_question_effectiveness_model_creation(self):
        """QuestionEffectiveness model can be created"""
        from backend.core_service.app.models.curriculum import QuestionEffectiveness
        
        eff = QuestionEffectiveness(
            question_id='q-123',
            discrimination_index=0.5,
            difficulty_index=0.6,
            effectiveness_score=0.7
        )
        
        assert eff.effectiveness_score == 0.7


# =============================================================================
# Integration Test (requires running services)
# =============================================================================

class TestIntegration:
    """Integration tests - require running backend"""
    
    @pytest.mark.skip(reason="Requires running backend services")
    @pytest.mark.asyncio
    async def test_full_learning_cycle(self):
        """Full learning cycle: submit assessment -> learn -> generate"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # 1. Submit assessment
            submit_res = await session.post(
                "http://localhost:5001/api/assessments/submit",
                json={
                    "assessment_id": "test-assessment",
                    "responses": [
                        {"question_id": "q1", "answer": "A", "is_correct": True}
                    ]
                }
            )
            assert submit_res.status == 200
            
            # 2. Check progress
            progress_res = await session.get(
                "http://localhost:5001/api/questions/progress/topic-123"
            )
            progress = await progress_res.json()
            
            # 3. Trigger generation if at threshold
            if progress.get('should_generate'):
                gen_res = await session.post(
                    "http://localhost:5001/api/questions/trigger-generation/topic-123"
                )
                gen_data = await gen_res.json()
                assert gen_data.get('success') is True


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
