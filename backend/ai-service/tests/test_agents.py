"""
Tests for AI Agents
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestModerationAgent:
    """Test moderation agent"""
    
    @pytest.mark.asyncio
    async def test_academic_message(self):
        """Test classification of academic messages"""
        from app.agents.moderation import ModerationAgent
        
        agent = ModerationAgent()
        
        result = await agent.execute({
            'message': 'Can you explain how photosynthesis works?',
            'user_id': 'user-123'
        })
        
        data = result.get('data', {})
        assert data['classification'] == 'academic'
        assert not data['was_blocked']
    
    @pytest.mark.asyncio
    async def test_off_topic_message(self):
        """Test classification of off-topic messages"""
        from app.agents.moderation import ModerationAgent
        
        agent = ModerationAgent()
        
        result = await agent.execute({
            'message': 'What pizza should I order for the party?',
            'user_id': 'user-123'
        })
        
        data = result.get('data', {})
        assert data['classification'] == 'off_topic'
    
    @pytest.mark.asyncio
    async def test_inappropriate_message(self):
        """Test blocking inappropriate messages"""
        from app.agents.moderation import ModerationAgent
        
        agent = ModerationAgent()
        
        result = await agent.execute({
            'message': 'Can you cheat on my exam for me?',
            'user_id': 'user-123'
        })
        
        data = result.get('data', {})
        assert data['classification'] == 'inappropriate'
        assert data['was_blocked']


class TestStudyPlannerAgent:
    """Test study planner agent"""
    
    @pytest.mark.asyncio
    async def test_generate_plan(self):
        """Test generating study plan"""
        from app.agents.study_planner import StudyPlannerAgent
        
        agent = StudyPlannerAgent()
        
        result = await agent.execute({
            'weak_topics': [
                {'topic': 'Photosynthesis', 'subject': 'Biology', 'confidence_score': 40},
                {'topic': 'Algebra', 'subject': 'Math', 'confidence_score': 35}
            ],
            'student_schedule': {'Monday': [9, 10, 11]},
            'upcoming_exams': []
        })
        
        data = result.get('data', {})
        assert 'timetable' in data
        assert 'prioritized_topics' in data
        assert 'recommendations' in data
    
    @pytest.mark.asyncio
    async def test_prioritize_by_urgency(self):
        """Test topic prioritization by urgency"""
        from app.agents.study_planner import StudyPlannerAgent
        
        agent = StudyPlannerAgent()
        
        result = await agent.execute({
            'weak_topics': [
                {'topic': 'Easy Topic', 'subject': 'Math', 'confidence_score': 70},
                {'topic': 'Hard Topic', 'subject': 'Math', 'confidence_score': 20}
            ]
        })
        
        data = result.get('data', {})
        prioritized = data['prioritized_topics']
        
        # Lower confidence should have higher urgency
        assert prioritized[0]['topic'] == 'Hard Topic'


class TestAssessmentAgent:
    """Test assessment generation agent"""
    
    @pytest.mark.asyncio
    async def test_generate_assessment(self):
        """Test generating assessment"""
        from app.agents.assessment_agent import AssessmentAgent
        
        with patch.object(AssessmentAgent, '_generate_questions', new_callable=AsyncMock) as mock:
            mock.return_value = [
                {
                    'question': 'What is 2+2?',
                    'options': ['3', '4', '5', '6'],
                    'correct_answer': 'B',
                    'explanation': 'Basic addition'
                }
            ]
            
            agent = AssessmentAgent()
            
            result = await agent.execute({
                'weak_topics': [
                    {'topic': 'Addition', 'subject': 'Math'}
                ],
                'num_questions': 5,
                'difficulty': 'easy'
            })
            
            data = result.get('data', {})
            assert 'assessment' in data
            assert 'questions' in data['assessment']
    
    @pytest.mark.asyncio
    async def test_empty_topics(self):
        """Test with no topics provided"""
        from app.agents.assessment_agent import AssessmentAgent
        
        agent = AssessmentAgent()
        
        result = await agent.execute({
            'weak_topics': [],
            'num_questions': 5
        })
        
        data = result.get('data', {})
        assert data['assessment']['num_questions'] == 0


class TestBaseAgent:
    """Test base agent functionality"""
    
    def test_validate_input(self):
        """Test input validation"""
        from app.agents.base_agent import BaseAgent, AgentContext
        
        class TestAgent(BaseAgent):
            async def execute(self, input_data):
                self.validate_input(input_data, ['required_field'])
                return {}
        
        agent = TestAgent(AgentContext.TUTOR)
        
        with pytest.raises(ValueError) as exc:
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                agent.execute({})
            )
        
        assert 'required_field' in str(exc.value)
    
    def test_format_output(self):
        """Test output formatting"""
        from app.agents.base_agent import BaseAgent, AgentContext
        
        class TestAgent(BaseAgent):
            async def execute(self, input_data):
                return self.format_output({'result': 'test'})
        
        agent = TestAgent(AgentContext.TUTOR)
        
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            agent.execute({})
        )
        
        assert 'timestamp' in result
        assert 'agent' in result
        assert result['data'] == {'result': 'test'}
