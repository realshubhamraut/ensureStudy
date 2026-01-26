"""
AI Agents Package - LangGraph Agent Architecture

This package provides a complete AI agent system:

Agents:
- OrchestratorAgent: Routes queries to appropriate sub-agents
- TutorAgent: Academic Q&A with ABCR/TAL/MCP
- ResearchAgent: Web search, PDF download, YouTube
- CurriculumAgent: Personalized learning paths from syllabus
- ContentAgent: Notes, quizzes, flowcharts (via tools)

Tools (19 total):
- Web Tools: web_search, pdf_search, download_pdf, youtube_search
- RAG Tools: vector_search, rag_retrieve, index_content
- Content Tools: llm_generate, generate_questions, generate_notes
- Media Tools: ocr_extract, transcribe_audio, extract_pdf_text

Usage:
    # Via Orchestrator (recommended)
    from app.agents import get_orchestrator
    
    orchestrator = get_orchestrator()
    result = await orchestrator.chat("explain photosynthesis")
    
    # Curriculum Agent
    from app.agents import get_curriculum_agent
    
    agent = get_curriculum_agent()
    result = await agent.generate({"syllabus_id": "xxx", "user_id": "yyy"})
    
    # Tool access
    from app.agents.tools import invoke_tool
    
    result = await invoke_tool("web_search", query="machine learning")
"""
from .orchestrator import OrchestratorAgent, get_orchestrator
from .tutor_agent import TutorAgent
from .research_agent import ResearchAgent, get_research_agent
from .curriculum_agent import CurriculumAgent, get_curriculum_agent

__all__ = [
    # Main orchestrator
    "OrchestratorAgent",
    "get_orchestrator",
    # Agents
    "TutorAgent",
    "ResearchAgent",
    "get_research_agent",
    "CurriculumAgent",
    "get_curriculum_agent",
]

