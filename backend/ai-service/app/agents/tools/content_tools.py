"""
Content Generation Tools - LangGraph Agent Tools

Tools for generating content: LLM responses, questions, flowcharts, summaries.
Wraps existing services: llm_provider.py, question_generator.py, flowchart_generator.py
"""
from typing import List, Dict, Any, Optional
import logging

from .base_tool import AgentTool, ToolParameter, get_tool_registry

logger = logging.getLogger(__name__)


# ============================================================================
# LLM Generate Tool
# ============================================================================

async def _llm_generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Generate text using the LLM"""
    try:
        from app.services.llm_provider import get_llm
        
        llm = get_llm()
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = llm.invoke(full_prompt)
        
        return {
            "success": True,
            "response": response.strip(),
            "prompt_tokens": len(prompt.split()),
            "response_tokens": len(response.split())
        }
    except Exception as e:
        logger.error(f"[LLM-GENERATE] Error: {e}")
        return {
            "success": False,
            "response": "",
            "error": str(e)
        }


llm_generate_tool = AgentTool(
    name="llm_generate",
    description="Generate text using the LLM (Large Language Model). Use for explanations, summaries, and content generation.",
    func=_llm_generate,
    parameters=[
        ToolParameter(
            name="prompt",
            type="string",
            description="The prompt/question to send to the LLM",
            required=True
        ),
        ToolParameter(
            name="system_prompt",
            type="string",
            description="Optional system prompt for context",
            required=False,
            default=None
        ),
        ToolParameter(
            name="max_tokens",
            type="integer",
            description="Maximum tokens in response",
            required=False,
            default=1024
        ),
        ToolParameter(
            name="temperature",
            type="number",
            description="Temperature for generation (0-1)",
            required=False,
            default=0.7
        )
    ],
    category="content"
)


# ============================================================================
# Generate Questions Tool
# ============================================================================

async def _generate_questions(
    topic: str,
    content: str = "",
    num_questions: int = 5,
    question_type: str = "mixed",
    difficulty: str = "medium"
) -> Dict[str, Any]:
    """Generate quiz questions on a topic"""
    try:
        from app.services.question_generator import generate_questions
        
        questions = await generate_questions(
            topic=topic,
            content=content,
            num_questions=num_questions,
            question_type=question_type,
            difficulty=difficulty
        )
        
        return {
            "success": True,
            "topic": topic,
            "questions": questions,
            "count": len(questions)
        }
    except Exception as e:
        logger.error(f"[GENERATE-QUESTIONS] Error: {e}")
        return {
            "success": False,
            "topic": topic,
            "questions": [],
            "error": str(e)
        }


generate_questions_tool = AgentTool(
    name="generate_questions",
    description="Generate quiz/assessment questions on a topic. Supports MCQ, short answer, and true/false.",
    func=_generate_questions,
    parameters=[
        ToolParameter(
            name="topic",
            type="string",
            description="Topic to generate questions about",
            required=True
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Optional content to base questions on",
            required=False,
            default=""
        ),
        ToolParameter(
            name="num_questions",
            type="integer",
            description="Number of questions to generate",
            required=False,
            default=5
        ),
        ToolParameter(
            name="question_type",
            type="string",
            description="Type of questions: 'mcq', 'short_answer', 'true_false', 'mixed'",
            required=False,
            default="mixed",
            enum=["mcq", "short_answer", "true_false", "mixed"]
        ),
        ToolParameter(
            name="difficulty",
            type="string",
            description="Difficulty level: 'easy', 'medium', 'hard'",
            required=False,
            default="medium",
            enum=["easy", "medium", "hard"]
        )
    ],
    category="content"
)


# ============================================================================
# Generate Flowchart Tool
# ============================================================================

async def _generate_flowchart(
    topic: str,
    content: str = "",
    chart_type: str = "concept_map"
) -> Dict[str, Any]:
    """Generate a flowchart or concept map"""
    try:
        from app.services.flowchart_generator import generate_flowchart
        
        result = await generate_flowchart(
            topic=topic,
            content=content,
            chart_type=chart_type
        )
        
        return {
            "success": True,
            "topic": topic,
            "mermaid_code": result.get("mermaid", ""),
            "nodes": result.get("nodes", []),
            "edges": result.get("edges", [])
        }
    except Exception as e:
        logger.error(f"[GENERATE-FLOWCHART] Error: {e}")
        return {
            "success": False,
            "topic": topic,
            "error": str(e)
        }


generate_flowchart_tool = AgentTool(
    name="generate_flowchart",
    description="Generate a visual flowchart or concept map for a topic. Returns Mermaid diagram code.",
    func=_generate_flowchart,
    parameters=[
        ToolParameter(
            name="topic",
            type="string",
            description="Topic to visualize",
            required=True
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Optional content to structure",
            required=False,
            default=""
        ),
        ToolParameter(
            name="chart_type",
            type="string",
            description="Type of chart: 'flowchart', 'concept_map', 'timeline'",
            required=False,
            default="concept_map",
            enum=["flowchart", "concept_map", "timeline", "mindmap"]
        )
    ],
    category="content"
)


# ============================================================================
# Summarize Content Tool
# ============================================================================

async def _summarize_content(
    content: str,
    max_length: int = 200,
    style: str = "bullet_points"
) -> Dict[str, Any]:
    """Summarize content into key points"""
    try:
        from app.services.llm_provider import get_llm
        
        llm = get_llm()
        
        style_instructions = {
            "bullet_points": "Summarize as bullet points",
            "paragraph": "Summarize as a concise paragraph",
            "key_facts": "Extract key facts as numbered points",
            "eli5": "Explain like I'm 5 years old"
        }
        
        prompt = f"""{style_instructions.get(style, 'Summarize')} (max {max_length} words):

{content}

Summary:"""
        
        response = llm.invoke(prompt)
        
        return {
            "success": True,
            "summary": response.strip(),
            "style": style,
            "original_length": len(content.split()),
            "summary_length": len(response.split())
        }
    except Exception as e:
        logger.error(f"[SUMMARIZE] Error: {e}")
        return {
            "success": False,
            "summary": "",
            "error": str(e)
        }


summarize_content_tool = AgentTool(
    name="summarize_content",
    description="Summarize text content into key points or a concise paragraph.",
    func=_summarize_content,
    parameters=[
        ToolParameter(
            name="content",
            type="string",
            description="Content to summarize",
            required=True
        ),
        ToolParameter(
            name="max_length",
            type="integer",
            description="Maximum words in summary",
            required=False,
            default=200
        ),
        ToolParameter(
            name="style",
            type="string",
            description="Summary style",
            required=False,
            default="bullet_points",
            enum=["bullet_points", "paragraph", "key_facts", "eli5"]
        )
    ],
    category="content"
)


# ============================================================================
# Generate Notes Tool
# ============================================================================

async def _generate_notes(
    topic: str,
    content: str = "",
    style: str = "structured"
) -> Dict[str, Any]:
    """Generate study notes on a topic"""
    try:
        from app.services.llm_provider import get_llm
        
        llm = get_llm()
        
        style_prompts = {
            "structured": "Create well-structured study notes with headings, subheadings, and bullet points",
            "cornell": "Create notes in Cornell note-taking format with main points, cues, and summary",
            "outline": "Create an outline format with Roman numerals and indentation",
            "flashcard": "Create flashcard-style notes with question-answer pairs"
        }
        
        base_content = content if content else f"the topic: {topic}"
        
        prompt = f"""{style_prompts.get(style, 'Create study notes')} about {topic}.

Content to use:
{base_content}

Notes:"""
        
        response = llm.invoke(prompt)
        
        return {
            "success": True,
            "topic": topic,
            "notes": response.strip(),
            "style": style
        }
    except Exception as e:
        logger.error(f"[GENERATE-NOTES] Error: {e}")
        return {
            "success": False,
            "topic": topic,
            "notes": "",
            "error": str(e)
        }


generate_notes_tool = AgentTool(
    name="generate_notes",
    description="Generate study notes on a topic in various formats (structured, Cornell, outline, flashcard).",
    func=_generate_notes,
    parameters=[
        ToolParameter(
            name="topic",
            type="string",
            description="Topic to create notes for",
            required=True
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Optional source content",
            required=False,
            default=""
        ),
        ToolParameter(
            name="style",
            type="string",
            description="Note style",
            required=False,
            default="structured",
            enum=["structured", "cornell", "outline", "flashcard"]
        )
    ],
    category="content"
)


# ============================================================================
# Register All Tools
# ============================================================================

def register_content_tools():
    """Register all content tools with the global registry"""
    registry = get_tool_registry()
    
    registry.register(llm_generate_tool)
    registry.register(generate_questions_tool)
    registry.register(generate_flowchart_tool)
    registry.register(summarize_content_tool)
    registry.register(generate_notes_tool)
    
    logger.info(f"[CONTENT-TOOLS] Registered {len(registry.list_tools('content'))} content tools")


# Auto-register on import
register_content_tools()
