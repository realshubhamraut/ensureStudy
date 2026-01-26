"""
Agent Tools Package - LangGraph Tool Framework

This package provides the tool infrastructure for AI agents:

Modules:
- base_tool: AgentTool class, ToolRegistry, decorators
- web_tools: Web search, PDF download, YouTube
- rag_tools: Vector search, RAG retrieval, indexing
- content_tools: LLM generation, questions, notes, flowcharts
- media_tools: OCR, transcription, video processing

Usage:
    from app.agents.tools import get_tool_registry, invoke_tool
    
    # List all tools
    registry = get_tool_registry()
    print(registry.list_names())
    
    # Invoke a tool
    result = await invoke_tool("web_search", query="photosynthesis")
"""
from .base_tool import (
    AgentTool,
    ToolParameter,
    ToolResult,
    ToolRegistry,
    get_tool_registry,
    invoke_tool,
    list_available_tools,
    agent_tool
)

# Import tool modules to register them
from . import web_tools
from . import rag_tools
from . import content_tools
from . import media_tools

__all__ = [
    # Base classes
    "AgentTool",
    "ToolParameter", 
    "ToolResult",
    "ToolRegistry",
    # Functions
    "get_tool_registry",
    "invoke_tool",
    "list_available_tools",
    "agent_tool",
]


def get_all_tools():
    """Get all registered tools as a dict"""
    registry = get_tool_registry()
    return {t.name: t for t in registry.list_tools()}


def get_tool_schemas():
    """Get JSON schemas for all tools (for LLM function calling)"""
    registry = get_tool_registry()
    return registry.get_schemas()
