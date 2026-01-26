"""
Agent Tool Base Classes - LangGraph Tool Framework

This module provides the foundation for converting services into
LangGraph-compatible tools that agents can invoke dynamically.

Key Features:
- Type-safe tool definitions with Pydantic
- Async support for non-blocking execution
- Automatic error handling and logging
- Tool registry for dynamic tool discovery
"""
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from dataclasses import dataclass, field
from functools import wraps
from datetime import datetime
import asyncio
import logging
import inspect

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# Tool Parameter Definition
# ============================================================================

@dataclass
class ToolParameter:
    """Definition of a tool parameter"""
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


@dataclass
class ToolResult:
    """Result from a tool invocation"""
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    tool_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "tool_name": self.tool_name,
            "timestamp": self.timestamp
        }


# ============================================================================
# Agent Tool Class
# ============================================================================

@dataclass
class AgentTool:
    """
    A tool that can be invoked by agents.
    
    Tools wrap existing service functions and make them accessible
    to LangGraph agents with proper typing and error handling.
    
    Example:
        tool = AgentTool(
            name="web_search",
            description="Search the web for educational content",
            func=search_web,
            parameters=[
                ToolParameter("query", "string", "Search query"),
                ToolParameter("num_results", "integer", "Number of results", default=5)
            ]
        )
        
        result = await tool.invoke(query="photosynthesis", num_results=3)
    """
    name: str
    description: str
    func: Callable
    parameters: List[ToolParameter] = field(default_factory=list)
    category: str = "general"  # "web", "rag", "content", "media"
    is_async: bool = True
    timeout_seconds: float = 60.0
    
    def __post_init__(self):
        # Auto-detect if function is async
        self.is_async = asyncio.iscoroutinefunction(self.func)
    
    async def invoke(self, **kwargs) -> ToolResult:
        """
        Invoke the tool with given parameters.
        
        Returns:
            ToolResult with success status and data/error
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate required parameters
            for param in self.parameters:
                if param.required and param.name not in kwargs:
                    if param.default is not None:
                        kwargs[param.name] = param.default
                    else:
                        return ToolResult(
                            success=False,
                            data=None,
                            error=f"Missing required parameter: {param.name}",
                            tool_name=self.name
                        )
            
            # Invoke function
            logger.info(f"[TOOL] Invoking {self.name} with {list(kwargs.keys())}")
            
            if self.is_async:
                result = await asyncio.wait_for(
                    self.func(**kwargs),
                    timeout=self.timeout_seconds
                )
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.func(**kwargs)
                )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.info(f"[TOOL] {self.name} completed in {execution_time:.2f}ms")
            
            return ToolResult(
                success=True,
                data=result,
                execution_time_ms=execution_time,
                tool_name=self.name
            )
            
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool execution timed out after {self.timeout_seconds}s",
                tool_name=self.name
            )
        except Exception as e:
            logger.error(f"[TOOL] {self.name} error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                tool_name=self.name
            )
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert tool to JSON schema for LLM function calling"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


# ============================================================================
# Tool Decorator
# ============================================================================

def agent_tool(
    name: str,
    description: str,
    category: str = "general",
    timeout: float = 60.0
):
    """
    Decorator to convert a function into an AgentTool.
    
    Usage:
        @agent_tool(
            name="web_search",
            description="Search the web for content",
            category="web"
        )
        async def search_web(query: str, num_results: int = 5):
            ...
    """
    def decorator(func: Callable) -> AgentTool:
        # Extract parameters from function signature
        sig = inspect.signature(func)
        parameters = []
        
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue
            
            # Determine parameter type from annotation
            annotation = param.annotation
            param_type = "string"  # default
            
            if annotation != inspect.Parameter.empty:
                if annotation == int:
                    param_type = "integer"
                elif annotation == bool:
                    param_type = "boolean"
                elif annotation == float:
                    param_type = "number"
                elif annotation == list or getattr(annotation, '__origin__', None) == list:
                    param_type = "array"
                elif annotation == dict or getattr(annotation, '__origin__', None) == dict:
                    param_type = "object"
            
            # Determine if required and default value
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default
            
            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=f"Parameter: {param_name}",
                required=required,
                default=default
            ))
        
        return AgentTool(
            name=name,
            description=description,
            func=func,
            parameters=parameters,
            category=category,
            timeout_seconds=timeout
        )
    
    return decorator


# ============================================================================
# Tool Registry
# ============================================================================

class ToolRegistry:
    """
    Registry for all available agent tools.
    
    Provides:
    - Tool registration and discovery
    - Category-based filtering
    - Schema generation for LLM function calling
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, AgentTool] = {}
        return cls._instance
    
    def register(self, tool: AgentTool) -> None:
        """Register a tool"""
        self._tools[tool.name] = tool
        logger.info(f"[REGISTRY] Registered tool: {tool.name} ({tool.category})")
    
    def get(self, name: str) -> Optional[AgentTool]:
        """Get a tool by name"""
        return self._tools.get(name)
    
    def list_tools(self, category: Optional[str] = None) -> List[AgentTool]:
        """List all tools, optionally filtered by category"""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools
    
    def list_names(self, category: Optional[str] = None) -> List[str]:
        """List tool names"""
        return [t.name for t in self.list_tools(category)]
    
    def get_schemas(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get JSON schemas for all tools (for LLM function calling)"""
        return [t.to_schema() for t in self.list_tools(category)]
    
    async def invoke(self, tool_name: str, **kwargs) -> ToolResult:
        """Invoke a tool by name"""
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}",
                tool_name=tool_name
            )
        return await tool.invoke(**kwargs)


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry"""
    return ToolRegistry()


# ============================================================================
# Convenience Functions
# ============================================================================

async def invoke_tool(name: str, **kwargs) -> ToolResult:
    """Convenience function to invoke a tool by name"""
    registry = get_tool_registry()
    return await registry.invoke(name, **kwargs)


def list_available_tools(category: Optional[str] = None) -> List[str]:
    """List all available tool names"""
    registry = get_tool_registry()
    return registry.list_names(category)
