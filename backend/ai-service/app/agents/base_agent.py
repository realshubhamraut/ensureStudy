"""
Base Agent with MCP Protocol Support
"""
from typing import Dict, Any, List
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod


class AgentContext(Enum):
    """Bounded contexts for each agent (MCP)"""
    TUTOR = "tutor"
    STUDY_PLANNER = "study_planner"
    ASSESSMENT = "assessment"
    NOTES_GENERATOR = "notes_generator"
    MODERATION = "moderation"
    SCRAPER = "scraper"


class BaseAgent(ABC):
    """
    Base agent with Model Context Protocol (MCP) support.
    
    All agents inherit from this class and implement the execute method.
    """
    
    def __init__(self, context: AgentContext):
        self.context = context
        self.responsibilities: List[str] = []
        self.communication_channels: List[str] = []
        self.created_at = datetime.utcnow()
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main task.
        
        Args:
            input_data: Input parameters for the task
        
        Returns:
            Formatted output dictionary
        """
        pass
    
    def validate_input(self, input_data: Dict[str, Any], required_keys: List[str]) -> None:
        """
        Validate that all required keys are present in input data.
        
        Args:
            input_data: Input to validate
            required_keys: List of required keys
        
        Raises:
            ValueError: If any required keys are missing
        """
        missing = [k for k in required_keys if k not in input_data]
        if missing:
            raise ValueError(f"Missing required keys: {', '.join(missing)}")
    
    def format_output(
        self,
        data: Any,
        output_type: str = "json",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Format agent output in standard MCP format.
        
        Args:
            data: Output data
            output_type: Type of output (json, text, etc.)
            metadata: Additional metadata
        
        Returns:
            Formatted output dictionary
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.context.value,
            "output_type": output_type,
            "data": data,
            "metadata": metadata or {}
        }
    
    def log_execution(self, input_data: Dict, output: Dict, duration_ms: float) -> None:
        """Log agent execution for monitoring"""
        # In production, this would log to a monitoring system
        print(f"[{self.context.value}] Executed in {duration_ms:.2f}ms")
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "context": self.context.value,
            "responsibilities": self.responsibilities,
            "created_at": self.created_at.isoformat()
        }
