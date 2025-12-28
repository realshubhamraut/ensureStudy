"""
Pydantic Schemas for AI Tutor API
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from enum import Enum


# ============================================================================
# Request Schemas
# ============================================================================

class ResponseMode(str, Enum):
    """Response detail level."""
    SHORT = "short"
    DETAILED = "detailed"


class LanguageStyle(str, Enum):
    """Language complexity level."""
    SCIENTIFIC = "scientific"  # Professional, technical language
    LAYMAN = "layman"          # Everyday language, easy to understand
    SIMPLE = "simple"          # Simplest possible explanation, ELI5 style


class SubjectType(str, Enum):
    """Academic subject categories."""
    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    HISTORY = "history"
    ENGLISH = "english"
    GENERAL = "general"


class ConversationMessage(BaseModel):
    """A message in conversation history."""
    role: Literal["user", "assistant"]
    content: str


class TutorQueryRequest(BaseModel):
    """Request schema for AI Tutor query endpoint."""
    
    user_id: str = Field(..., min_length=1, description="Student user ID")
    question: str = Field(..., min_length=3, max_length=1000, description="Academic question")
    subject: Optional[SubjectType] = Field(None, description="Subject hint for retrieval")
    classroom_id: Optional[str] = Field(None, description="Classroom ID for material-specific context")
    response_mode: ResponseMode = Field(ResponseMode.SHORT, description="Response detail level")
    language_style: LanguageStyle = Field(LanguageStyle.LAYMAN, description="Language complexity level")
    find_resources: bool = Field(False, description="Whether to fetch web resources (YouTube, images, articles)")
    conversation_history: Optional[List[ConversationMessage]] = Field(
        None, 
        description="Previous messages for context (enables follow-up questions)"
    )
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Ensure question is meaningful."""
        if len(v.strip()) < 3:
            raise ValueError("Question must be at least 3 characters")
        return v.strip()
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "user_id": "usr_student_001",
                "question": "What is Newton's first law of motion?",
                "subject": "physics",
                "response_mode": "short",
                "language_style": "layman",
                "find_resources": True,
                "conversation_history": [
                    {"role": "user", "content": "Explain Newton's laws"},
                    {"role": "assistant", "content": "Newton formulated three laws of motion..."}
                ]
            }]
        }
    }


# ============================================================================
# Response Schemas
# ============================================================================

class SourceInfo(BaseModel):
    """Information about a retrieved source."""
    
    document_id: str
    chunk_id: str
    title: Optional[str] = None
    url: Optional[str] = None
    page_number: int = 0
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class ResponseMetadata(BaseModel):
    """Metadata about the response generation."""
    
    tokens_used: int
    retrieval_time_ms: int
    llm_time_ms: int
    request_id: str


class TutorResponseData(BaseModel):
    """Data payload of successful response."""
    
    answer_short: str
    answer_detailed: Optional[str] = None
    sources: List[SourceInfo]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    recommended_actions: List[str]
    metadata: ResponseMetadata
    web_resources: Optional[dict] = Field(None, description="Web resources (videos, images, articles)")
    flowchart_mermaid: Optional[str] = Field(None, description="Mermaid flowchart code for visual explanation")


class TutorQueryResponse(BaseModel):
    """Full response schema for AI Tutor query."""
    
    success: bool
    data: Optional[TutorResponseData] = None
    error: Optional[dict] = None


# ============================================================================
# Internal Schemas
# ============================================================================

class ModerationResult(BaseModel):
    """Result of academic moderation check."""
    
    decision: Literal["allow", "warn", "block"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    category: str
    reason: Optional[str] = None


class RetrievedChunk(BaseModel):
    """A chunk retrieved from vector database."""
    
    document_id: str
    chunk_id: str
    text: str
    similarity_score: float
    title: Optional[str] = None
    page_number: Optional[int] = None
    url: Optional[str] = None
    metadata: dict = {}


class AssembledContext(BaseModel):
    """Context assembled for LLM input."""
    
    context_text: str
    chunks_used: List[RetrievedChunk]
    total_tokens: int
