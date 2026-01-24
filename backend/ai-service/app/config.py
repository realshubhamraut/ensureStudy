"""
AI Tutor Configuration Settings

Uses Hugging Face models ONLY:
- Embeddings: sentence-transformers/all-mpnet-base-v2
- LLM: google/flan-t5-base (local) or google/flan-t5-large
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Configuration for AI Tutor service."""
    
    # API Settings
    APP_NAME: str = "AI Tutor Service"
    DEBUG: bool = True
    
    # Qdrant Settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "classroom_materials"
    
    # Embedding Settings (Hugging Face - FREE, LOCAL)
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIMENSION: int = 768  # all-mpnet-base-v2 dimension
    # LLM Settings (Mistral-7B-Instruct - FREE via HuggingFace)
    # Best choice for RAG-grounded educational QA
    LLM_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.2"
    LLM_MAX_NEW_TOKENS: int = 1024  # Mistral supports longer responses
    LLM_TEMPERATURE: float = 0.3  # Slightly higher for varied responses
    LLM_USE_GPU: bool = False  # Set True if GPU available (recommended)
    
    # Retrieval Settings
    TOP_K_RESULTS: int = 8
    SIMILARITY_THRESHOLD: float = 0.5
    
    # Context Settings (MCP Token Budget)
    MAX_CONTEXT_TOKENS: int = 2000
    SHORT_MODE_TOKENS: int = 1500
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
