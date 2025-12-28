"""
Enhanced Logging Utility for AI Tutor

Provides rich, colorful terminal output for debugging.
Shows: requests, clicks, scrolls, API calls, errors, and more.
"""
import json
import logging
import sys
from datetime import datetime
from typing import Optional, Any
import uuid


# ============================================================================
# ANSI Colors for Terminal
# ============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Background
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


# ============================================================================
# Logger Configuration
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Colored formatter for terminal output."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.DIM + Colors.WHITE,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BG_RED + Colors.WHITE,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Get color for level
        color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
        
        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Build colored output
        level_str = f"{color}{record.levelname:8}{Colors.RESET}"
        time_str = f"{Colors.DIM}{timestamp}{Colors.RESET}"
        name_str = f"{Colors.CYAN}{record.name}{Colors.RESET}"
        
        message = f"{time_str} {level_str} [{name_str}] {record.getMessage()}"
        
        # Add extra fields if present
        if hasattr(record, "extra_data") and record.extra_data:
            for key, value in record.extra_data.items():
                message += f"\n    {Colors.DIM}├─ {key}: {Colors.RESET}{value}"
        
        return message


def setup_logger(name: str = "ai_tutor", level: int = logging.DEBUG) -> logging.Logger:
    """Configure colored logger for terminal output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with colors
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())
    handler.setLevel(level)
    logger.addHandler(handler)
    
    return logger


# Global loggers
api_logger = setup_logger("API")
retrieval_logger = setup_logger("RETRIEVAL")
llm_logger = setup_logger("LLM")
mcp_logger = setup_logger("MCP")
moderation_logger = setup_logger("MODERATION")


# ============================================================================
# Request ID Generator
# ============================================================================

def generate_request_id() -> str:
    """Generate unique request ID."""
    return f"req_{uuid.uuid4().hex[:8]}"


# ============================================================================
# Logging Functions
# ============================================================================

def log_request_start(method: str, path: str, request_id: str, body: Optional[dict] = None):
    """Log incoming API request."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}→ INCOMING REQUEST{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    
    api_logger.info(f"{Colors.BOLD}{method} {path}{Colors.RESET}")
    api_logger.info(f"Request ID: {Colors.CYAN}{request_id}{Colors.RESET}")
    
    if body:
        # Truncate question if too long
        display_body = body.copy()
        if "question" in display_body and len(display_body["question"]) > 100:
            display_body["question"] = display_body["question"][:100] + "..."
        api_logger.info(f"Body: {json.dumps(display_body, indent=2)}")


def log_request_end(request_id: str, status: int, duration_ms: int):
    """Log completed request."""
    color = Colors.GREEN if status < 400 else Colors.RED
    print(f"\n{Colors.BOLD}{color}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{color}← RESPONSE SENT{Colors.RESET}")
    print(f"{Colors.BOLD}{color}{'='*60}{Colors.RESET}")
    
    api_logger.info(f"Request {request_id} completed: {color}{status}{Colors.RESET} in {duration_ms}ms")


def log_moderation(question: str, decision: str, confidence: float, category: str):
    """Log moderation result."""
    icon = "✅" if decision == "allow" else "⚠️" if decision == "warn" else "❌"
    color = Colors.GREEN if decision == "allow" else Colors.YELLOW if decision == "warn" else Colors.RED
    
    moderation_logger.info(f"{icon} Moderation: {color}{decision.upper()}{Colors.RESET}")
    moderation_logger.debug(f"  Category: {category}, Confidence: {confidence:.2f}")
    moderation_logger.debug(f"  Question: {question[:80]}...")


def log_retrieval_start(query: str, subject: Optional[str]):
    """Log retrieval starting."""
    retrieval_logger.info(f"🔍 Searching Qdrant for relevant content...")
    retrieval_logger.debug(f"  Query: {query[:60]}...")
    if subject:
        retrieval_logger.debug(f"  Subject filter: {subject}")


def log_retrieval_result(chunks_count: int, top_score: float, time_ms: int):
    """Log retrieval results."""
    retrieval_logger.info(f"📚 Retrieved {Colors.CYAN}{chunks_count}{Colors.RESET} chunks in {time_ms}ms")
    if chunks_count > 0:
        retrieval_logger.info(f"  Top similarity: {Colors.GREEN}{top_score:.3f}{Colors.RESET}")


def log_retrieval_chunks(chunks: list):
    """Log individual chunks (debug level)."""
    for i, chunk in enumerate(chunks[:5]):  # Show top 5
        doc_id = chunk.document_id if hasattr(chunk, 'document_id') else chunk.get('document_id', 'N/A')
        score = chunk.similarity_score if hasattr(chunk, 'similarity_score') else chunk.get('similarity_score', 0)
        text = chunk.text if hasattr(chunk, 'text') else chunk.get('text', '')
        
        retrieval_logger.debug(f"  Chunk {i+1}: [{doc_id}] score={score:.3f}")
        retrieval_logger.debug(f"    → {text[:80]}...")


def log_mcp_context(token_budget: int, tokens_used: int, chunks_used: int):
    """Log MCP context assembly."""
    usage_pct = (tokens_used / token_budget) * 100
    color = Colors.GREEN if usage_pct < 80 else Colors.YELLOW if usage_pct < 95 else Colors.RED
    
    mcp_logger.info(f"📋 Context assembled for LLM")
    mcp_logger.info(f"  Chunks used: {chunks_used}")
    mcp_logger.info(f"  Token budget: {tokens_used}/{token_budget} ({color}{usage_pct:.0f}%{Colors.RESET})")


def log_llm_start(model: str, prompt_preview: str):
    """Log LLM call starting."""
    llm_logger.info(f"🤖 Calling {Colors.MAGENTA}{model}{Colors.RESET}")
    llm_logger.debug(f"  Prompt preview: {prompt_preview[:100]}...")


def log_llm_result(answer_preview: str, confidence: float, time_ms: int):
    """Log LLM response."""
    color = Colors.GREEN if confidence >= 0.8 else Colors.YELLOW if confidence >= 0.6 else Colors.RED
    
    llm_logger.info(f"✨ Answer generated in {time_ms}ms")
    llm_logger.info(f"  Confidence: {color}{confidence:.2f}{Colors.RESET}")
    llm_logger.debug(f"  Answer: {answer_preview[:80]}...")


def log_error(error_type: str, message: str, request_id: Optional[str] = None):
    """Log error."""
    print(f"\n{Colors.BG_RED}{Colors.WHITE}{Colors.BOLD} ERROR {Colors.RESET}")
    api_logger.error(f"❌ {error_type}: {message}")
    if request_id:
        api_logger.error(f"  Request ID: {request_id}")


def log_startup(service_name: str, port: int):
    """Log service startup."""
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  🚀 {service_name} STARTED{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.RESET}")
    print(f"  📍 Running on: {Colors.CYAN}http://localhost:{port}{Colors.RESET}")
    print(f"  📋 Docs: {Colors.CYAN}http://localhost:{port}/docs{Colors.RESET}")
    print(f"\n{Colors.DIM}Waiting for requests...{Colors.RESET}\n")


def log_action(action: str, details: Optional[str] = None):
    """Log user action (click, scroll, etc)."""
    api_logger.info(f"👆 Action: {Colors.CYAN}{action}{Colors.RESET}")
    if details:
        api_logger.debug(f"  Details: {details}")


# ============================================================================
# Query Logging (Combined)
# ============================================================================

def log_query_received(
    request_id: str,
    user_id: str,
    question: str,
    subject: Optional[str] = None
) -> None:
    """Log incoming query."""
    log_request_start(
        method="POST",
        path="/api/ai-tutor/query",
        request_id=request_id,
        body={"user_id": user_id, "question": question, "subject": subject}
    )


def log_moderation_result(
    request_id: str,
    user_id: str,
    decision: str,
    confidence: float,
    category: str
) -> None:
    """Log moderation result."""
    log_moderation(
        question=f"[{request_id}]",
        decision=decision,
        confidence=confidence,
        category=category
    )


def log_retrieval_result_full(
    request_id: str,
    sources_count: int,
    top_score: float,
    retrieval_time_ms: int
) -> None:
    """Log retrieval results with request_id."""
    retrieval_logger.info(f"📚 Retrieved {Colors.CYAN}{sources_count}{Colors.RESET} chunks in {retrieval_time_ms}ms")
    if sources_count > 0:
        retrieval_logger.info(f"  Top similarity: {Colors.GREEN}{top_score:.3f}{Colors.RESET}")


def log_query_processed(
    request_id: str,
    user_id: str,
    question: str,
    subject: Optional[str],
    sources_count: int,
    confidence: float,
    retrieval_time_ms: int,
    llm_time_ms: int,
    total_time_ms: int,
    success: bool
) -> None:
    """Log completed query."""
    log_request_end(request_id, 200 if success else 500, total_time_ms)
    
    if success:
        print(f"\n{Colors.GREEN}✅ Query processed successfully{Colors.RESET}")
        print(f"  Sources: {sources_count}, Confidence: {confidence:.2f}")
        print(f"  Retrieval: {retrieval_time_ms}ms, LLM: {llm_time_ms}ms, Total: {total_time_ms}ms")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    log_startup("AI Tutor Service", 8001)
    
    req_id = generate_request_id()
    log_query_received(req_id, "usr_123", "What is Newton's first law?", "physics")
    log_moderation_result(req_id, "usr_123", "allow", 0.95, "physics")
    log_retrieval_result(req_id, 3, 0.89, 45)
    log_llm_result("Newton's first law states...", 0.87, 1200)
    log_query_processed(req_id, "usr_123", "What is Newton's first law?", "physics", 3, 0.87, 45, 1200, 1300, True)
