"""
AI Tutor Service - FastAPI Application with Comprehensive Logging
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time

from .config import settings
from .api.routes.tutor import router as tutor_router  # /api/ai-tutor (legacy)
from .api.tutor import router as tutor_chat_router  # /api/tutor (new with TAL/ABCR/MCP)
from .api.routes.chat import router as pdf_chat_router  # /api/tutor/chat (PDF-specific)
from .api.routes.anchor_routes import router as anchor_router
from .api.routes.evaluation import router as evaluation_router
from .api.routes.mock_interview import router as mock_interview_router
from .api.routes.softskills import router as softskills_router
from .api.routes.web_ingest import router as web_ingest_router
from .api.routes.indexing import router as indexing_router
from .api.routes.grading import router as grading_router
from .api.notes import router as notes_router
from .api.meetings import router as meetings_router
from .api.meeting_qa import router as meeting_qa_router
from .api.process_recording import router as process_recording_router
from .proctor.api import router as proctor_router
from .utils.logging import log_startup, log_action, log_error, Colors


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="RAG-based AI Tutor for academic question answering",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)


# ============================================================================
# Request Logging Middleware
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing."""
    start = time.time()
    # Log request
    method = request.method
    path = request.url.path
    
    if path not in ["/health", "/favicon.ico"]:
        print(f"\n{Colors.DIM}{'─'*40}{Colors.RESET}")
        print(f"{Colors.CYAN}→{Colors.RESET} {method} {path}")
    
    # Process request
    try:
        response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)
        
        if path not in ["/health", "/favicon.ico"]:
            status_color = Colors.GREEN if response.status_code < 400 else Colors.RED
            print(f"{Colors.CYAN}←{Colors.RESET} {status_color}{response.status_code}{Colors.RESET} in {duration_ms}ms")
        
        return response
    except Exception as e:
        log_error("RequestError", str(e))
        raise


# CORS middleware - allow all origins for LAN access
# Note: When using allow_origins=["*"], credentials must be False
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for LAN access
    allow_credentials=False,  # Must be False when using wildcard origin
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(tutor_router)  # /api/ai-tutor (legacy)
app.include_router(tutor_chat_router)  # /api/tutor (new with TAL/ABCR/MCP)
app.include_router(pdf_chat_router)  # /api/tutor/chat (PDF-specific)
app.include_router(anchor_router)
app.include_router(evaluation_router)
app.include_router(mock_interview_router)
app.include_router(softskills_router)
app.include_router(notes_router)
app.include_router(meetings_router)
app.include_router(meeting_qa_router)
app.include_router(process_recording_router)
app.include_router(proctor_router)
app.include_router(web_ingest_router)
app.include_router(indexing_router)
app.include_router(grading_router)


@app.on_event("startup")
async def startup_event():
    """Log service startup and optionally preload models."""
    import os
    
    log_startup(settings.APP_NAME, 8001)
    
    # Log configuration
    print(f"{Colors.DIM}Configuration:{Colors.RESET}")
    print(f"  Embedding Model: {Colors.CYAN}{settings.EMBEDDING_MODEL}{Colors.RESET}")
    print(f"  LLM Model: {Colors.CYAN}{settings.LLM_MODEL}{Colors.RESET}")
    print(f"  Qdrant: {Colors.CYAN}{settings.QDRANT_HOST}:{settings.QDRANT_PORT}{Colors.RESET}")
    print(f"  Debug Mode: {Colors.CYAN}{settings.DEBUG}{Colors.RESET}")
    
    # Optional: Preload models for faster first request
    if os.getenv("PRELOAD_MODELS", "false").lower() == "true":
        print(f"\n{Colors.CYAN}Preloading models...{Colors.RESET}")
        try:
            # Preload embedding model
            from sentence_transformers import SentenceTransformer
            print(f"  Loading {settings.EMBEDDING_MODEL}...")
            SentenceTransformer(settings.EMBEDDING_MODEL)
            print(f"  {Colors.GREEN}✓ Embedding model loaded{Colors.RESET}")
        except Exception as e:
            print(f"  {Colors.RED}✗ Embedding model failed: {e}{Colors.RESET}")
        
        # Skip classifier preload if SKIP_MODERATION is set
        if os.getenv("SKIP_MODERATION", "false").lower() != "true":
            try:
                from app.services.llm_provider import get_classifier
                print(f"  Loading classifier (bart-large-mnli)...")
                get_classifier()
                print(f"  {Colors.GREEN}✓ Classifier loaded{Colors.RESET}")
            except Exception as e:
                print(f"  {Colors.RED}✗ Classifier failed: {e}{Colors.RESET}")
    
    print()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint."""
    log_action("ROOT_ACCESS", "Someone accessed the root endpoint")
    return {
        "message": "AI Tutor Service",
        "docs": "/docs" if settings.DEBUG else "Disabled in production"
    }


# ============================================================================
# Action Logging Endpoints (for frontend debugging)
# ============================================================================

@app.post("/api/log/action")
async def log_frontend_action(request: Request):
    """
    Log frontend actions for debugging.
    
    Request body:
    {
        "action": "click" | "scroll" | "navigate" | "input",
        "target": "button_id" | "page_path" | "element_id",
        "details": "optional additional info"
    }
    """
    body = await request.json()
    action = body.get("action", "unknown")
    target = body.get("target", "unknown")
    details = body.get("details", "")
    
    icon = {
        "click": "👆",
        "scroll": "📜",
        "navigate": "🧭",
        "input": "⌨️",
        "hover": "👀",
        "error": "❌"
    }.get(action, "📌")
    
    print(f"{icon} {Colors.MAGENTA}[FRONTEND]{Colors.RESET} {action.upper()}: {target}")
    if details:
        print(f"   {Colors.DIM}→ {details}{Colors.RESET}")
    
    return {"logged": True}
