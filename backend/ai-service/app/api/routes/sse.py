"""
Server-Sent Events (SSE) endpoint for streaming resource updates to frontend.

This allows PDFs and other resources to appear dynamically while they're being crawled,
instead of requiring a page refresh.
"""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import asyncio
import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sse", tags=["sse"])

# In-memory store for active SSE connections
# Maps request_id -> queue of events
_active_streams: Dict[str, asyncio.Queue] = {}


def get_stream(request_id: str) -> Optional[asyncio.Queue]:
    """Get the event queue for a request."""
    return _active_streams.get(request_id)


def create_stream(request_id: str) -> asyncio.Queue:
    """Create a new event queue for a request."""
    queue = asyncio.Queue()
    _active_streams[request_id] = queue
    logger.info(f"[SSE] Created stream for request: {request_id}")
    return queue


def close_stream(request_id: str):
    """Close and cleanup an event stream."""
    if request_id in _active_streams:
        del _active_streams[request_id]
        logger.info(f"[SSE] Closed stream for request: {request_id}")


async def push_event(request_id: str, event_type: str, data: Dict[str, Any]):
    """Push an event to a specific request's stream."""
    queue = _active_streams.get(request_id)
    if queue:
        await queue.put({
            "event": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        })
        logger.info(f"[SSE] Pushed {event_type} to {request_id}")


async def push_pdf_update(request_id: str, pdf: Dict[str, Any]):
    """Push a PDF update to the stream."""
    await push_event(request_id, "pdf_added", {
        "type": "pdf",
        "pdf": pdf
    })


async def push_loading_status(request_id: str, status: str, progress: int = 0):
    """Push loading status update."""
    await push_event(request_id, "loading_status", {
        "status": status,
        "progress": progress
    })


async def push_complete(request_id: str, total_pdfs: int):
    """Signal that resource loading is complete."""
    await push_event(request_id, "complete", {
        "total_pdfs": total_pdfs
    })


@router.get("/resources/{request_id}")
async def stream_resources(request_id: str, request: Request):
    """
    SSE endpoint for streaming resource updates to the frontend.
    
    The frontend connects to this endpoint after sending a query,
    and receives real-time updates as PDFs are discovered and processed.
    
    Events:
    - loading_status: {"status": "Searching for PDFs...", "progress": 25}
    - pdf_added: {"pdf": {id, title, url, source, ...}}
    - complete: {"total_pdfs": 3}
    """
    
    async def event_generator():
        queue = create_stream(request_id)
        
        try:
            # Send initial connection event
            yield {
                "event": "connected",
                "data": json.dumps({
                    "request_id": request_id,
                    "message": "Connected to resource stream"
                })
            }
            
            # Keep connection alive and send events
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                
                try:
                    # Wait for events with timeout (heartbeat every 15s)
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    
                    yield {
                        "event": event["event"],
                        "data": json.dumps(event["data"])
                    }
                    
                    # If complete event, close stream
                    if event["event"] == "complete":
                        break
                        
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield {
                        "event": "heartbeat",
                        "data": json.dumps({"timestamp": datetime.utcnow().isoformat()})
                    }
                    
        except asyncio.CancelledError:
            logger.info(f"[SSE] Client disconnected: {request_id}")
        finally:
            close_stream(request_id)
    
    return EventSourceResponse(event_generator())


@router.post("/notify/{request_id}")
async def notify_stream(request_id: str, request: Request):
    """
    Internal endpoint for backend workers to push events to a stream.
    
    Request body should contain:
    {
        "event": "pdf_added" | "loading_status" | "complete",
        "data": {...}
    }
    """
    body = await request.json()
    event_type = body.get("event", "update")
    data = body.get("data", {})
    
    await push_event(request_id, event_type, data)
    
    return {"success": True, "request_id": request_id}
