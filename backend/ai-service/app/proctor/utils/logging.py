"""
Proctoring Logger - Logs proctoring events and results
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def log_proctor_event(
    session_id: str,
    event_type: str,
    details: Optional[Dict[str, Any]] = None,
    level: str = "info"
):
    """
    Log a proctoring event.
    
    Args:
        session_id: Proctoring session ID
        event_type: Type of event (start, frame, flag, stop, etc.)
        details: Optional event details
        level: Log level (debug, info, warning, error)
    """
    timestamp = datetime.utcnow().isoformat()
    
    message = f"[PROCTOR] session={session_id} event={event_type}"
    
    if details:
        # Format details for logging
        detail_str = " ".join(f"{k}={v}" for k, v in details.items())
        message += f" {detail_str}"
    
    if level == "debug":
        logger.debug(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    else:
        logger.info(message)


def log_session_start(session_id: str, assessment_id: str, student_id: str):
    """Log session start event"""
    log_proctor_event(
        session_id=session_id,
        event_type="session_start",
        details={
            "assessment_id": assessment_id,
            "student_id": student_id
        }
    )


def log_session_end(session_id: str, integrity_score: int, flags: list, frames: int):
    """Log session end event"""
    log_proctor_event(
        session_id=session_id,
        event_type="session_end",
        details={
            "integrity_score": integrity_score,
            "flags": ",".join(flags) if flags else "none",
            "frames_processed": frames
        }
    )


def log_flag_triggered(session_id: str, flag: str, value: float, threshold: float):
    """Log when a flag is triggered"""
    log_proctor_event(
        session_id=session_id,
        event_type="flag_triggered",
        details={
            "flag": flag,
            "value": round(value, 2),
            "threshold": threshold
        },
        level="warning"
    )


def log_critical_event(session_id: str, event: str, details: Optional[Dict[str, Any]] = None):
    """Log a critical proctoring event"""
    log_proctor_event(
        session_id=session_id,
        event_type=f"critical_{event}",
        details=details,
        level="warning"
    )
