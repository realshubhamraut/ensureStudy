"""
Session Telemetry - Structured logging and metrics for sessions

Tracks:
- Session lifecycle events (create, expire, delete)
- Turn events (add, relatedness)
- Resource events (append, dedup, evict)
- Aggregated metrics for dashboards
"""
import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# Telemetry Logger
# ============================================================================

class SessionTelemetry:
    """
    Structured telemetry for session events.
    
    All events are logged with [TELEMETRY] prefix for easy filtering.
    Metrics are aggregated in-memory for dashboard access.
    """
    
    def __init__(self):
        self._metrics = defaultdict(int)
        self._started_at = datetime.utcnow()
        
    # ========================================================================
    # Session Events
    # ========================================================================
    
    def log_session_created(
        self, 
        session_id: str, 
        user_id: str,
        classroom_id: Optional[str] = None
    ):
        """Log session creation"""
        self._metrics["sessions_created"] += 1
        self._metrics["sessions_active"] += 1
        logger.info(
            f"[TELEMETRY] event=session_created "
            f"session_id={session_id[:8]}... user_id={user_id[:8]}... "
            f"classroom_id={classroom_id[:8] if classroom_id else 'none'}..."
        )
    
    def log_session_loaded(self, session_id: str, source: str):
        """Log session loaded from cache/DB"""
        self._metrics["sessions_loaded"] += 1
        logger.info(
            f"[TELEMETRY] event=session_loaded "
            f"session_id={session_id[:8]}... source={source}"
        )
    
    def log_session_expired(self, session_id: str, duration_hours: float):
        """Log session expiration"""
        self._metrics["sessions_expired"] += 1
        self._metrics["sessions_active"] -= 1
        logger.info(
            f"[TELEMETRY] event=session_expired "
            f"session_id={session_id[:8]}... duration_hours={duration_hours:.1f}"
        )
    
    # ========================================================================
    # Turn Events
    # ========================================================================
    
    def log_turn_added(
        self, 
        session_id: str, 
        turn_number: int,
        related: bool,
        relatedness_score: Optional[float] = None
    ):
        """Log turn addition"""
        self._metrics["turns_total"] += 1
        if related:
            self._metrics["turns_related"] += 1
        else:
            self._metrics["turns_unrelated"] += 1
            
        logger.info(
            f"[TELEMETRY] event=turn_added "
            f"session_id={session_id[:8]}... turn={turn_number} "
            f"related={related} score={relatedness_score if relatedness_score is not None else 0:.2f}"
        )
    
    # ========================================================================
    # Resource Events
    # ========================================================================
    
    def log_resource_appended(
        self,
        session_id: str,
        resource_type: str,
        source: str,
        inserted: bool,
        reason: str
    ):
        """Log resource append attempt"""
        self._metrics["resources_total"] += 1
        if inserted:
            self._metrics["resources_inserted"] += 1
        else:
            self._metrics[f"resources_dedup_{reason}"] += 1
            
        logger.info(
            f"[TELEMETRY] event=resource_appended "
            f"session_id={session_id[:8]}... type={resource_type} source={source} "
            f"inserted={inserted} reason={reason}"
        )
    
    def log_resource_evicted(self, session_id: str, resource_title: str):
        """Log LRU eviction"""
        self._metrics["resources_evicted"] += 1
        logger.info(
            f"[TELEMETRY] event=resource_evicted "
            f"session_id={session_id[:8]}... title={resource_title}"
        )
    
    # ========================================================================
    # Cache Events
    # ========================================================================
    
    def log_cache_hit(self, session_id: str):
        """Log cache hit"""
        self._metrics["cache_hits"] += 1
        logger.debug(f"[TELEMETRY] event=cache_hit session_id={session_id[:8]}...")
    
    def log_cache_miss(self, session_id: str):
        """Log cache miss"""
        self._metrics["cache_misses"] += 1
        logger.debug(f"[TELEMETRY] event=cache_miss session_id={session_id[:8]}...")
    
    def log_db_fallback(self, session_id: str):
        """Log DB fallback after cache miss"""
        self._metrics["db_fallbacks"] += 1
        logger.info(f"[TELEMETRY] event=db_fallback session_id={session_id[:8]}...")
    
    # ========================================================================
    # Retrieval Events
    # ========================================================================
    
    def log_retrieval_priority(
        self,
        session_id: str,
        session_resources_count: int,
        classroom_hits: int,
        global_hits: int,
        web_hits: int
    ):
        """Log retrieval with session priority"""
        logger.info(
            f"[TELEMETRY] event=retrieval_priority "
            f"session_id={session_id[:8]}... session_resources={session_resources_count} "
            f"classroom={classroom_hits} global={global_hits} web={web_hits}"
        )
    
    # ========================================================================
    # Metrics
    # ========================================================================
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated metrics for dashboard.
        
        Returns:
            Dict with all telemetry metrics
        """
        uptime_seconds = (datetime.utcnow() - self._started_at).total_seconds()
        
        # Calculate rates
        cache_total = self._metrics["cache_hits"] + self._metrics["cache_misses"]
        cache_hit_rate = (
            self._metrics["cache_hits"] / cache_total 
            if cache_total > 0 else 0
        )
        
        turns_total = self._metrics["turns_total"]
        relatedness_rate = (
            self._metrics["turns_related"] / turns_total 
            if turns_total > 0 else 0
        )
        
        resources_total = self._metrics["resources_total"]
        dedup_rate = (
            (resources_total - self._metrics["resources_inserted"]) / resources_total
            if resources_total > 0 else 0
        )
        
        return {
            "uptime_seconds": int(uptime_seconds),
            "sessions": {
                "created": self._metrics["sessions_created"],
                "active": max(0, self._metrics["sessions_active"]),
                "expired": self._metrics["sessions_expired"],
                "loaded": self._metrics["sessions_loaded"]
            },
            "turns": {
                "total": turns_total,
                "related": self._metrics["turns_related"],
                "unrelated": self._metrics["turns_unrelated"],
                "relatedness_rate": round(relatedness_rate, 3)
            },
            "resources": {
                "total_attempts": resources_total,
                "inserted": self._metrics["resources_inserted"],
                "dedup_url": self._metrics["resources_dedup_duplicate_url"],
                "dedup_hash": self._metrics["resources_dedup_duplicate_hash"],
                "dedup_vector": self._metrics["resources_dedup_duplicate_vector"],
                "evicted": self._metrics["resources_evicted"],
                "dedup_rate": round(dedup_rate, 3)
            },
            "cache": {
                "hits": self._metrics["cache_hits"],
                "misses": self._metrics["cache_misses"],
                "hit_rate": round(cache_hit_rate, 3),
                "db_fallbacks": self._metrics["db_fallbacks"]
            }
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self._metrics = defaultdict(int)
        self._started_at = datetime.utcnow()
        logger.info("[TELEMETRY] Metrics reset")


# ============================================================================
# Singleton
# ============================================================================

_telemetry: Optional[SessionTelemetry] = None


def get_session_telemetry() -> SessionTelemetry:
    """Get singleton telemetry instance"""
    global _telemetry
    if _telemetry is None:
        _telemetry = SessionTelemetry()
    return _telemetry
