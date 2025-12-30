"""
Session Repository - PostgreSQL persistence for tutor sessions

Provides async CRUD operations for:
- TutorSession
- SessionTurn
- SessionResource
"""
import os
import logging
from typing import Optional, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

def get_db_url(async_mode: bool = True) -> str:
    """Get database URL from environment"""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "ensure_study")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    
    if async_mode:
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


# ============================================================================
# Repository
# ============================================================================

class SessionRepository:
    """
    Async repository for session persistence.
    
    Uses raw SQL for compatibility - doesn't require models to be defined.
    """
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or get_db_url(async_mode=False)
        self._engine = None
        self._async_engine = None
        
    @property
    def engine(self):
        """Lazy load sync engine"""
        if self._engine is None:
            self._engine = create_engine(self.db_url)
        return self._engine
    
    # ========================================================================
    # Session CRUD
    # ========================================================================
    
    def save_session(self, session_data: dict) -> bool:
        """
        Save or update a session in PostgreSQL.
        
        Args:
            session_data: Dict with session_id, user_id, classroom_id, config, etc.
            
        Returns:
            True if successful
        """
        try:
            with self.engine.connect() as conn:
                # Check if exists
                result = conn.execute(
                    text("SELECT id FROM tutor_sessions WHERE id = :id"),
                    {"id": session_data["session_id"]}
                ).fetchone()
                
                if result:
                    # Update
                    conn.execute(text("""
                        UPDATE tutor_sessions 
                        SET last_active_at = :last_active_at,
                            config = :config::jsonb,
                            is_active = :is_active
                        WHERE id = :id
                    """), {
                        "id": session_data["session_id"],
                        "last_active_at": session_data.get("last_active_at", datetime.utcnow()),
                        "config": str(session_data.get("config", {})).replace("'", '"'),
                        "is_active": session_data.get("is_active", True)
                    })
                else:
                    # Insert
                    conn.execute(text("""
                        INSERT INTO tutor_sessions (id, user_id, classroom_id, created_at, last_active_at, config, is_active)
                        VALUES (:id, :user_id, :classroom_id, :created_at, :last_active_at, :config::jsonb, :is_active)
                    """), {
                        "id": session_data["session_id"],
                        "user_id": session_data["user_id"],
                        "classroom_id": session_data.get("classroom_id"),
                        "created_at": session_data.get("created_at", datetime.utcnow()),
                        "last_active_at": session_data.get("last_active_at", datetime.utcnow()),
                        "config": str(session_data.get("config", {})).replace("'", '"'),
                        "is_active": session_data.get("is_active", True)
                    })
                
                conn.commit()
                logger.info(f"[DB] Saved session: {session_data['session_id'][:8]}...")
                return True
                
        except Exception as e:
            logger.error(f"[DB] Failed to save session: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """
        Load a session from PostgreSQL.
        
        Args:
            session_id: Session UUID
            
        Returns:
            Session dict or None
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT id, user_id, classroom_id, created_at, last_active_at, config, is_active
                    FROM tutor_sessions
                    WHERE id = :id AND is_active = true
                """), {"id": session_id}).fetchone()
                
                if not result:
                    return None
                
                return {
                    "session_id": str(result[0]),
                    "user_id": str(result[1]),
                    "classroom_id": str(result[2]) if result[2] else None,
                    "created_at": result[3].isoformat() if result[3] else None,
                    "last_active_at": result[4].isoformat() if result[4] else None,
                    "config": result[5] or {},
                    "is_active": result[6]
                }
                
        except Exception as e:
            logger.error(f"[DB] Failed to get session: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """Soft delete a session"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    UPDATE tutor_sessions SET is_active = false WHERE id = :id
                """), {"id": session_id})
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"[DB] Failed to delete session: {e}")
            return False
    
    # ========================================================================
    # Turn CRUD
    # ========================================================================
    
    def save_turn(self, session_id: str, turn_data: dict) -> bool:
        """
        Save a turn to PostgreSQL.
        
        Args:
            session_id: Session UUID
            turn_data: Dict with turn_number, question, question_hash, etc.
        """
        try:
            import json
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO session_turns 
                    (session_id, turn_number, question, question_hash, question_embedding, related_to_previous, relatedness_score, created_at)
                    VALUES (:session_id, :turn_number, :question, :question_hash, :embedding::jsonb, :related, :score, :created_at)
                    ON CONFLICT (session_id, turn_number) DO UPDATE SET
                        question = EXCLUDED.question,
                        related_to_previous = EXCLUDED.related_to_previous,
                        relatedness_score = EXCLUDED.relatedness_score
                """), {
                    "session_id": session_id,
                    "turn_number": turn_data["turn_number"],
                    "question": turn_data["question"],
                    "question_hash": turn_data["question_hash"],
                    "embedding": json.dumps(turn_data.get("embedding", [])),
                    "related": turn_data.get("related", False),
                    "score": turn_data.get("relatedness_score"),
                    "created_at": turn_data.get("timestamp", datetime.utcnow())
                })
                conn.commit()
                logger.info(f"[DB] Saved turn {turn_data['turn_number']} for session {session_id[:8]}...")
                return True
                
        except Exception as e:
            logger.error(f"[DB] Failed to save turn: {e}")
            return False
    
    def get_turns(self, session_id: str) -> List[dict]:
        """Load all turns for a session"""
        try:
            with self.engine.connect() as conn:
                results = conn.execute(text("""
                    SELECT turn_number, question, question_hash, question_embedding, related_to_previous, relatedness_score, created_at
                    FROM session_turns
                    WHERE session_id = :session_id
                    ORDER BY turn_number
                """), {"session_id": session_id}).fetchall()
                
                return [
                    {
                        "turn_number": r[0],
                        "question": r[1],
                        "question_hash": r[2],
                        "embedding": r[3] or [],
                        "related": r[4],
                        "relatedness_score": r[5],
                        "timestamp": r[6].isoformat() if r[6] else None
                    }
                    for r in results
                ]
                
        except Exception as e:
            logger.error(f"[DB] Failed to get turns: {e}")
            return []
    
    # ========================================================================
    # Resource CRUD
    # ========================================================================
    
    def save_resource(self, session_id: str, resource_data: dict) -> bool:
        """Save a resource to PostgreSQL"""
        try:
            import json
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO session_resources 
                    (id, session_id, resource_type, source, url, canonical_url, content_hash, title, preview_summary, inline_render, inserted_at, last_referenced_at)
                    VALUES (:id, :session_id, :type, :source, :url, :canonical_url, :hash, :title, :preview, :inline, :inserted_at, :last_ref)
                    ON CONFLICT (id) DO UPDATE SET
                        last_referenced_at = EXCLUDED.last_referenced_at
                """), {
                    "id": resource_data["resource_id"],
                    "session_id": session_id,
                    "type": resource_data["resource_type"],
                    "source": resource_data["source"],
                    "url": resource_data.get("url"),
                    "canonical_url": resource_data.get("canonical_url"),
                    "hash": resource_data.get("content_hash"),
                    "title": resource_data.get("title", ""),
                    "preview": resource_data.get("preview_summary"),
                    "inline": resource_data.get("inline_render", False),
                    "inserted_at": resource_data.get("inserted_at", datetime.utcnow()),
                    "last_ref": resource_data.get("last_referenced_at", datetime.utcnow())
                })
                conn.commit()
                logger.info(f"[DB] Saved resource {resource_data['resource_id'][:8]}... for session {session_id[:8]}...")
                return True
                
        except Exception as e:
            logger.error(f"[DB] Failed to save resource: {e}")
            return False
    
    def get_resources(self, session_id: str) -> List[dict]:
        """Load all resources for a session"""
        try:
            with self.engine.connect() as conn:
                results = conn.execute(text("""
                    SELECT id, resource_type, source, url, canonical_url, content_hash, title, preview_summary, inline_render, inserted_at, last_referenced_at
                    FROM session_resources
                    WHERE session_id = :session_id
                    ORDER BY last_referenced_at DESC
                """), {"session_id": session_id}).fetchall()
                
                return [
                    {
                        "resource_id": str(r[0]),
                        "resource_type": r[1],
                        "source": r[2],
                        "url": r[3],
                        "canonical_url": r[4],
                        "content_hash": r[5],
                        "title": r[6],
                        "preview_summary": r[7],
                        "inline_render": r[8],
                        "inserted_at": r[9].isoformat() if r[9] else None,
                        "last_referenced_at": r[10].isoformat() if r[10] else None
                    }
                    for r in results
                ]
                
        except Exception as e:
            logger.error(f"[DB] Failed to get resources: {e}")
            return []
    
    # ========================================================================
    # Cleanup
    # ========================================================================
    
    def delete_expired_sessions(self, ttl_hours: int = 24) -> int:
        """
        Delete sessions older than TTL.
        
        Returns:
            Number of sessions deleted
        """
        try:
            cutoff = datetime.utcnow() - timedelta(hours=ttl_hours)
            
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    UPDATE tutor_sessions 
                    SET is_active = false 
                    WHERE last_active_at < :cutoff AND is_active = true
                    RETURNING id
                """), {"cutoff": cutoff})
                
                deleted = result.rowcount
                conn.commit()
                
                if deleted > 0:
                    logger.info(f"[DB] Expired {deleted} sessions older than {ttl_hours}h")
                
                return deleted
                
        except Exception as e:
            logger.error(f"[DB] Failed to delete expired sessions: {e}")
            return 0
    
    def load_full_session(self, session_id: str) -> Optional[dict]:
        """
        Load session with all turns and resources.
        
        Returns:
            Complete session dict ready to restore to memory
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        session["turns"] = self.get_turns(session_id)
        session["resources"] = self.get_resources(session_id)
        session["turn_embeddings"] = [t.get("embedding", []) for t in session["turns"]]
        
        return session


# ============================================================================
# Singleton
# ============================================================================

_repository: Optional[SessionRepository] = None


def get_session_repository() -> SessionRepository:
    """Get singleton repository instance"""
    global _repository
    if _repository is None:
        _repository = SessionRepository()
    return _repository
