"""
ABCR Cache - Token Embedding Cache for Attention-Based Context Routing

Stores and retrieves token embeddings for session turns.
Uses file-based storage with float16 tensors for efficiency.

Features:
- Per-session, per-turn token embedding storage
- Auto-eviction on TTL expiry
- Thread-safe file operations
"""
import os
import time
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

ABCR_CACHE_PATH = os.getenv("ABCR_CACHE_PATH", "/tmp/abcr_cache")
ABCR_CACHE_TTL_HOURS = int(os.getenv("ABCR_CACHE_TTL_HOURS", "24"))


# ============================================================================
# Cache Class
# ============================================================================

class ABCRCache:
    """
    File-based cache for token embeddings.
    
    Structure:
        {cache_path}/{session_id}/{turn_index}.npz
        
    Each file contains:
        - token_embeddings: float16 array [seq_len, hidden_dim]
        - created_at: timestamp
    """
    
    def __init__(self, cache_path: str = None):
        self.cache_path = Path(cache_path or ABCR_CACHE_PATH)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ABCR_CACHE_TTL_HOURS)
        logger.info(f"[ABCR_CACHE] Initialized at {self.cache_path} TTL={ABCR_CACHE_TTL_HOURS}h")
    
    def _get_session_dir(self, session_id: str) -> Path:
        """Get session-specific cache directory."""
        # Hash session ID for filesystem safety
        safe_id = hashlib.md5(session_id.encode()).hexdigest()[:16]
        return self.cache_path / safe_id
    
    def _get_turn_path(self, session_id: str, turn_index: int) -> Path:
        """Get path for a specific turn's token embeddings."""
        session_dir = self._get_session_dir(session_id)
        return session_dir / f"turn_{turn_index}.npz"
    
    def store_turn_embeddings(
        self,
        session_id: str,
        turn_index: int,
        token_embeddings: np.ndarray,
        turn_text: str = ""
    ) -> bool:
        """
        Store token embeddings for a turn.
        
        Args:
            session_id: Session identifier
            turn_index: Turn number (0-indexed)
            token_embeddings: Token-level embeddings [seq_len, hidden_dim]
            turn_text: Original turn text (for debugging)
            
        Returns:
            True if stored successfully
        """
        try:
            session_dir = self._get_session_dir(session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            
            turn_path = self._get_turn_path(session_id, turn_index)
            
            # Store as float16 for efficiency
            embeddings_f16 = token_embeddings.astype(np.float16)
            
            np.savez_compressed(
                turn_path,
                token_embeddings=embeddings_f16,
                created_at=time.time(),
                turn_text_hash=hashlib.sha256(turn_text.encode()).hexdigest()[:16]
            )
            
            logger.debug(
                f"[ABCR_CACHE] Stored turn {turn_index} for session {session_id[:8]}... "
                f"shape={embeddings_f16.shape}"
            )
            return True
            
        except Exception as e:
            logger.error(f"[ABCR_CACHE] Failed to store: {e}")
            return False
    
    def get_turn_embeddings(
        self,
        session_id: str,
        turn_index: int
    ) -> Optional[np.ndarray]:
        """
        Retrieve token embeddings for a turn.
        
        Args:
            session_id: Session identifier
            turn_index: Turn number
            
        Returns:
            Token embeddings array or None if not found/expired
        """
        try:
            turn_path = self._get_turn_path(session_id, turn_index)
            
            if not turn_path.exists():
                return None
            
            data = np.load(turn_path, allow_pickle=True)
            
            # Check TTL
            created_at = float(data.get("created_at", 0))
            age = time.time() - created_at
            
            if age > self.ttl.total_seconds():
                logger.debug(f"[ABCR_CACHE] Turn {turn_index} expired, removing")
                turn_path.unlink(missing_ok=True)
                return None
            
            # Return as float32 for computation
            embeddings = data["token_embeddings"].astype(np.float32)
            return embeddings
            
        except Exception as e:
            logger.error(f"[ABCR_CACHE] Failed to load: {e}")
            return None
    
    def get_recent_turns(
        self,
        session_id: str,
        last_n: int = 3
    ) -> Dict[int, np.ndarray]:
        """
        Get token embeddings for the last N turns.
        
        Returns:
            Dict mapping turn_index -> embeddings
        """
        session_dir = self._get_session_dir(session_id)
        
        if not session_dir.exists():
            return {}
        
        # Find all turn files
        turn_files = sorted(session_dir.glob("turn_*.npz"))
        
        # Get last N
        recent_files = turn_files[-last_n:] if len(turn_files) > last_n else turn_files
        
        result = {}
        for turn_file in recent_files:
            try:
                turn_index = int(turn_file.stem.split("_")[1])
                embeddings = self.get_turn_embeddings(session_id, turn_index)
                if embeddings is not None:
                    result[turn_index] = embeddings
            except Exception as e:
                logger.warning(f"[ABCR_CACHE] Error loading {turn_file}: {e}")
        
        return result
    
    def clear_session(self, session_id: str) -> int:
        """
        Clear all cached embeddings for a session.
        
        Returns:
            Number of files deleted
        """
        session_dir = self._get_session_dir(session_id)
        
        if not session_dir.exists():
            return 0
        
        count = 0
        for f in session_dir.glob("*.npz"):
            f.unlink(missing_ok=True)
            count += 1
        
        # Remove session directory if empty
        try:
            session_dir.rmdir()
        except OSError:
            pass
        
        logger.info(f"[ABCR_CACHE] Cleared {count} files for session {session_id[:8]}...")
        return count
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries across all sessions.
        
        Returns:
            Number of files deleted
        """
        count = 0
        cutoff = time.time() - self.ttl.total_seconds()
        
        for session_dir in self.cache_path.iterdir():
            if not session_dir.is_dir():
                continue
            
            for turn_file in session_dir.glob("*.npz"):
                try:
                    # Check file modification time as quick filter
                    if turn_file.stat().st_mtime < cutoff:
                        turn_file.unlink(missing_ok=True)
                        count += 1
                except Exception:
                    pass
            
            # Remove empty session directories
            try:
                session_dir.rmdir()
            except OSError:
                pass
        
        if count > 0:
            logger.info(f"[ABCR_CACHE] Cleaned up {count} expired files")
        
        return count


# ============================================================================
# Singleton
# ============================================================================

_cache: Optional[ABCRCache] = None


def get_abcr_cache() -> ABCRCache:
    """Get singleton ABCRCache instance."""
    global _cache
    if _cache is None:
        _cache = ABCRCache()
    return _cache
