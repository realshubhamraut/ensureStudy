"""
API Key Manager - Rotating Key Management for External Services

Provides centralized management of API keys with:
- Comma-separated keys in .env for multiple keys per service
- Round-robin rotation for load distribution
- Per-key failure tracking with automatic skip
- Thread-safe singleton pattern
- Simple getter functions for drop-in replacement

Usage:
    from app.services.api_key_manager import get_key, mark_key_failed
    
    # Get next available key (rotates automatically)
    api_key = get_key("GROQ_API_KEY")
    
    # Mark key as failed (rate limited, invalid, etc.)
    mark_key_failed("GROQ_API_KEY", api_key)

.env format:
    # Multiple keys separated by commas (no spaces)
    GROQ_API_KEY=gsk_xxx1,gsk_xxx2,gsk_xxx3
    
    # Single keys still work (backwards compatible)
    YOUTUBE_API_KEY=single_key_here
"""

import os
import time
import logging
from typing import Dict, List, Optional
from threading import Lock
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KeyState:
    """State tracking for a single API key."""
    key: str
    use_count: int = 0
    fail_count: int = 0
    last_used: float = 0.0
    last_failed: float = 0.0
    is_disabled: bool = False


@dataclass  
class ServiceKeys:
    """Keys and rotation state for a single service."""
    keys: List[KeyState] = field(default_factory=list)
    current_index: int = 0
    lock: Lock = field(default_factory=Lock)


class APIKeyManager:
    """
    Centralized API key manager with rotating key support.
    
    Thread-safe singleton that manages multiple API keys per service.
    """
    
    _instance = None
    _lock = Lock()
    
    # Failure cooldown - how long to skip a failed key (seconds)
    FAILURE_COOLDOWN = 60  # 1 minute
    
    # Max failures before disabling key
    MAX_FAILURES = 5
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._services: Dict[str, ServiceKeys] = {}
        self._initialized = True
        logger.info("[KEY-MGR] API Key Manager initialized")
    
    def _load_keys(self, service_name: str) -> ServiceKeys:
        """Load keys from environment for a service."""
        env_value = os.getenv(service_name, "")
        
        if not env_value:
            return ServiceKeys()
        
        # Split by comma for multiple keys
        raw_keys = [k.strip() for k in env_value.split(",") if k.strip()]
        
        service_keys = ServiceKeys()
        for key in raw_keys:
            service_keys.keys.append(KeyState(key=key))
        
        if len(raw_keys) > 1:
            logger.info(f"[KEY-MGR] Loaded {len(raw_keys)} keys for {service_name}")
        
        return service_keys
    
    def get_key(self, service_name: str) -> Optional[str]:
        """
        Get next available API key for a service.
        
        Uses round-robin rotation with failure tracking.
        Returns None if no valid keys are available.
        """
        if service_name not in self._services:
            self._services[service_name] = self._load_keys(service_name)
        
        service = self._services[service_name]
        
        if not service.keys:
            return None
        
        with service.lock:
            # Try to find an available key
            attempts = len(service.keys)
            current_time = time.time()
            
            for _ in range(attempts):
                key_state = service.keys[service.current_index]
                
                # Check if key is usable
                is_usable = (
                    not key_state.is_disabled and
                    (key_state.fail_count == 0 or 
                     current_time - key_state.last_failed > self.FAILURE_COOLDOWN)
                )
                
                if is_usable:
                    # Use this key
                    key_state.use_count += 1
                    key_state.last_used = current_time
                    
                    # Rotate to next key for next call
                    service.current_index = (service.current_index + 1) % len(service.keys)
                    
                    # Short preview for logging
                    preview = f"{key_state.key[:8]}...{key_state.key[-4:]}" if len(key_state.key) > 12 else key_state.key
                    logger.debug(f"[KEY-MGR] Using key {preview} for {service_name} (use #{key_state.use_count})")
                    
                    return key_state.key
                
                # Try next key
                service.current_index = (service.current_index + 1) % len(service.keys)
            
            # All keys are in cooldown or disabled
            logger.warning(f"[KEY-MGR] No available keys for {service_name} - all in cooldown or disabled")
            
            # Return first key anyway (better than nothing)
            return service.keys[0].key if service.keys else None
    
    def mark_failed(self, service_name: str, key: str, reason: str = ""):
        """
        Mark a key as failed (rate limited, invalid, etc.).
        
        Key will be skipped for FAILURE_COOLDOWN seconds.
        After MAX_FAILURES, key will be disabled permanently.
        """
        if service_name not in self._services:
            return
        
        service = self._services[service_name]
        
        with service.lock:
            for key_state in service.keys:
                if key_state.key == key:
                    key_state.fail_count += 1
                    key_state.last_failed = time.time()
                    
                    preview = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else key
                    
                    if key_state.fail_count >= self.MAX_FAILURES:
                        key_state.is_disabled = True
                        logger.warning(f"[KEY-MGR] Key {preview} DISABLED after {self.MAX_FAILURES} failures: {reason}")
                    else:
                        logger.warning(f"[KEY-MGR] Key {preview} failed ({key_state.fail_count}/{self.MAX_FAILURES}): {reason}")
                    
                    break
    
    def reset_key(self, service_name: str, key: str):
        """Reset failure state for a key (use after successful call)."""
        if service_name not in self._services:
            return
        
        service = self._services[service_name]
        
        with service.lock:
            for key_state in service.keys:
                if key_state.key == key:
                    if key_state.fail_count > 0:
                        key_state.fail_count = 0
                        key_state.is_disabled = False
                    break
    
    def get_stats(self) -> Dict:
        """Get usage statistics for all services."""
        stats = {}
        
        for service_name, service in self._services.items():
            with service.lock:
                stats[service_name] = {
                    "total_keys": len(service.keys),
                    "active_keys": sum(1 for k in service.keys if not k.is_disabled),
                    "keys": [
                        {
                            "preview": f"{k.key[:8]}..." if len(k.key) > 8 else k.key,
                            "uses": k.use_count,
                            "failures": k.fail_count,
                            "disabled": k.is_disabled
                        }
                        for k in service.keys
                    ]
                }
        
        return stats


# Singleton instance
_manager = APIKeyManager()


# Convenience functions for easy import
def get_key(service_name: str) -> Optional[str]:
    """Get next available API key for a service."""
    return _manager.get_key(service_name)


def mark_key_failed(service_name: str, key: str, reason: str = ""):
    """Mark a key as failed."""
    _manager.mark_failed(service_name, key, reason)


def reset_key(service_name: str, key: str):
    """Reset failure state for a key."""
    _manager.reset_key(service_name, key)


def get_key_stats() -> Dict:
    """Get usage statistics."""
    return _manager.get_stats()


# Quick test
if __name__ == "__main__":
    # Test with mock env
    os.environ["TEST_API_KEY"] = "key1,key2,key3"
    
    print("Testing key rotation:")
    for i in range(6):
        key = get_key("TEST_API_KEY")
        print(f"  Call {i+1}: {key}")
    
    print("\nTesting failure tracking:")
    mark_key_failed("TEST_API_KEY", "key1", "rate limited")
    for i in range(3):
        key = get_key("TEST_API_KEY")
        print(f"  After failure, call {i+1}: {key}")
    
    print("\nStats:", get_key_stats())
