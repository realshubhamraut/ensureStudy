"""
Debug Logger for AI Crawler Agent

Provides:
- Colored console logging with [CRAWLER] prefix
- File saving to /tmp/debug/
- Pipeline stage tracking
- Token counting
"""
import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path


# ============================================================================
# Console Colors
# ============================================================================

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


# ============================================================================
# Debug Directory Setup
# ============================================================================

DEBUG_ROOT = Path("/tmp/debug")
DEBUG_RAW = DEBUG_ROOT / "raw"
DEBUG_CLEANED = DEBUG_ROOT / "cleaned"
DEBUG_CHUNKS = DEBUG_ROOT / "chunks"
DEBUG_LOGS = DEBUG_ROOT / "logs"


def ensure_debug_dirs():
    """Create debug directories if they don't exist."""
    for d in [DEBUG_ROOT, DEBUG_RAW, DEBUG_CLEANED, DEBUG_CHUNKS, DEBUG_LOGS]:
        d.mkdir(parents=True, exist_ok=True)


def get_file_hash(content: str) -> str:
    """Generate a short hash for file naming."""
    return hashlib.md5(content.encode()[:1000]).hexdigest()[:12]


# ============================================================================
# Crawler Logger
# ============================================================================

class CrawlerLogger:
    """Logger for the AI Crawler Agent with colored output and file saving."""
    
    PREFIX = f"{Colors.CYAN}[CRAWLER]{Colors.RESET}"
    
    def __init__(self, query: str):
        self.query = query
        self.start_time = datetime.now()
        self.stats = {
            "urls_searched": 0,
            "urls_fetched": 0,
            "chunks_created": 0,
            "vectors_inserted": 0,
            "total_tokens": 0
        }
        ensure_debug_dirs()
        
        # Session log file
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_file = DEBUG_LOGS / f"session_{timestamp}.log"
    
    def _log(self, message: str, color: str = Colors.WHITE):
        """Print colored log message."""
        formatted = f"{self.PREFIX} {color}{message}{Colors.RESET}"
        print(formatted)
        
        # Also save to log file
        with open(self.log_file, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {message}\n")
    
    def header(self, title: str):
        """Print a header line."""
        line = "═" * 50
        self._log(f"{Colors.BOLD}{line}{Colors.RESET}")
        self._log(f"{Colors.BOLD}🔍 {title}{Colors.RESET}")
        self._log(f"{Colors.BOLD}{line}{Colors.RESET}")
    
    def divider(self):
        """Print a divider line."""
        self._log(f"{Colors.DIM}{'─' * 50}{Colors.RESET}")
    
    def search_start(self, query: str):
        """Log search start."""
        self.header(f"Query: \"{query}\"")
    
    def search_results(self, results: List[Dict[str, Any]]):
        """Log search results with URLs and trust scores."""
        self._log(f"Search results ({len(results)} found):", Colors.YELLOW)
        self.stats["urls_searched"] = len(results)
        
        for i, r in enumerate(results[:5], 1):
            trust = int(r.get('trust', 0) * 100)
            url = r.get('url', 'unknown')[:60]
            self._log(f"  {i}. {url}... ({trust}%)", Colors.DIM)
    
    def fetch_start(self, url: str):
        """Log fetch start."""
        self.divider()
        self._log(f"Fetching: {url[:70]}...", Colors.BLUE)
        self.stats["urls_fetched"] += 1
    
    def save_raw(self, content: str, url: str) -> str:
        """Save raw HTML/content and log."""
        file_hash = get_file_hash(url)
        ext = ".pdf" if ".pdf" in url.lower() else ".html"
        filepath = DEBUG_RAW / f"{file_hash}{ext}"
        
        with open(filepath, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)
        
        size_kb = len(content) / 1024
        self._log(f"Raw saved: {filepath} ({size_kb:.1f} KB)", Colors.DIM)
        return str(filepath)
    
    def save_cleaned(self, text: str, url: str, token_count: int) -> str:
        """Save cleaned text and log token count."""
        file_hash = get_file_hash(url)
        filepath = DEBUG_CLEANED / f"{file_hash}.txt"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        
        self.stats["total_tokens"] += token_count
        self._log(f"Cleaned tokens: {token_count}", Colors.GREEN)
        self._log(f"Cleaned saved: {filepath}", Colors.DIM)
        return str(filepath)
    
    def save_chunks(self, chunks: List[Dict[str, Any]], url: str) -> str:
        """Save chunks and log count."""
        file_hash = get_file_hash(url)
        filepath = DEBUG_CHUNKS / f"{file_hash}_chunks.json"
        
        # Save chunks with metadata
        chunks_data = {
            "url": url,
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "index": i,
                    "tokens": c.get("metadata", {}).get("token_count", 0),
                    "text_preview": c.get("text", "")[:100] + "..."
                }
                for i, c in enumerate(chunks)
            ]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)
        
        self.stats["chunks_created"] += len(chunks)
        avg_tokens = sum(c.get("metadata", {}).get("token_count", 0) for c in chunks) / max(len(chunks), 1)
        self._log(f"Created {len(chunks)} chunks (avg {avg_tokens:.0f} tokens)", Colors.GREEN)
        self._log(f"Chunks saved: {filepath}", Colors.DIM)
        return str(filepath)
    
    def log_embedding(self, count: int, dims: int = 384):
        """Log embedding creation."""
        self._log(f"Embedded {count} chunks ({dims} dims)", Colors.MAGENTA)
    
    def log_qdrant_insert(self, count: int, collection: str = "web_resources"):
        """Log Qdrant insert."""
        self.stats["vectors_inserted"] += count
        self._log(f"Inserted {count} vectors into Qdrant ({collection}) ✓", Colors.GREEN)
    
    def log_error(self, error: str):
        """Log an error."""
        self._log(f"❌ Error: {error}", Colors.RED)
    
    def complete(self, sources_count: int):
        """Log pipeline completion."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.divider()
        self._log(f"{Colors.BOLD}═{'═' * 49}{Colors.RESET}")
        self._log(
            f"✅ Pipeline complete: {sources_count} sources, "
            f"{self.stats['chunks_created']} chunks, "
            f"{elapsed:.1f}s",
            Colors.GREEN
        )
        self._log(f"   Total tokens processed: {self.stats['total_tokens']}", Colors.DIM)
        self._log(f"   Vectors in Qdrant: {self.stats['vectors_inserted']}", Colors.DIM)
        self._log(f"   Log file: {self.log_file}", Colors.DIM)


# ============================================================================
# Utility Functions
# ============================================================================

def count_tokens(text: str) -> int:
    """Approximate token count (words * 1.3 for English)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Fallback: approximate
        return int(len(text.split()) * 1.3)
