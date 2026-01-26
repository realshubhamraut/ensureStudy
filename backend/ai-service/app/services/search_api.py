"""
Search API Service - Multi-Source Web Search Integration

Provides unified interface for web search APIs:
- Serper (Google SERP) - Primary
- Brave Search - Alternative  
- DuckDuckGo - Fallback

Each provider returns top N educational URLs for crawling.
"""
import os
import logging
import re
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
import httpx

logger = logging.getLogger(__name__)


# ============================================================================
# Search Result Model
# ============================================================================

@dataclass
class SearchResult:
    """Unified search result from any provider"""
    url: str
    title: str
    snippet: str
    domain: str
    trust_score: float = 0.7
    source_type: str = "web"
    
    @classmethod
    def from_serper(cls, item: Dict[str, Any]) -> "SearchResult":
        """Create from Serper API response"""
        url = item.get("link", "")
        domain = urlparse(url).netloc if url else ""
        return cls(
            url=url,
            title=item.get("title", ""),
            snippet=item.get("snippet", ""),
            domain=domain,
            trust_score=calculate_trust_score(url),
            source_type=detect_source_type(url)
        )
    
    @classmethod
    def from_brave(cls, item: Dict[str, Any]) -> "SearchResult":
        """Create from Brave Search API response"""
        url = item.get("url", "")
        domain = urlparse(url).netloc if url else ""
        return cls(
            url=url,
            title=item.get("title", ""),
            snippet=item.get("description", ""),
            domain=domain,
            trust_score=calculate_trust_score(url),
            source_type=detect_source_type(url)
        )


# ============================================================================
# Trust Scoring (from web_ingest_service.py)
# ============================================================================

HIGH_TRUST_PATTERNS = {
    r'\.edu$': 0.95,
    r'\.edu\.': 0.93,
    r'wikipedia\.org': 0.95,
    r'britannica\.com': 0.93,
    r'khanacademy\.org': 0.92,
    r'ocw\.mit\.edu': 0.95,
    r'coursera\.org': 0.88,
    r'edx\.org': 0.88,
    r'\.gov$': 0.85,
    r'\.gov\.': 0.83,
    r'ncert\.|cbse\.|ignou\.': 0.90,
}

MEDIUM_TRUST_PATTERNS = {
    r'geeksforgeeks\.org': 0.82,
    r'tutorialspoint\.com': 0.78,
    r'w3schools\.com': 0.75,
    r'byjus\.com': 0.80,
    r'toppr\.com': 0.78,
    r'vedantu\.com': 0.78,
    r'study\.com': 0.78,
    r'scholarpedia\.org': 0.88,
    r'mathworld\.wolfram': 0.92,
    r'hyperphysics\.': 0.90,
    r'chem\.libretexts\.': 0.88,
    r'statisticshowto\.com': 0.80,
    r'simplypsychology\.org': 0.80,
    r'investopedia\.com': 0.78,
    r'sciencedirect\.com': 0.88,
}

BLOCKED_DOMAINS = {
    'pinterest.com', 'pinterest.co.uk',
    'facebook.com', 'twitter.com', 'instagram.com',
    'tiktok.com', 'linkedin.com',
    'amazon.com', 'ebay.com', 'alibaba.com',
    'quora.com',  # User-generated, unreliable
    'reddit.com',  # User-generated
}


def calculate_trust_score(url: str) -> float:
    """Calculate trust score for a URL"""
    if not url:
        return 0.5
    
    url_lower = url.lower()
    domain = urlparse(url).netloc.lower()
    
    # Check blocked domains
    for blocked in BLOCKED_DOMAINS:
        if blocked in domain:
            return 0.0
    
    # Check high-trust patterns
    for pattern, score in HIGH_TRUST_PATTERNS.items():
        if re.search(pattern, domain):
            return score
    
    # Check medium-trust patterns
    for pattern, score in MEDIUM_TRUST_PATTERNS.items():
        if re.search(pattern, domain):
            return score
    
    # Default trust for unknown domains
    return 0.6


def detect_source_type(url: str) -> str:
    """Detect source type from URL"""
    url_lower = url.lower()
    
    patterns = {
        r'wikipedia': 'encyclopedia',
        r'britannica': 'encyclopedia',
        r'\.edu': 'academic',
        r'coursera|edx|udemy': 'course',
        r'youtube': 'video',
        r'github': 'code',
        r'arxiv|researchgate|scholar': 'research',
        r'byjus|toppr|vedantu|khanacademy': 'educational',
        r'\.gov': 'government',
        r'geeksforgeeks|tutorialspoint': 'tutorial',
    }
    
    for pattern, source_type in patterns.items():
        if re.search(pattern, url_lower):
            return source_type
    
    return 'web'


# ============================================================================
# Serper API Client
# ============================================================================

class SerperSearchClient:
    """
    Google Search via Serper API
    
    Free tier: 2,500 credits on signup
    Docs: https://serper.dev/docs
    """
    
    BASE_URL = "https://google.serper.dev/search"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            logger.warning("[Serper] No API key found. Set SERPER_API_KEY in .env")
    
    async def search(
        self, 
        query: str, 
        num_results: int = 5,
        country: str = "us",
        language: str = "en"
    ) -> List[SearchResult]:
        """
        Search Google via Serper API
        
        Args:
            query: Search query
            num_results: Number of results to return (max 10)
            country: Country code (us, in, uk, etc.)
            language: Language code
            
        Returns:
            List of SearchResult objects
        """
        if not self.api_key:
            logger.error("[Serper] Cannot search without API key")
            return []
        
        # Add educational context to query
        enhanced_query = f"{query} educational explanation"
        
        payload = {
            "q": enhanced_query,
            "num": min(num_results * 2, 20),  # Get extra for filtering
            "gl": country,
            "hl": language,
        }
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            print(f"[Serper] Searching: '{query}' (enhanced: '{enhanced_query}')")
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    self.BASE_URL,
                    json=payload,
                    headers=headers
                )
                
                if response.status_code != 200:
                    logger.error(f"[Serper] API error: {response.status_code} - {response.text}")
                    return []
                
                data = response.json()
                
                # Extract organic results
                organic_results = data.get("organic", [])
                
                # Convert to SearchResult and filter
                results = []
                seen_domains = set()
                
                for item in organic_results:
                    result = SearchResult.from_serper(item)
                    
                    # Skip blocked/low-trust domains
                    if result.trust_score < 0.5:
                        continue
                    
                    # Skip duplicate domains
                    if result.domain in seen_domains:
                        continue
                    
                    seen_domains.add(result.domain)
                    results.append(result)
                    
                    if len(results) >= num_results:
                        break
                
                print(f"[Serper] ✅ Found {len(results)} quality results")
                for r in results:
                    print(f"  - [{r.source_type}] {r.domain} (trust: {r.trust_score:.2f})")
                
                return results
                
        except Exception as e:
            logger.error(f"[Serper] Search error: {e}")
            return []


# ============================================================================
# Brave Search API Client
# ============================================================================

class BraveSearchClient:
    """
    Brave Search API
    
    Free tier: 2,000 requests/month
    Docs: https://brave.com/search/api/
    """
    
    BASE_URL = "https://api.search.brave.com/res/v1/web/search"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        if not self.api_key:
            logger.warning("[Brave] No API key found. Set BRAVE_API_KEY in .env")
    
    async def search(
        self, 
        query: str, 
        num_results: int = 5
    ) -> List[SearchResult]:
        """Search using Brave API"""
        if not self.api_key:
            return []
        
        params = {
            "q": f"{query} educational",
            "count": min(num_results * 2, 20),
        }
        
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json"
        }
        
        try:
            print(f"[Brave] Searching: '{query}'")
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    self.BASE_URL,
                    params=params,
                    headers=headers
                )
                
                if response.status_code != 200:
                    logger.error(f"[Brave] API error: {response.status_code}")
                    return []
                
                data = response.json()
                web_results = data.get("web", {}).get("results", [])
                
                results = []
                seen_domains = set()
                
                for item in web_results:
                    result = SearchResult.from_brave(item)
                    
                    if result.trust_score < 0.5:
                        continue
                    
                    if result.domain in seen_domains:
                        continue
                    
                    seen_domains.add(result.domain)
                    results.append(result)
                    
                    if len(results) >= num_results:
                        break
                
                print(f"[Brave] ✅ Found {len(results)} quality results")
                return results
                
        except Exception as e:
            logger.error(f"[Brave] Search error: {e}")
            return []


# ============================================================================
# Unified Search Interface
# ============================================================================

class MultiSourceSearcher:
    """
    Unified search interface with fallback chain:
    1. Serper (Google SERP)
    2. Brave Search
    3. DuckDuckGo (instant answers only)
    """
    
    def __init__(self):
        self.serper = SerperSearchClient()
        self.brave = BraveSearchClient()
        self.provider = os.getenv("WEB_SEARCH_PROVIDER", "serper")
        self.max_results = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
    
    async def search(self, query: str, num_results: Optional[int] = None) -> List[SearchResult]:
        """
        Search using configured provider with fallback
        
        Args:
            query: Search query
            num_results: Override default max results
            
        Returns:
            List of SearchResult objects
        """
        n = num_results or self.max_results
        
        # Try Serper first
        if self.provider == "serper" and self.serper.api_key:
            results = await self.serper.search(query, n)
            if results:
                return results
            print("[Search] Serper failed, trying Brave...")
        
        # Try Brave as fallback
        if self.brave.api_key:
            results = await self.brave.search(query, n)
            if results:
                return results
            print("[Search] Brave failed...")
        
        # No results from any provider
        print("[Search] ⚠️ All search providers failed")
        return []


# ============================================================================
# Helper Functions
# ============================================================================

async def search_web(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function for web search
    
    Returns list of dicts with: url, title, snippet, domain, trust_score
    """
    searcher = MultiSourceSearcher()
    results = await searcher.search(query, num_results)
    
    return [
        {
            "url": r.url,
            "title": r.title,
            "snippet": r.snippet,
            "domain": r.domain,
            "trust_score": r.trust_score,
            "source_type": r.source_type
        }
        for r in results
    ]


async def search_pdfs(query: str, num_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search specifically for PDF documents.
    
    Uses Serper API with filetype:pdf to find educational PDFs.
    
    Args:
        query: Search topic (e.g., "inferential statistics")
        num_results: Number of PDF results to return
        
    Returns:
        List of dicts with: url, title, snippet, domain, trust_score
    """
    # Enhance query to search for PDFs
    pdf_query = f"{query} filetype:pdf"
    
    print(f"[PDF-SEARCH] Searching for PDFs: '{pdf_query}'")
    
    searcher = MultiSourceSearcher()
    # Get more results to filter for PDFs
    results = await searcher.search(pdf_query, num_results * 3)
    
    # Filter to only PDF URLs
    pdf_results = []
    for r in results:
        url_lower = r.url.lower()
        # Check if URL ends with .pdf or contains /pdf/ path
        if url_lower.endswith('.pdf') or '/pdf/' in url_lower:
            pdf_results.append({
                "url": r.url,
                "title": r.title,
                "snippet": r.snippet,
                "domain": r.domain,
                "trust_score": r.trust_score,
                "source_type": "pdf"
            })
            if len(pdf_results) >= num_results:
                break
    
    print(f"[PDF-SEARCH] ✅ Found {len(pdf_results)} PDF results")
    for r in pdf_results:
        print(f"  - {r['domain']}: {r['title'][:50]}...")
    
    return pdf_results


# Quick test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("=== Testing Web Search ===")
        results = await search_web("what is inferential statistics", 5)
        print(f"Found {len(results)} web results")
        
        print("\n=== Testing PDF Search ===")
        pdf_results = await search_pdfs("inferential statistics", 3)
        print(f"Found {len(pdf_results)} PDF results")
        for r in pdf_results:
            print(f"  [{r['trust_score']:.2f}] {r['title'][:50]}...")
            print(f"      URL: {r['url']}")
    
    asyncio.run(test())
