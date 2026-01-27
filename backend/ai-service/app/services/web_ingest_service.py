"""
Web Knowledge Ingest Service - MULTI-WORKER PARALLEL CRAWLER

8-Worker Pipeline Architecture:
  Worker-1: Topic Extraction (NLP)
  Worker-2A: Serper API Search (Google SERP - top 5 results)
  Worker-2B: DuckDuckGo Fallback (if Serper fails)
  Worker-3: Wikipedia Search API
  Worker-4: Wikipedia Content Fetch
  Worker-5: Parallel Page Crawlers (async, 3-5 URLs)
  Worker-6: Content Cleaner (trafilatura)
  Worker-7: Chunk + Embed (MiniLM)

Multi-source search: Serper API (primary), DuckDuckGo (fallback), Wikipedia (always).
Guarantees ≥1 resource when internet available.
"""
import os
import logging
import hashlib
import time
import re
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse, quote

logger = logging.getLogger(__name__)

# User-Agent for all HTTP requests
USER_AGENT = "ensureStudy/1.0 (contact@ensurstudy.ai)"


# ============================================================================
# Dynamic Trust Scoring
# ============================================================================

# High-trust domain patterns
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

# Medium-trust educational patterns
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
}

# Source type detection
SOURCE_TYPE_PATTERNS = {
    r'wikipedia': 'encyclopedia',
    r'britannica': 'encyclopedia',
    r'\.edu': 'academic',
    r'coursera|edx|udemy': 'course',
    r'youtube': 'video',
    r'github': 'code',
    r'arxiv|researchgate|scholar': 'research',
    r'byjus|toppr|vedantu|khanacademy': 'educational',
    r'\.gov': 'government',
}


def calculate_trust_score(url: str) -> float:
    """
    Calculate trust score for a URL dynamically.
    
    Returns score from 0.5 to 0.95 based on domain patterns.
    """
    url_lower = url.lower()
    domain = urlparse(url).netloc.lower()
    
    # Check high-trust patterns first
    for pattern, score in HIGH_TRUST_PATTERNS.items():
        if re.search(pattern, domain):
            return score
    
    # Check medium-trust patterns
    for pattern, score in MEDIUM_TRUST_PATTERNS.items():
        if re.search(pattern, domain):
            return score
    
    # Check for educational indicators in URL
    if any(x in url_lower for x in ['education', 'learn', 'tutorial', 'course', 'lesson']):
        return 0.75
    
    # Check for research/academic indicators
    if any(x in url_lower for x in ['research', 'paper', 'journal', 'publication']):
        return 0.80
    
    # Default score for unknown sources
    return 0.65


def detect_source_type(url: str) -> str:
    """Detect the type of source from URL."""
    url_lower = url.lower()
    
    for pattern, source_type in SOURCE_TYPE_PATTERNS.items():
        if re.search(pattern, url_lower):
            return source_type
    
    return 'web'


def extract_source_name(url: str) -> str:
    """Extract a human-readable source name from URL."""
    domain = urlparse(url).netloc
    
    # Remove common prefixes
    domain = re.sub(r'^www\.', '', domain)
    domain = re.sub(r'^m\.', '', domain)
    
    # Known mappings
    name_map = {
        'en.wikipedia.org': 'Wikipedia',
        'wikipedia.org': 'Wikipedia',
        'khanacademy.org': 'Khan Academy',
        'britannica.com': 'Britannica',
        'coursera.org': 'Coursera',
        'edx.org': 'edX',
        'ocw.mit.edu': 'MIT OpenCourseWare',
        'geeksforgeeks.org': 'GeeksforGeeks',
        'tutorialspoint.com': 'TutorialsPoint',
        'byjus.com': 'Byjus',
        'toppr.com': 'Toppr',
        'vedantu.com': 'Vedantu',
    }
    
    if domain in name_map:
        return name_map[domain]
    
    # Capitalize first part of domain
    parts = domain.split('.')
    return parts[0].title()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class IngestedResource:
    """An ingested web resource with clean content."""
    id: str
    url: str
    title: str
    source_name: str
    source_type: str
    trust_score: float
    clean_content: str
    summary: str
    word_count: int
    chunk_count: int
    fetched_at: str
    stored_in_qdrant: bool
    error: Optional[str] = None


@dataclass
class IngestResult:
    """Result of web ingest operation."""
    success: bool
    query: str
    resources: List[IngestedResource]
    total_chunks_stored: int
    processing_time_ms: int
    error: Optional[str] = None


# ============================================================================
# Dynamic Content Search
# ============================================================================


# ============================================================================
# Worker-1: Topic Extraction (NLP)
# ============================================================================

def worker1_extract_topic(query: str, conversation_history: list = None) -> str:
    """
    WORKER-1: Extract topic from user query using NLP approach.
    
    Now context-aware! Uses conversation history for follow-up questions.
    
    Process:
    1. Detect if query is a follow-up (short, uses pronouns/references)
    2. If follow-up, extract topic from previous conversation
    3. Otherwise, extract from current query
    
    Args:
        query: Full user query
        conversation_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
        
    Returns:
        Extracted topic/keyword
    """
    from collections import Counter
    
    # Stopwords
    stopwords = {
        'help', 'me', 'understand', 'explain', 'describe', 'what', 'is', 'are', 
        'how', 'does', 'do', 'why', 'when', 'tell', 'about', 'please', 'can', 
        'you', 'could', 'would', 'show', 'teach', 'learn', 'study', 'the', 'a',
        'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'step',
        'simple', 'simpler', 'terms', 'words', 'explanation', 'context', 'more'
    }
    
    # Follow-up detection patterns
    followup_patterns = [
        'what were', 'what are', 'what is', 'how does', 'why did', 'why is',
        'tell me more', 'explain more', 'elaborate', 'continue',
        'main causes', 'main effects', 'main reasons',
        'their', 'its', 'them', 'they', 'it', 'this', 'that', 'these', 'those'
    ]
    
    # Check if query is a follow-up
    query_lower = query.lower().strip()
    is_followup = (
        len(query.split()) <= 6 and  # Short query
        any(p in query_lower for p in followup_patterns)
    )
    
    # If follow-up and we have conversation history, extract topic from history
    if is_followup and conversation_history:
        print(f"[WORKER-1] Detected follow-up question, using conversation context")
        
        # Find the main topic from previous user messages
        for msg in reversed(conversation_history):
            if msg.get('role') == 'user':
                prev_query = msg.get('content', '')
                # Extract topic from previous user query
                prev_words = prev_query.lower().split()
                prev_content = [w for w in prev_words if w not in stopwords and len(w) > 2]
                
                if prev_content:
                    # Combine previous topic with current query context
                    current_words = query_lower.split()
                    current_content = [w for w in current_words if w not in stopwords and len(w) > 2]
                    
                    # Better merging: Take strictly the educational keywords from previous + current meaningful words
                    # If previous had educational keywords, prioritize them
                    prev_edu = [w for w in prev_content if w in educational_keywords]
                    if prev_edu:
                        topic_base = prev_edu
                    else:
                        # Fallback to last few meaningful words
                        topic_base = prev_content[-3:]
                    
                    # Add current query's specifics (e.g., "causes", "consequences")
                    combined = topic_base + current_content
                    
                    # Deduplicate preserving order
                    seen = set()
                    final_words = [x for x in combined if not (x in seen or seen.add(x))]
                    
                    topic = ' '.join(final_words).strip()
                    
                    # FORCE FIX: If topic is still generic after merging, force-prepend previous topic
                    if topic.lower() in ['main causes', 'main effects', 'causes', 'effects', 'reasons', 'consequences']:
                         topic = f"{prev_query} {topic}"
                    
                    print(f"[WORKER-1] Topic extracted from context: '{topic}'")
                    print(f"[WORKER-1] Previous query was: '{prev_query[:50]}...'")
                    return topic
    
    # Educational keywords - expanded to include more terms
    educational_keywords = {
        'equation', 'equations', 'formula', 'formulas', 'theorem', 'theorems',
        'quadratic', 'linear', 'polynomial', 'calculus', 'algebra', 'geometry',
        'photosynthesis', 'respiration', 'mitosis', 'meiosis', 'cell', 'cells',
        'physics', 'chemistry', 'biology', 'force', 'energy', 'velocity',
        'atom', 'molecule', 'reaction', 'synthesis', 'war', 'revolution',
        'derivative', 'integral', 'function', 'graph', 'vector', 'matrix',
        'newton', 'newtons', 'einstein', 'darwin', 'motion', 'gravity', 'laws', 'law',
        'french', 'world', 'civil', 'american', 'industrial',  # History keywords
        'first', 'second', 'third',  # Ordinal - important for "first law"
    }
    
    # Standard extraction from current query
    import string
    # Keep apostrophes for possessive forms (newton's -> newtons)
    clean_query = query.lower().replace("'s", "s").replace("'", "")
    words = clean_query.translate(str.maketrans('', '', string.punctuation.replace("'", ""))).split()
    content_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    topic_words = []
    
    # First pass: collect all educational keywords
    for word in content_words:
        if word in educational_keywords:
            topic_words.append(word)
    
    # If no educational keywords found, use last few content words
    if not topic_words:
        topic_words = content_words[-min(3, len(content_words)):]
    else:
        # Add preceding word for context (e.g., "newton's first" -> include context)
        for i, word in enumerate(content_words):
            if word in educational_keywords and i > 0:
                prev_word = content_words[i-1]
                if prev_word not in topic_words and prev_word not in stopwords:
                    topic_words.insert(0, prev_word)
    
    topic = ' '.join(topic_words).strip()
    
    # Fallback: use last 3 words if topic is too short
    if not topic or len(topic) < 3:
        words = query.translate(str.maketrans('', '', string.punctuation)).split()
        topic = ' '.join(words[-min(3, len(words)):])
    
    print(f"[WORKER-1] Topic extracted: '{topic}'")
    return topic


# ============================================================================
# Wikipedia REST API (Primary Source)
# ============================================================================

# ============================================================================
# Worker-3: Wikipedia Search API
# ============================================================================

async def worker3_wikipedia_search(topic: str) -> Optional[Dict[str, Any]]:
    """
    WORKER-3: Search Wikipedia API to find best matching article.
    Uses action=query&list=search for robust fuzzy matching.
    
    Args:
        topic: Extracted topic/keyword
        
    Returns:
        Dict with canonical_title, url, pageid
    """
    try:
        import requests
        
        print(f"[WORKER-3] Searching Wikipedia for: '{topic}'")
        
        # Use action=query&list=search for better fuzzy matching
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": topic,
            "srlimit": 1,
            "format": "json"
        }
        
        headers = {"User-Agent": USER_AGENT}
        search_response = requests.get(search_url, params=search_params, headers=headers, timeout=10)
        
        if search_response.status_code != 200:
            print(f"[WORKER-3] ❌ Wikipedia search failed: {search_response.status_code}")
            return None
        
        search_data = search_response.json()
        
        # Extract search results
        search_results = search_data.get('query', {}).get('search', [])
        
        if not search_results:
            print(f"[WORKER-3] ❌ No Wikipedia articles found for '{topic}'")
            return None
        
        # Get best match (first result)
        best_match = search_results[0]
        canonical_title = best_match['title'].replace(' ', '_')
        pageid = best_match['pageid']
        wiki_url = f"https://en.wikipedia.org/wiki/{quote(canonical_title, safe='_-')}"
        
        print(f"[WORKER-3] ✅ Wikipedia title resolved: {canonical_title}")
        print(f"[WORKER-3] Article URL: {wiki_url}")
        
        return {
            'canonical_title': canonical_title,
            'url': wiki_url,
            'pageid': pageid,
            'title_display': best_match['title']
        }
                
    except Exception as e:
        print(f"[WORKER-3] ❌ Wikipedia Search API error: {e}")
    
    return None


# ============================================================================
# Worker-4: Wikipedia Content Fetch
# ============================================================================

async def worker4_wikipedia_content(canonical_title: str) -> Optional[Dict[str, Any]]:
    """
    WORKER-4: Fetch Wikipedia article content.
    Uses rest_v1/page/html + rest_v1/page/summary.
    
    Args:
        canonical_title: Canonical Wikipedia title (e.g. "Quadratic_equation")
        
    Returns:
        Dict with url, title, extract, content
    """
    try:
        print(f"[WORKER-4] Fetching Wikipedia article: {canonical_title}")
        
        # Fetch summary (extract, description)
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{canonical_title}"
        
        headers = {"User-Agent": USER_AGENT}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(summary_url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                extract_text = data.get('extract', '')
                
                print(f"[WORKER-4] ✅ Wikipedia article fetched: {len(extract_text)} chars")
                
                return {
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'title': data.get('title', canonical_title),
                    'extract': extract_text,
                    'description': data.get('description', ''),
                    'timestamp': data.get('timestamp', '')
                }
            else:
                print(f"[WORKER-4] ❌ Wikipedia REST API returned {response.status_code}")
                
    except Exception as e:
        print(f"[WORKER-4] ❌ Wikipedia content fetch error: {e}")
    
    return None


# ============================================================================
# Worker-5: Parallel Page Crawlers
# ============================================================================

async def worker5_parallel_crawl(urls: List[str]) -> List[Dict[str, Any]]:
    """
    WORKER-5: Fetch all URLs concurrently using async httpx.
    No serial crawling - all fetches happen in parallel.
    
    Args:
        urls: List of URLs to fetch
        
    Returns:
        List of {url, content, status_code}
    """
    print(f"[WORKER-5] Parallel crawl started for {len(urls)} URLs")
    
    async def fetch_one(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
        try:
            headers = {"User-Agent": USER_AGENT}
            response = await client.get(url, headers=headers, follow_redirects=True)
            return {
                'url': url,
                'content': response.text,
                'status_code': response.status_code
            }
        except Exception as e:
            print(f"[WORKER-5] ⚠ Fetch error for {url[:50]}: {e}")
            return {'url': url, 'content': None, 'status_code': 0}
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [fetch_one(client, url) for url in urls]
        results = await asyncio.gather(*tasks)
    
    successful = [r for r in results if r['status_code'] == 200]
    print(f"[WORKER-5] ✅ Crawled {len(successful)}/{len(urls)} URLs")
    
    return successful


# ============================================================================
# Worker-2: DuckDuckGo Discovery (1 call/sec limit)
# ============================================================================

def worker2_duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    WORKER-2: DuckDuckGo URL Discovery.
    Makes EXACTLY 1 API call per query (rate limit: 1 call/sec).
    
    ENHANCED: Now fetches from broader educational sources, not just Wikipedia/edu.
    
    Args:
        query: Full user query (not shortened keywords)
        max_results: Maximum URLs to return (default 5)
        
    Returns:
        List of search results with URLs
    """
    try:
        from ddgs import DDGS
        
        print(f"[WORKER-2] DuckDuckGo search for: {query[:50]}...")
        
        with DDGS() as ddgs:
            # ENHANCED: Search for educational content with site biases
            # Include trusted educational sources in query context
            educational_query = f"{query} site:wikipedia.org OR site:khanacademy.org OR site:britannica.com OR site:byjus.com OR site:toppr.com OR explanation tutorial"
            results = list(ddgs.text(
                educational_query,
                max_results=min(max_results + 3, 10),  # Get extra for filtering
                region='us-en',
                safesearch='on'
            ))
        
        # Filter out low-quality domains
        blocked_domains = ['pinterest', 'facebook', 'twitter', 'instagram', 'tiktok', 'reddit', 'quora', 'yahoo']
        filtered_results = [
            r for r in results 
            if r.get('href') and not any(bd in r.get('href', '').lower() for bd in blocked_domains)
        ]
        
        # Prioritize trusted educational domains
        trusted_domains = ['wikipedia', 'khanacademy', 'britannica', 'byjus', 'toppr', 'ncert', '.edu', 'coursera', 'edx', 'mit.edu']
        
        def get_priority(result):
            url = result.get('href', '').lower()
            for idx, domain in enumerate(trusted_domains):
                if domain in url:
                    return idx
            return len(trusted_domains) + 1
        
        # Sort by trust priority
        sorted_results = sorted(filtered_results, key=get_priority)
        
        urls_found = [r.get('href', '') for r in sorted_results if r.get('href')]
        print(f"[WORKER-2] ✅ DuckDuckGo URLs ({len(urls_found)}): {urls_found[:5]}")
        
        return sorted_results[:max_results]
        
    except Exception as e:
        print(f"[WORKER-2] ❌ DuckDuckGo error: {e}")
        return []


# ============================================================================
# Worker-6B: PDF Search & Download (+ pdf download modifier)
# ============================================================================

async def worker6b_pdf_search(
    topic: str,
    user_id: str = "",
    max_pdfs: int = 3,
    classroom_id: str = None
) -> List[Dict[str, Any]]:
    """
    WORKER-6B: Search for educational PDFs and download them.
    
    Appends '+ pdf download' to queries for better PDF discovery.
    Uses existing pdf_downloader infrastructure.
    
    Args:
        topic: Extracted topic/keyword from user query
        user_id: User ID for tracking downloads
        max_pdfs: Maximum number of PDFs to download (default 3)
        classroom_id: Optional classroom to store downloaded PDFs in
        
    Returns:
        List of downloaded PDF info with extracted text
    """
    import os
    import httpx
    
    print(f"\n{'='*62}")
    print(f"[WORKER-6B] 📄 PDF Search: '{topic}'")
    print(f"[WORKER-6B] 📚 Classroom ID: {classroom_id or 'None (not storing)'}")
    print(f"[WORKER-6B] 👤 User ID: {user_id or 'anonymous'}")
    print(f"{'='*62}")
    
    try:
        from .pdf_downloader import WebPDFDownloader
        from .search_api import MultiSourceSearcher
        from .pdf_extractor import pdf_extractor
        
        # Create enhanced query with '+ pdf download' modifier
        pdf_query = f"{topic} + pdf download filetype:pdf"
        print(f"[WORKER-6B] 🔍 Enhanced query: '{pdf_query}'")
        
        # Search for PDFs using multi-source searcher
        print(f"[WORKER-6B] 🌐 Searching multiple sources...")
        searcher = MultiSourceSearcher()
        results = await searcher.search(pdf_query, num_results=max_pdfs * 2)
        print(f"[WORKER-6B] 📊 Search returned {len(results)} results")
        
        # Filter to only PDF URLs
        pdf_urls = []
        for r in results:
            url = r.url.lower()
            print(f"[WORKER-6B] 🔗 Checking URL: {r.url[:80]}...")
            if url.endswith('.pdf') or 'pdf' in url:
                # Skip Wikipedia (user requested exclusion)
                if 'wikipedia.org' not in url:
                    pdf_urls.append(r.url)
                    print(f"[WORKER-6B] ✓ Found PDF: {r.url[:60]}...")
                    if len(pdf_urls) >= max_pdfs:
                        break
                else:
                    print(f"[WORKER-6B] ⊘ Skipping Wikipedia PDF")
            else:
                print(f"[WORKER-6B] ⊘ Not a PDF, skipping")
        
        if not pdf_urls:
            print(f"[WORKER-6B] ⚠ No PDFs found for: '{topic}'")
            print(f"[WORKER-6B] 💡 Try a more specific query or different topic")
            return []
        
        print(f"\n[WORKER-6B] 📥 Downloading {len(pdf_urls)} PDFs...")
        
        # Download PDFs
        downloader = WebPDFDownloader()
        download_results = await downloader.download_multiple(
            urls=pdf_urls,
            user_id=user_id,
            topic=topic,
            max_downloads=max_pdfs
        )
        
        print(f"[WORKER-6B] 📦 Download complete: {len(download_results)} files")
        
        # Extract text from downloaded PDFs
        pdf_documents = []
        for i, result in enumerate(download_results, 1):
            print(f"\n[WORKER-6B] Processing PDF {i}/{len(download_results)}: {result.file_name}")
            
            if not result.success:
                print(f"[WORKER-6B] ❌ Download failed: {result.error or 'Unknown error'}")
                continue
                
            if not result.file_path:
                print(f"[WORKER-6B] ❌ No file path returned")
                continue
                
            try:
                print(f"[WORKER-6B] 📖 Extracting text from {result.file_name}...")
                # Extract text from PDF (returns tuple: text, has_images, page_count)
                text, has_images, page_count = pdf_extractor.extract_text_from_pdf(result.file_path)
                
                if text and len(text) > 100:
                    pdf_doc = {
                        'file_path': result.file_path,
                        'file_name': result.file_name,
                        'source_url': result.source_url,
                        'file_size': result.file_size,
                        'extracted_text': text[:50000],  # Limit text size
                        'word_count': len(text.split()),
                        'page_count': page_count,
                        'has_images': has_images
                    }
                    pdf_documents.append(pdf_doc)
                    print(f"[WORKER-6B] ✅ Extracted {len(text)} chars, {len(text.split())} words, {page_count} pages")
                    
                    # Store to classroom if classroom_id provided
                    if classroom_id:
                        try:
                            core_service_url = os.getenv('CORE_SERVICE_URL', 'http://localhost:9000')
                            
                            # Build file URL for the downloaded PDF
                            file_url = f"/api/files/pdfs/{result.file_name}"
                            
                            print(f"[WORKER-6B] 💾 Storing in classroom {classroom_id}...")
                            
                            async with httpx.AsyncClient(timeout=10.0) as client:
                                response = await client.post(
                                    f"{core_service_url}/api/classroom/{classroom_id}/web-materials",
                                    headers={"X-Service-Key": "internal-ai-service"},
                                    json={
                                        "name": result.file_name,
                                        "url": file_url,
                                        "type": "application/pdf",
                                        "size": result.file_size,
                                        "source_url": result.source_url,
                                        "description": f"Web search: {topic}"
                                    }
                                )
                                
                                if response.status_code in [200, 201]:
                                    data = response.json()
                                    if data.get('duplicate'):
                                        print(f"[WORKER-6B] ⚠ Already exists in classroom")
                                    else:
                                        print(f"[WORKER-6B] ✅ Stored in classroom materials!")
                                else:
                                    print(f"[WORKER-6B] ⚠ Failed to store: {response.status_code}")
                                    
                        except Exception as store_err:
                            print(f"[WORKER-6B] ⚠ Storage error (non-fatal): {store_err}")
                else:
                    print(f"[WORKER-6B] ⚠ Insufficient text in {result.file_name} ({len(text) if text else 0} chars)")
                    
            except Exception as e:
                print(f"[WORKER-6B] ⚠ Text extraction failed for {result.file_name}: {e}")
        
        print(f"\n{'='*62}")
        print(f"[WORKER-6B] ✅ COMPLETE: {len(pdf_documents)} PDFs processed successfully")
        if classroom_id:
            print(f"[WORKER-6B] 📚 Materials stored in classroom: {classroom_id}")
        print(f"{'='*62}\n")
        
        return pdf_documents
        
    except ImportError as e:
        print(f"[WORKER-6B] ⚠ Import error (optional dep): {e}")
        logger.error(f"[WORKER-6B] Import error: {e}")
        return []
    except Exception as e:
        print(f"[WORKER-6B] ❌ PDF search error: {e}")
        logger.error(f"[WORKER-6B] PDF search error: {e}")
        import traceback
        traceback.print_exc()
        return []


def fetch_content(url: str, crawler_log=None) -> Optional[str]:
    """
    Fetch and extract clean content from URL.
    Uses trafilatura for robust extraction.
    Args:
        url: URL to fetch
        crawler_log: Optional CrawlerLogger for debug output
    """
    try:
        import trafilatura
        
        if crawler_log:
            crawler_log.fetch_start(url)
        
        # Fetch HTML
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            if crawler_log:
                crawler_log.log_error(f"Failed to fetch: {url}")
            else:
                logger.warning(f"Failed to fetch: {url}")
            return None
        
        # Save raw HTML for debugging
        if crawler_log:
            crawler_log.save_raw(downloaded, url)
        
        # Extract main content
        text = trafilatura.extract(
            downloaded,
            include_tables=True,
            include_images=False,
            include_links=False,
            include_formatting=True,
            include_comments=False,
            output_format='txt'
        )
        
        if not text or len(text) < 100:
            if crawler_log:
                crawler_log.log_error(f"Insufficient content from: {url}")
            else:
                logger.warning(f"Insufficient content from: {url}")
            return None
        
        return text
        
    except Exception as e:
        if crawler_log:
            crawler_log.log_error(f"Fetch error: {e}")
        else:
            logger.error(f"Fetch error for {url}: {e}")
        return None


# ============================================================================
# Qdrant Integration
# ============================================================================

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using MiniLM.
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
        
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return []


def store_in_qdrant(chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
    """
    Store chunks with embeddings in Qdrant.
    Returns number of chunks stored.
    """
    # WORKER-7: Store in Qdrant vector database
    print(f"[WORKER-7] Storing {len(chunks)} chunks in Qdrant...")
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance, PointStruct
        import os
        
        # Use Qdrant SERVER (not file-based) to avoid concurrent access issues
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=30)
        collection_name = "web_content"  # Separate from cache
        
        # Ensure collection exists
        collections = client.get_collections().collections
        if collection_name not in [c.name for c in collections]:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=len(embeddings[0]) if embeddings else 384,
                    distance=Distance.COSINE
                )
            )
            print(f"[WORKER-7] Created collection: {collection_name}")
        
        # Create points
        points = []
        base_id = int(time.time() * 1000)  # Unique ID base
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=base_id + i,
                vector=embedding,
                payload={
                    'text': chunk['text'],
                    'source_url': chunk['metadata'].get('source_url', ''),
                    'source_type': chunk['metadata'].get('source_type', ''),
                    'source_trust': chunk['metadata'].get('source_trust', 0.5),
                    'fetched_at': chunk['metadata'].get('fetched_at', ''),
                    'chunk_index': chunk['metadata'].get('chunk_index', 0),
                    'total_chunks': chunk['metadata'].get('total_chunks', 1)
                }
            )
            points.append(point)
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=collection_name, points=batch)
        
        print(f"[WORKER-7] ✅ Stored {len(points)} chunks in Qdrant (server: {qdrant_host}:{qdrant_port})")
        return len(points)
        
    except Exception as e:
        logger.error(f"Qdrant storage error: {e}")
        return 0


# ============================================================================
# Main Ingest Function
# ============================================================================

async def ingest_web_resources(
    query: str,
    subject: Optional[str] = None,
    max_sources: int = 5,  # Increased for multi-source
    conversation_history: list = None,
    search_pdfs: bool = False,
    user_id: str = "",
    classroom_id: str = None  # Pass to worker6b for PDF storage
) -> IngestResult:
    """
    RATE-LIMIT SAFE Web Knowledge Ingest Pipeline.
    
    Now context-aware! Uses conversation_history for follow-up questions.
    
    Pipeline:
    1. Topic extraction (context-aware for follow-ups)
    2. Wikipedia REST API (primary, no limits)
    3. DuckDuckGo (1 call only, max 3 URLs)
    4. Parallel httpx fetch (all URLs concurrently)
    5. Extract with trafilatura
    6. Chunk (300-800 tokens)
    7. Embed with MiniLM
    8. Store in Qdrant
    
    Args:
        query: Search query
        subject: Optional subject filter
        max_sources: Maximum sources (default 3)
        conversation_history: Previous messages for follow-up context
        
    Returns:
        IngestResult with resources
    """
    start_time = time.time()
    
    # Initialize crawler logger
    from .debug_logger import CrawlerLogger
    crawler_log = CrawlerLogger(query)
    print("\n" + "="*62)
    print(f"[WEB] 🔍 Query: \"{query}\"")
    print("="*62)
    
    try:
        from .content_normalizer import normalize_content
        from .chunking_service import chunk_for_qdrant
        import re
        from collections import Counter
        
        # WORKER-1: Context-aware topic extraction
        topic = worker1_extract_topic(query, conversation_history)
        print(f"[WORKER-1] Topic: {topic}")
        
        # WORKER-3: Wikipedia Search API
        wiki_search = await worker3_wikipedia_search(topic)
        
        # WORKER-4: Wikipedia Content Fetch  
        wiki_data = None
        if wiki_search:
            wiki_data = await worker4_wikipedia_content(wiki_search['canonical_title'])
        
        # WORKER-2A: Serper API Search (primary - multi-source)
        serper_urls = []
        try:
            from .search_api import search_web
            serper_results = await search_web(topic, num_results=5)
            serper_urls = [r['url'] for r in serper_results if r.get('url')]
            if serper_urls:
                print(f"[WORKER-2A] ✅ Serper found {len(serper_urls)} sources")
                for u in serper_urls:
                    print(f"  - {u}")
        except Exception as e:
            print(f"[WORKER-2A] ⚠️ Serper search failed: {e}")
        
        # WORKER-2B: DuckDuckGo search (always run for diverse sources)
        ddg_urls = []
        print(f"[WORKER-2B] DuckDuckGo search for diverse sources...")
        search_results = worker2_duckduckgo_search(topic, max_results=5)
        ddg_urls = [r.get('href') for r in search_results if r.get('href')]
        
        # Collect URLs for parallel fetch
        urls_to_fetch = []
        
        # Add Wikipedia URL first (highest trust)
        if wiki_data:
            wiki_url = wiki_data.get('url')
            if wiki_url:
                urls_to_fetch.append(wiki_url)
                print(f"[PIPELINE] Wikipedia URL added: {wiki_url}")
        
        # Add Serper URLs (multi-source)
        urls_to_fetch.extend(serper_urls)
        
        # Add DuckDuckGo URLs as fallback
        urls_to_fetch.extend(ddg_urls)
        
        # Fallback: If still no URLs, try Wikipedia Search API
        if not urls_to_fetch:
            print("[WEB] No URLs from REST/DDG, trying Wikipedia Search API...")
            import requests
            wiki_search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "opensearch",
                "search": topic,
                "limit": 2,
                "namespace": 0,
                "format": "json"
            }
            try:
                wiki_response = requests.get(wiki_search_url, params=params, timeout=10)
                if wiki_response.status_code == 200:
                    data = wiki_response.json()
                    if len(data) >= 4 and data[3]:
                        urls_to_fetch.extend(data[3][:2])
                        print(f"[WEB] Wikipedia Search found: {data[3][:2]}")
            except:
                pass
        
        # Remove duplicates
        urls_to_fetch = list(dict.fromkeys(urls_to_fetch))[:max_sources]
        
        if not urls_to_fetch:
            crawler_log.log_error("No sources found")
            return IngestResult(
                success=False,
                query=query,
                resources=[],
                total_chunks_stored=0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                error="No sources found"
            )
        
        # WORKER-5: Parallel fetch all URLs
        fetched_pages = await worker5_parallel_crawl(urls_to_fetch)
        
        resources = []
        all_chunks = []
        total_tokens = 0
        
        # Debug: Check what we got from Worker-5
        print(f"[DEBUG] Processing {len(fetched_pages)} fetched pages")
        print(f"[DEBUG] fetched_pages type: {type(fetched_pages)}")
        if fetched_pages:
            print(f"[DEBUG] First page keys: {list(fetched_pages[0].keys())}")
            print(f"[DEBUG] First page URL: {fetched_pages[0].get('url', 'NO URL KEY')}")
        
        # WORKER-6 + WORKER-7: Process each fetched page
        print(f"[DEBUG] Starting for loop...")
        for page in fetched_pages:
            url = page['url']
            html_content = page['content']
            
            print(f"\n{'='*70}")
            print(f"[WORKER-6] 🌐 PROCESSING URL: {url}")
            print(f"{'='*70}")
            
            try:
                # Extract clean text with trafilatura
                import trafilatura
                clean_text = trafilatura.extract(html_content)
                
                if not clean_text or len(clean_text) < 100:
                    print(f"[WORKER-6] ⚠ Insufficient content from {url[:60]}")
                    continue
                
                print(f"[WORKER-6] 📄 Extracted {len(clean_text)} characters")
                
                # Show content preview - FIRST 500 chars
                print(f"\n[CONTENT PREVIEW - First 500 chars]")
                print("-" * 50)
                print(clean_text[:500])
                print("-" * 50)
                
                # Show LAST 200 chars
                print(f"\n[CONTENT PREVIEW - Last 200 chars]")
                print("-" * 50)
                print(clean_text[-200:])
                print("-" * 50 + "\n")
                
                # LIMIT content size for performance (15000 chars max)
                MAX_CONTENT = 15000
                if len(clean_text) > MAX_CONTENT:
                    clean_text = clean_text[:MAX_CONTENT]
                    last_period = clean_text.rfind('.')
                    if last_period > MAX_CONTENT * 0.8:
                        clean_text = clean_text[:last_period + 1]
                    print(f"[WORKER-6] ✂ Truncated to {len(clean_text)} chars")
                
                # Count tokens
                from .debug_logger import count_tokens
                token_count = count_tokens(clean_text)
                total_tokens += token_count
                
                print(f"[WORKER-6] 📊 Token count: {token_count}")
                
                # Normalize
                normalized = normalize_content(clean_text, url.split('/')[-1], 'text')
                
                # Chunk
                chunks = chunk_for_qdrant(
                    text=normalized.text,
                    source_url=url,
                    source_type='webpage',
                    source_trust=calculate_trust_score(url)
                )
                
                print(f"[WORKER-6] ✅ Created {len(chunks)} chunks for Qdrant")
                
                # Show chunk previews
                for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                    print(f"  [Chunk {i+1}] {chunk['text'][:150]}...")
                
                if len(chunks) > 3:
                    print(f"  ... and {len(chunks) - 3} more chunks")
                
                # Generate resource ID
                resource_id = hashlib.md5(url.encode()).hexdigest()[:12]
                
                resources.append(IngestedResource(
                    id=f"web_{resource_id}",
                    url=url,
                    title=url.split('/')[-1].replace('_', ' '),
                    source_name=extract_source_name(url),
                    source_type='webpage',
                    trust_score=calculate_trust_score(url),
                    clean_content=normalized.text,
                    summary=normalized.summary,
                    word_count=normalized.word_count,
                    chunk_count=len(chunks),
                    fetched_at=datetime.now().isoformat(),
                    stored_in_qdrant=True
                ))
                
                all_chunks.extend(chunks)
                
                print(f"[WORKER-6] ✅ Source: {extract_source_name(url)} | Trust: {calculate_trust_score(url):.2f}")
                
            except Exception as e:
                print(f"[WORKER-6] ❌ Processing error for {url[:60]}: {e}")
                import traceback
                traceback.print_exc()
        
        # Worker-6 Complete
        print(f"\n{'='*70}")
        print(f"[WORKER-6] ✅ COMPLETE: {len(resources)} sources, {total_tokens} total tokens")
        print(f"{'='*70}\n")
        
        # WORKER-6B: Optional PDF Search & Indexing
        if search_pdfs:
            print(f"[WORKER-6B] PDF search enabled, searching for educational PDFs...")
            try:
                pdf_documents = await worker6b_pdf_search(
                    topic=topic,
                    user_id=user_id,
                    max_pdfs=3,
                    classroom_id=classroom_id  # Store PDFs in classroom materials
                )
                
                # Process PDF content and add to chunks
                for pdf_doc in pdf_documents:
                    text = pdf_doc.get('extracted_text', '')
                    if text and len(text) > 100:
                        # Create chunks from PDF text
                        pdf_chunks = chunk_for_qdrant(
                            text=text,
                            source_url=pdf_doc.get('source_url', ''),
                            source_type='web_pdf',
                            source_trust=0.85  # High trust for PDFs
                        )
                        
                        all_chunks.extend(pdf_chunks)
                        
                        # Add to resources
                        resources.append(IngestedResource(
                            id=f"pdf_{hashlib.md5(pdf_doc['source_url'].encode()).hexdigest()[:12]}",
                            url=pdf_doc.get('source_url', ''),
                            title=pdf_doc.get('file_name', 'PDF Document'),
                            source_name='Web PDF',
                            source_type='web_pdf',
                            trust_score=0.85,
                            clean_content=text[:1000],
                            summary=f"PDF with {pdf_doc.get('word_count', 0)} words",
                            word_count=pdf_doc.get('word_count', 0),
                            chunk_count=len(pdf_chunks),
                            fetched_at=datetime.now().isoformat(),
                            stored_in_qdrant=True
                        ))
                        
                        print(f"[WORKER-6B] Added {len(pdf_chunks)} chunks from {pdf_doc.get('file_name', 'PDF')}")
                        
            except Exception as e:
                print(f"[WORKER-6B] ⚠ PDF search failed (non-fatal): {e}")
        
        # WORKER-7: Chunk + Embed
        total_stored = 0
        if all_chunks:
            print(f"[WORKER-7] 📦 Storing {len(all_chunks)} chunks in Qdrant...")
            texts = [chunk['text'] for chunk in all_chunks]
            embeddings = generate_embeddings(texts)
            
            if embeddings:
                total_stored = store_in_qdrant(all_chunks, embeddings)
                print(f"[WORKER-7] ✅ Successfully stored {total_stored} chunks in Qdrant")
        
        # Pipeline complete
        elapsed = time.time() - start_time
        print("="*62)
        print(f"[WEB] ✅ Pipeline complete: {len(resources)} sources, {total_stored} chunks, {elapsed:.1f}s")
        print("="*62)
        
        return IngestResult(
            success=True,
            query=query,
            resources=resources,
            total_chunks_stored=total_stored,
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
        
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        return IngestResult(
            success=False,
            query=query,
            resources=[],
            total_chunks_stored=0,
            processing_time_ms=int((time.time() - start_time) * 1000),
            error=str(e)
        )


# ============================================================================
# Sync Wrapper for Non-Async Contexts
# ============================================================================

def ingest_web_resources_sync(
    query: str,
    subject: Optional[str] = None,
    max_sources: int = 3,
    search_pdfs: bool = False,
    user_id: str = "",
    classroom_id: str = None
) -> IngestResult:
    """Synchronous wrapper for ingest_web_resources."""
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        ingest_web_resources(query, subject, max_sources, search_pdfs=search_pdfs, user_id=user_id, classroom_id=classroom_id)
    )
