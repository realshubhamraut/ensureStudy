"""
PDF Downloader Service - Download PDFs from Web Search Results

This service downloads educational PDFs from web searches and stores them
as classroom materials for RAG indexing.

Flow:
1. Search for PDFs using Serper API (filetype:pdf)
2. Download PDF from URL
3. Save to local storage (uploads/web/)
4. Create ClassroomMaterial record via Core API
5. Trigger indexing in AI service
"""
import os
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse, unquote
import httpx

logger = logging.getLogger(__name__)

# Directory for web-downloaded PDFs
WEB_UPLOADS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    'data', 'web_resources'
)
os.makedirs(WEB_UPLOADS_DIR, exist_ok=True)


@dataclass
class DownloadResult:
    """Result of PDF download operation"""
    success: bool
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    file_size: int = 0
    source_url: str = ""
    error: Optional[str] = None


class WebPDFDownloader:
    """
    Download PDFs from web search results and store them locally.
    
    Features:
    - Async download with timeout
    - Size limit enforcement (max 50MB)
    - Filename extraction from URL/headers
    - Duplicate detection by URL
    """
    
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    DOWNLOAD_TIMEOUT = 60  # seconds
    
    def __init__(self):
        self.downloaded_urls = set()  # Track to avoid duplicates
    
    def _extract_filename(self, url: str, headers: dict) -> str:
        """Extract filename from Content-Disposition header or URL"""
        # Try Content-Disposition header first
        content_disp = headers.get('content-disposition', '')
        if 'filename=' in content_disp:
            import re
            match = re.search(r'filename[^;=\n]*=([\"\']?)([^;\"\'\n]*)\1', content_disp)
            if match:
                return unquote(match.group(2))
        
        # Fall back to URL path
        path = urlparse(url).path
        filename = os.path.basename(path)
        
        # If no filename or no extension, generate one
        if not filename or '.' not in filename:
            filename = f"document_{uuid.uuid4().hex[:8]}.pdf"
        
        return filename
    
    async def download_pdf(
        self, 
        url: str, 
        user_id: str,
        topic: str = ""
    ) -> DownloadResult:
        """
        Download a PDF from URL and save locally.
        
        Args:
            url: Source URL of the PDF
            user_id: User who triggered the download
            topic: Search topic (for naming)
            
        Returns:
            DownloadResult with file path and metadata
        """
        if url in self.downloaded_urls:
            return DownloadResult(
                success=False,
                source_url=url,
                error="Already downloaded in this session"
            )
        
        try:
            print(f"[PDF-DL] Downloading: {url[:80]}...")
            
            async with httpx.AsyncClient(
                timeout=self.DOWNLOAD_TIMEOUT,
                follow_redirects=True
            ) as client:
                # First, check headers with HEAD request
                try:
                    head_resp = await client.head(url)
                    content_length = int(head_resp.headers.get('content-length', 0))
                    content_type = head_resp.headers.get('content-type', '')
                    
                    if content_length > self.MAX_FILE_SIZE:
                        return DownloadResult(
                            success=False,
                            source_url=url,
                            error=f"File too large: {content_length / 1024 / 1024:.1f}MB"
                        )
                    
                    if 'pdf' not in content_type.lower() and not url.endswith('.pdf'):
                        return DownloadResult(
                            success=False,
                            source_url=url,
                            error=f"Not a PDF: {content_type}"
                        )
                except:
                    pass  # Continue with GET if HEAD fails
                
                # Download the file
                response = await client.get(url)
                
                if response.status_code != 200:
                    return DownloadResult(
                        success=False,
                        source_url=url,
                        error=f"HTTP {response.status_code}"
                    )
                
                content = response.content
                file_size = len(content)
                
                if file_size > self.MAX_FILE_SIZE:
                    return DownloadResult(
                        success=False,
                        source_url=url,
                        error=f"File too large: {file_size / 1024 / 1024:.1f}MB"
                    )
                
                # Extract filename
                original_name = self._extract_filename(url, dict(response.headers))
                
                # Create unique filename
                unique_id = uuid.uuid4().hex[:8]
                safe_topic = "".join(c for c in topic if c.isalnum() or c in ' -_')[:30]
                file_name = f"{safe_topic}_{unique_id}_{original_name}"
                file_path = os.path.join(WEB_UPLOADS_DIR, file_name)
                
                # Save file
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                print(f"[PDF-DL] ✅ Saved: {file_name} ({file_size/1024:.1f}KB)")
                
                self.downloaded_urls.add(url)
                
                return DownloadResult(
                    success=True,
                    file_path=file_path,
                    file_name=file_name,
                    file_size=file_size,
                    source_url=url
                )
                
        except asyncio.TimeoutError:
            return DownloadResult(
                success=False,
                source_url=url,
                error="Download timeout"
            )
        except Exception as e:
            logger.error(f"[PDF-DL] Download error for {url}: {e}")
            return DownloadResult(
                success=False,
                source_url=url,
                error=str(e)
            )
    
    async def download_multiple(
        self,
        urls: List[str],
        user_id: str,
        topic: str = "",
        max_downloads: int = 3
    ) -> List[DownloadResult]:
        """
        Download multiple PDFs concurrently.
        
        Args:
            urls: List of PDF URLs
            user_id: User who triggered the download
            topic: Search topic
            max_downloads: Maximum PDFs to download
            
        Returns:
            List of DownloadResult objects
        """
        # Limit to max_downloads
        urls = urls[:max_downloads]
        
        if not urls:
            return []
        
        print(f"[PDF-DL] Downloading {len(urls)} PDFs for topic: '{topic}'")
        
        # Download concurrently
        tasks = [
            self.download_pdf(url, user_id, topic)
            for url in urls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful downloads
        successful = [r for r in results if isinstance(r, DownloadResult) and r.success]
        failed = len(urls) - len(successful)
        
        print(f"[PDF-DL] Downloaded: {len(successful)}/{len(urls)} (failed: {failed})")
        
        return [r for r in results if isinstance(r, DownloadResult)]


# ============================================================================
# Integration with Search API
# ============================================================================

async def search_and_download_pdfs(
    query: str,
    user_id: str,
    num_results: int = 3
) -> List[DownloadResult]:
    """
    Search for PDFs and download them.
    
    Args:
        query: Search topic (e.g., "inferential statistics")
        user_id: User ID for tracking
        num_results: Number of PDFs to download
        
    Returns:
        List of successful download results
    """
    from .search_api import MultiSourceSearcher
    
    # Search for PDFs
    print(f"[PDF-SEARCH] Searching for PDFs: '{query}'")
    
    searcher = MultiSourceSearcher()
    
    # Modify query to search for PDFs
    pdf_query = f"{query} filetype:pdf"
    results = await searcher.search(pdf_query, num_results * 2)  # Get extra for filtering
    
    # Filter to only PDF URLs
    pdf_urls = []
    for r in results:
        url = r.url.lower()
        if url.endswith('.pdf') or 'pdf' in url:
            pdf_urls.append(r.url)
            if len(pdf_urls) >= num_results:
                break
    
    if not pdf_urls:
        print(f"[PDF-SEARCH] No PDFs found for: '{query}'")
        return []
    
    print(f"[PDF-SEARCH] Found {len(pdf_urls)} PDF URLs")
    
    # Download PDFs
    downloader = WebPDFDownloader()
    results = await downloader.download_multiple(pdf_urls, user_id, query, num_results)
    
    return [r for r in results if r.success]


# Quick test
if __name__ == "__main__":
    async def test():
        results = await search_and_download_pdfs(
            "inferential statistics",
            user_id="test-user-123",
            num_results=2
        )
        print(f"\n=== Downloaded {len(results)} PDFs ===")
        for r in results:
            print(f"  - {r.file_name}: {r.file_size/1024:.1f}KB")
            print(f"    Source: {r.source_url[:60]}...")
    
    asyncio.run(test())
