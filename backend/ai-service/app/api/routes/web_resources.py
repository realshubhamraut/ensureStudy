"""
Web Resources API Routes
Handles searching for and downloading educational PDFs from the web.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import httpx

router = APIRouter(prefix="/api/web-resources", tags=["Web Resources"])


class PDFSearchRequest(BaseModel):
    """Request to search for PDFs"""
    query: str
    user_id: str
    num_results: int = 3


class PDFSearchResult(BaseModel):
    """Single PDF search result"""
    url: str
    title: str
    snippet: str
    domain: str
    trust_score: float


class DownloadedPDF(BaseModel):
    """Downloaded PDF info"""
    file_name: str
    file_path: str
    file_size: int
    source_url: str


class WebResourcesResponse(BaseModel):
    """Response from PDF search and download"""
    success: bool
    query: str
    pdfs_found: int
    pdfs_downloaded: int
    downloaded: List[DownloadedPDF]
    errors: List[str]


@router.post("/search-pdfs", response_model=List[PDFSearchResult])
async def search_pdfs_endpoint(request: PDFSearchRequest):
    """
    Search for educational PDFs using Serper API.
    
    Returns list of PDF URLs without downloading.
    """
    from app.services.search_api import search_pdfs
    
    results = await search_pdfs(request.query, request.num_results)
    
    return [
        PDFSearchResult(
            url=r["url"],
            title=r["title"],
            snippet=r["snippet"],
            domain=r["domain"],
            trust_score=r["trust_score"]
        )
        for r in results
    ]


@router.post("/download-pdfs", response_model=WebResourcesResponse)
async def download_pdfs_endpoint(request: PDFSearchRequest):
    """
    Search for and download educational PDFs.
    
    1. Searches for PDFs using Serper API
    2. Downloads top N PDFs to local storage
    3. Creates database records (via Core API)
    4. Returns download results
    """
    from app.services.search_api import search_pdfs
    from app.services.pdf_downloader import WebPDFDownloader
    
    # Search for PDFs
    search_results = await search_pdfs(request.query, request.num_results * 2)
    
    if not search_results:
        return WebResourcesResponse(
            success=False,
            query=request.query,
            pdfs_found=0,
            pdfs_downloaded=0,
            downloaded=[],
            errors=["No PDFs found for this query"]
        )
    
    # Download PDFs
    downloader = WebPDFDownloader()
    pdf_urls = [r["url"] for r in search_results]
    
    download_results = await downloader.download_multiple(
        urls=pdf_urls,
        user_id=request.user_id,
        topic=request.query,
        max_downloads=request.num_results
    )
    
    # Collect successful downloads
    downloaded = []
    errors = []
    
    for result in download_results:
        if result.success:
            downloaded.append(DownloadedPDF(
                file_name=result.file_name,
                file_path=result.file_path,
                file_size=result.file_size,
                source_url=result.source_url
            ))
            
            # Register with Core API (create material record)
            try:
                await register_web_material(
                    file_path=result.file_path,
                    file_name=result.file_name,
                    file_size=result.file_size,
                    source_url=result.source_url,
                    user_id=request.user_id,
                    topic=request.query
                )
            except Exception as e:
                print(f"[WEB-RES] Warning: Failed to register material: {e}")
        else:
            errors.append(f"{result.source_url[:50]}...: {result.error}")
    
    return WebResourcesResponse(
        success=len(downloaded) > 0,
        query=request.query,
        pdfs_found=len(search_results),
        pdfs_downloaded=len(downloaded),
        downloaded=downloaded,
        errors=errors
    )


async def register_web_material(
    file_path: str,
    file_name: str,
    file_size: int,
    source_url: str,
    user_id: str,
    topic: str
) -> bool:
    """
    Register downloaded PDF as a web material in Core API.
    
    Creates a ClassroomMaterial record with source='web'.
    """
    core_api_url = os.getenv("CORE_API_URL", "http://localhost:8000")
    
    # Create the material via Core API
    payload = {
        "name": file_name,
        "file_path": file_path,
        "file_size": file_size,
        "source_url": source_url,
        "user_id": user_id,
        "topic": topic,
        "source": "web"
    }
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{core_api_url}/api/web-resources/register",
                json=payload,
                headers={"X-Service-Key": "internal-ai-service"}
            )
            
            if response.status_code in (200, 201):
                print(f"[WEB-RES] ✅ Registered: {file_name}")
                return True
            else:
                print(f"[WEB-RES] ❌ Registration failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"[WEB-RES] ❌ Registration error: {e}")
        return False


@router.get("/list/{user_id}")
async def list_user_web_resources(user_id: str):
    """
    List all web resources downloaded by a user.
    """
    core_api_url = os.getenv("CORE_API_URL", "http://localhost:8000")
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{core_api_url}/api/web-resources/user/{user_id}",
                headers={"X-Service-Key": "internal-ai-service"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"resources": [], "error": f"Core API returned {response.status_code}"}
                
    except Exception as e:
        return {"resources": [], "error": str(e)}
