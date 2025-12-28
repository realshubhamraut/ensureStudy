"""
PDF Processing Service
Extracts text from PDFs for indexing into the RAG system.

Uses PyMuPDF (fitz) for text-based PDFs with OCR fallback for scanned documents.
"""
import os
import logging
import tempfile
import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Content extracted from a single PDF page"""
    page_number: int
    text: str
    has_images: bool
    word_count: int
    is_scanned: bool  # True if page appears to be scanned (no text layer)


@dataclass
class PDFExtractionResult:
    """Result of PDF text extraction"""
    success: bool
    pages: List[PageContent]
    total_pages: int
    total_words: int
    extraction_method: str  # 'text' or 'ocr'
    error: Optional[str] = None


class PDFProcessor:
    """
    Extract text from PDF documents for RAG indexing.
    
    Supports:
    - Text-based PDFs (fast extraction)
    - Scanned PDFs (OCR fallback)
    - Image-heavy documents
    """
    
    def __init__(self, ocr_service=None):
        """
        Initialize PDF processor.
        
        Args:
            ocr_service: Optional OCRService instance for scanned PDFs
        """
        self.ocr_service = ocr_service
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required libraries are installed."""
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
            logger.info("[PDF] PyMuPDF loaded successfully")
        except ImportError:
            logger.error("[PDF] PyMuPDF not installed. Run: pip install pymupdf")
            self.fitz = None
    
    def extract_from_url(self, file_url: str) -> PDFExtractionResult:
        """
        Download and extract text from a PDF URL.
        
        Args:
            file_url: URL of the PDF file
            
        Returns:
            PDFExtractionResult with extracted pages
        """
        try:
            logger.info(f"[PDF] Downloading from: {file_url[:80]}...")
            
            # Download to temp file
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            
            try:
                result = self.extract_from_file(tmp_path)
                return result
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
                
        except requests.RequestException as e:
            logger.error(f"[PDF] Download failed: {e}")
            return PDFExtractionResult(
                success=False,
                pages=[],
                total_pages=0,
                total_words=0,
                extraction_method='none',
                error=f"Failed to download PDF: {str(e)}"
            )
    
    def extract_from_file(self, file_path: str) -> PDFExtractionResult:
        """
        Extract text from a local PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            PDFExtractionResult with extracted pages
        """
        if not self.fitz:
            return PDFExtractionResult(
                success=False,
                pages=[],
                total_pages=0,
                total_words=0,
                extraction_method='none',
                error="PyMuPDF not installed"
            )
        
        try:
            logger.info(f"[PDF] Processing: {file_path}")
            doc = self.fitz.open(file_path)
            
            pages = []
            total_words = 0
            scanned_pages = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text("text").strip()
                word_count = len(text.split()) if text else 0
                
                # Check for images
                image_list = page.get_images()
                has_images = len(image_list) > 0
                
                # Detect if page is scanned (has images but no text)
                is_scanned = has_images and word_count < 10
                
                if is_scanned:
                    scanned_pages += 1
                    # Try OCR if available and page appears scanned
                    if self.ocr_service:
                        ocr_text = self._ocr_page(page)
                        if ocr_text:
                            text = ocr_text
                            word_count = len(text.split())
                
                page_content = PageContent(
                    page_number=page_num + 1,
                    text=text,
                    has_images=has_images,
                    word_count=word_count,
                    is_scanned=is_scanned
                )
                pages.append(page_content)
                total_words += word_count
            
            doc.close()
            
            # Determine extraction method
            if scanned_pages > len(pages) / 2:
                method = 'ocr' if self.ocr_service else 'text_limited'
            else:
                method = 'text'
            
            logger.info(f"[PDF] Extracted {total_words} words from {len(pages)} pages")
            
            return PDFExtractionResult(
                success=True,
                pages=pages,
                total_pages=len(pages),
                total_words=total_words,
                extraction_method=method
            )
            
        except Exception as e:
            logger.error(f"[PDF] Extraction failed: {e}")
            return PDFExtractionResult(
                success=False,
                pages=[],
                total_pages=0,
                total_words=0,
                extraction_method='none',
                error=str(e)
            )
    
    def _ocr_page(self, page) -> Optional[str]:
        """
        Run OCR on a PDF page.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Extracted text or None
        """
        if not self.ocr_service:
            return None
        
        try:
            import numpy as np
            
            # Render page as image (higher DPI for better OCR)
            mat = self.fitz.Matrix(2.0, 2.0)  # 2x zoom = ~144 DPI
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to numpy array
            img_data = pix.samples
            img = np.frombuffer(img_data, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            
            # Convert RGBA to BGR if needed
            if pix.n == 4:
                img = img[:, :, :3]
            
            # Run OCR
            result = self.ocr_service.extract_text(img)
            return result.text if result.text else None
            
        except Exception as e:
            logger.warning(f"[PDF] OCR failed for page: {e}")
            return None
    
    def get_full_text(self, result: PDFExtractionResult) -> str:
        """
        Combine all pages into a single text string.
        
        Args:
            result: PDFExtractionResult from extraction
            
        Returns:
            Combined text with page markers
        """
        if not result.success:
            return ""
        
        parts = []
        for page in result.pages:
            if page.text:
                parts.append(f"[Page {page.page_number}]\n{page.text}")
        
        return "\n\n".join(parts)
    
    def get_pages_text(self, result: PDFExtractionResult) -> List[Dict[str, Any]]:
        """
        Get text organized by page for chunking.
        
        Args:
            result: PDFExtractionResult from extraction
            
        Returns:
            List of dicts with page_number and text
        """
        return [
            {
                "page_number": page.page_number,
                "text": page.text,
                "word_count": page.word_count
            }
            for page in result.pages
            if page.text  # Only include pages with text
        ]


# Convenience function for quick extraction
def extract_text_from_pdf(file_path: str) -> str:
    """
    Quick function to extract all text from a PDF.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Combined text from all pages
    """
    processor = PDFProcessor()
    result = processor.extract_from_file(file_path)
    return processor.get_full_text(result)


def extract_text_from_url(url: str) -> str:
    """
    Quick function to extract text from a PDF URL.
    
    Args:
        url: URL of PDF file
        
    Returns:
        Combined text from all pages
    """
    processor = PDFProcessor()
    result = processor.extract_from_url(url)
    return processor.get_full_text(result)
