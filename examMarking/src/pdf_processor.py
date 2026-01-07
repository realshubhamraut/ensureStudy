"""
PDF Processing Module
Extracts text from PDF documents with error handling
"""

import pdfplumber
from pathlib import Path
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Extract text content from PDF files"""
    
    def __init__(self):
        self.last_metadata = None
    
    def extract_text(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing:
                - text: Extracted text content
                - pages: Number of pages
                - success: Boolean indicating success
                - error: Error message if failed
        """
        result = {
            "text": "",
            "pages": 0,
            "success": False,
            "error": None
        }
        
        try:
            pdf_path = Path(pdf_path)
            
            if not pdf_path.exists():
                result["error"] = f"File not found: {pdf_path}"
                logger.error(result["error"])
                return result
            
            if not pdf_path.suffix.lower() == '.pdf':
                result["error"] = f"Not a PDF file: {pdf_path}"
                logger.error(result["error"])
                return result
            
            # Extract text using pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                result["pages"] = len(pdf.pages)
                text_parts = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                        logger.debug(f"Extracted {len(page_text)} characters from page {page_num}")
                
                result["text"] = "\n\n".join(text_parts)
                result["success"] = True
                
                # Store metadata
                self.last_metadata = {
                    "filename": pdf_path.name,
                    "pages": result["pages"],
                    "characters": len(result["text"]),
                    "words": len(result["text"].split())
                }
                
                logger.info(f"Successfully extracted text from {pdf_path.name}: "
                          f"{self.last_metadata['words']} words from {result['pages']} pages")
            
        except Exception as e:
            result["error"] = f"Error processing PDF: {str(e)}"
            logger.error(result["error"])
        
        return result
    
    def get_metadata(self) -> Optional[Dict]:
        """Get metadata from the last processed PDF"""
        return self.last_metadata
    
    def validate_pdf(self, pdf_path: str) -> bool:
        """
        Quick validation check for PDF file
        
        Args:
            pdf_path: Path to check
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists() or not pdf_path.suffix.lower() == '.pdf':
                return False
            
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages) > 0
        except:
            return False


# Utility function for easy import
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Convenience function to extract text from PDF
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text or empty string if failed
    """
    processor = PDFProcessor()
    result = processor.extract_text(pdf_path)
    return result["text"] if result["success"] else ""
