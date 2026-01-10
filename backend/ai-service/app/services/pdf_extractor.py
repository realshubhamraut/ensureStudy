"""
PDF Extractor Service
Extracts text from PDFs and performs OCR on handwritten/image-based pages

Uses:
- PyMuPDF (fitz) for text extraction - FREE, LOCAL
- Tesseract for OCR - FREE, LOCAL (requires: brew install tesseract)
"""
import os
import re
import tempfile
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import asyncio
import aiohttp

# PDF processing
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("[PDF] PyMuPDF not installed. Install with: pip install pymupdf")

# OCR
try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    print("[PDF] pytesseract not installed. Install with: pip install pytesseract Pillow")


@dataclass
class Question:
    """Represents a parsed question from assignment PDF"""
    number: int
    text: str
    points: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'number': self.number,
            'text': self.text,
            'points': self.points
        }


@dataclass
class ExtractedContent:
    """Represents extracted content from a PDF"""
    full_text: str
    questions: List[Question]
    page_count: int
    has_images: bool
    ocr_used: bool


class PDFExtractor:
    """
    Extracts text and questions from PDFs.
    Supports both typed text and handwritten content via OCR.
    """
    
    def __init__(self):
        self.ocr_available = pytesseract is not None
        self.pdf_available = fitz is not None
    
    async def download_pdf(self, url: str) -> str:
        """Download PDF from URL to temp file"""
        temp_path = tempfile.mktemp(suffix='.pdf')
        
        # Handle local file paths
        if url.startswith('/') or url.startswith('file://'):
            local_path = url.replace('file://', '')
            if os.path.exists(local_path):
                return local_path
        
        # Handle relative URLs (from local storage)
        if url.startswith('/api/') or url.startswith('/uploads/'):
            # Construct local path from upload URL
            # Adjust based on your storage configuration
            base_dir = os.getenv('UPLOAD_DIR', '/Users/proxim/projects/ensureStudy/backend/core-service/uploads')
            local_path = os.path.join(base_dir, url.split('/uploads/')[-1] if '/uploads/' in url else url.split('/')[-1])
            if os.path.exists(local_path):
                return local_path
        
        # Download from remote URL
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        with open(temp_path, 'wb') as f:
                            f.write(content)
                        return temp_path
                    else:
                        raise RuntimeError(f"Failed to download PDF: HTTP {response.status}")
        except Exception as e:
            raise RuntimeError(f"Failed to download PDF from {url}: {e}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, bool, int]:
        """
        Extract text from PDF using PyMuPDF.
        Returns: (text, has_images, page_count)
        """
        if not self.pdf_available:
            raise RuntimeError("PyMuPDF not installed")
        
        doc = fitz.open(pdf_path)
        full_text = ""
        has_images = False
        page_count = len(doc)  # Get count BEFORE closing
        
        for page_num, page in enumerate(doc):
            # Extract text
            page_text = page.get_text()
            
            # Check if page has images (might need OCR)
            image_list = page.get_images()
            if image_list:
                has_images = True
            
            # If no text but has images, likely scanned/handwritten
            if not page_text.strip() and image_list:
                # Mark for OCR
                full_text += f"\n[PAGE {page_num + 1} - REQUIRES OCR]\n"
            else:
                full_text += page_text
        
        doc.close()
        return full_text.strip(), has_images, page_count
    
    def ocr_pdf(self, pdf_path: str) -> str:
        """
        Perform OCR on PDF pages that are images.
        Used for handwritten submissions.
        """
        if not self.pdf_available or not self.ocr_available:
            return ""
        
        doc = fitz.open(pdf_path)
        ocr_text = ""
        
        for page_num, page in enumerate(doc):
            # Check if page has minimal text (likely image-based)
            page_text = page.get_text().strip()
            
            if len(page_text) < 50:  # Likely needs OCR
                # Render page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                
                # Save to temp file
                temp_img = tempfile.mktemp(suffix='.png')
                pix.save(temp_img)
                
                # Perform OCR
                try:
                    img = Image.open(temp_img)
                    text = pytesseract.image_to_string(img, lang='eng')
                    ocr_text += f"\n--- Page {page_num + 1} (OCR) ---\n{text}\n"
                except Exception as e:
                    print(f"[OCR] Failed on page {page_num + 1}: {e}")
                finally:
                    if os.path.exists(temp_img):
                        os.remove(temp_img)
            else:
                ocr_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        doc.close()
        return ocr_text.strip()
    
    def extract_questions(self, text: str) -> List[Question]:
        """
        Parse text to extract numbered questions.
        Handles formats like:
        - 1. What is...
        - Q1: What is...
        - Question 1: What is...
        - 1) What is...
        - (1) What is...
        """
        questions = []
        
        # Patterns for question detection
        patterns = [
            r'(?:Question\s*)?(\d+)\s*[.):]\s*(.+?)(?=(?:Question\s*)?\d+\s*[.):]|$)',  # Q1. or 1. or 1)
            r'Q(\d+)\s*[.:]\s*(.+?)(?=Q\d+|$)',  # Q1:
            r'\((\d+)\)\s*(.+?)(?=\(\d+\)|$)',  # (1)
        ]
        
        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                for num, q_text in matches:
                    q_text = q_text.strip()
                    if len(q_text) > 10:  # Minimum question length
                        # Extract points if mentioned (e.g., "[5 points]", "(10 marks)")
                        points_match = re.search(r'\[?\(?\s*(\d+)\s*(?:points?|marks?)\s*\]?\)?', q_text, re.IGNORECASE)
                        points = int(points_match.group(1)) if points_match else None
                        
                        questions.append(Question(
                            number=int(num),
                            text=q_text[:500],  # Limit length
                            points=points
                        ))
                break  # Use first matching pattern
        
        # Sort by question number
        questions.sort(key=lambda q: q.number)
        
        return questions
    
    async def extract_from_url(self, pdf_url: str, use_ocr: bool = True) -> ExtractedContent:
        """
        Main entry point: Download PDF, extract text, parse questions.
        
        Args:
            pdf_url: URL or path to PDF
            use_ocr: Whether to use OCR for image-based pages
            
        Returns:
            ExtractedContent with full text and parsed questions
        """
        # Download PDF
        pdf_path = await self.download_pdf(pdf_url)
        temp_file = not pdf_path == pdf_url  # Track if we created a temp file
        
        try:
            # Extract text
            text, has_images, page_count = self.extract_text_from_pdf(pdf_path)
            ocr_used = False
            
            # Use OCR if needed and available
            if use_ocr and self.ocr_available:
                if has_images or len(text) < 100:  # Likely handwritten
                    ocr_text = self.ocr_pdf(pdf_path)
                    if ocr_text:
                        text = ocr_text
                        ocr_used = True
            
            # Extract questions
            questions = self.extract_questions(text)
            
            print(f"[PDF] Extracted {len(questions)} questions, {len(text)} chars, OCR={ocr_used}")
            
            return ExtractedContent(
                full_text=text,
                questions=questions,
                page_count=page_count,
                has_images=has_images,
                ocr_used=ocr_used
            )
            
        finally:
            # Cleanup temp file
            if temp_file and os.path.exists(pdf_path) and pdf_path.startswith(tempfile.gettempdir()):
                os.remove(pdf_path)


# Singleton instance
pdf_extractor = PDFExtractor()
