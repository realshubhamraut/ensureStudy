"""
Document Processor Service
Handles page extraction, OCR, chunking, embedding, and indexing.
"""
import io
import re
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class OCRBlock:
    """OCR extracted text block with bounding box."""
    text: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    block_type: str = "text"  # text, formula, table


@dataclass
class PageResult:
    """OCR result for a single page."""
    page_number: int
    blocks: List[OCRBlock]
    full_text: str
    avg_confidence: float
    method: str  # nanonets, tesseract, paddleocr
    success: bool = True
    error: Optional[str] = None


class DocumentProcessor:
    """
    Processes uploaded documents through:
    1. Page extraction (PDF → images)
    2. OCR (Nanonets primary, Tesseract fallback)
    3. PII redaction
    4. Chunking with block provenance
    5. Embedding generation
    6. Qdrant indexing
    """
    
    def __init__(self):
        self.nanonets_ocr = None
        self.tesseract_available = False
        self.embedder = None
        self.confidence_threshold = 0.7
        
        self._init_ocr()
        self._init_embedder()
    
    def _init_ocr(self):
        """Initialize OCR engines."""
        # Try Tesseract (fallback)
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("[OCR] Tesseract available")
        except Exception as e:
            logger.warning(f"[OCR] Tesseract not available: {e}")
        
        # Nanonets OCR will be loaded on-demand
    
    def _init_embedder(self):
        """Initialize sentence transformer for embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("[EMBED] Loaded all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"[EMBED] Failed to load embedder: {e}")
    
    def extract_pages(
        self,
        file_content: bytes,
        mime_type: str
    ) -> Dict[str, Any]:
        """
        Extract pages from PDF or handle image files.
        
        Returns:
            {
                'success': bool,
                'page_count': int,
                'page_images': List[PIL.Image],
                'error': str (if failed)
            }
        """
        try:
            if 'pdf' in mime_type.lower():
                return self._extract_pdf_pages(file_content)
            elif any(t in mime_type.lower() for t in ['image', 'png', 'jpeg', 'jpg']):
                return self._handle_image(file_content)
            else:
                return {
                    'success': False,
                    'page_count': 0,
                    'page_images': [],
                    'error': f'Unsupported mime type: {mime_type}'
                }
        except Exception as e:
            logger.error(f"[EXTRACT] Error: {e}")
            return {
                'success': False,
                'page_count': 0,
                'page_images': [],
                'error': str(e)
            }
    
    def _extract_pdf_pages(self, pdf_content: bytes) -> Dict[str, Any]:
        """Extract pages from PDF as images."""
        try:
            # Try pdf2image first
            try:
                from pdf2image import convert_from_bytes
                images = convert_from_bytes(pdf_content, dpi=200)
                return {
                    'success': True,
                    'page_count': len(images),
                    'page_images': images
                }
            except ImportError:
                pass
            
            # Fallback to PyMuPDF
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=pdf_content, filetype="pdf")
                images = []
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    # Render at 200 DPI
                    mat = fitz.Matrix(200/72, 200/72)
                    pix = page.get_pixmap(matrix=mat)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
                
                doc.close()
                
                return {
                    'success': True,
                    'page_count': len(images),
                    'page_images': images
                }
            except ImportError:
                return {
                    'success': False,
                    'page_count': 0,
                    'page_images': [],
                    'error': 'No PDF library available (pdf2image or PyMuPDF)'
                }
                
        except Exception as e:
            return {
                'success': False,
                'page_count': 0,
                'page_images': [],
                'error': str(e)
            }
    
    def _handle_image(self, image_content: bytes) -> Dict[str, Any]:
        """Handle single image file."""
        try:
            img = Image.open(io.BytesIO(image_content))
            return {
                'success': True,
                'page_count': 1,
                'page_images': [img]
            }
        except Exception as e:
            return {
                'success': False,
                'page_count': 0,
                'page_images': [],
                'error': str(e)
            }
    
    def ocr_page(
        self,
        page_image: Image.Image,
        page_number: int
    ) -> Dict[str, Any]:
        """
        Run OCR on a single page image.
        Uses Nanonets as primary, Tesseract as fallback.
        
        Returns:
            {
                'page_number': int,
                'blocks': List[dict],
                'full_text': str,
                'avg_confidence': float,
                'method': str,
                'success': bool
            }
        """
        # Try Tesseract (most reliable fallback)
        if self.tesseract_available:
            result = self._ocr_tesseract(page_image, page_number)
            if result['success']:
                return result
        
        # If nothing available, return empty
        return {
            'page_number': page_number,
            'blocks': [],
            'full_text': '',
            'avg_confidence': 0,
            'block_count': 0,
            'text_length': 0,
            'method': 'none',
            'success': False,
            'error': 'No OCR engine available'
        }
    
    def _ocr_tesseract(
        self,
        page_image: Image.Image,
        page_number: int
    ) -> Dict[str, Any]:
        """Run Tesseract OCR."""
        try:
            import pytesseract
            
            # Get detailed data with bounding boxes
            data = pytesseract.image_to_data(
                page_image,
                output_type=pytesseract.Output.DICT,
                config='--psm 6'  # Assume uniform block of text
            )
            
            blocks = []
            current_block = {'text': [], 'bbox': None, 'confidences': []}
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = float(data['conf'][i])
                
                if text and conf > 0:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    blocks.append({
                        'text': text,
                        'bbox': [x, y, x + w, y + h],
                        'confidence': conf / 100,  # Normalize to 0-1
                        'block_type': 'text'
                    })
            
            # Combine text
            full_text = ' '.join(b['text'] for b in blocks)
            
            # Calculate average confidence
            confidences = [b['confidence'] for b in blocks if b['confidence'] > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Apply PII redaction
            full_text = self._redact_pii(full_text)
            for block in blocks:
                block['text'] = self._redact_pii(block['text'])
            
            return {
                'page_number': page_number,
                'blocks': blocks,
                'full_text': full_text,
                'avg_confidence': avg_confidence,
                'block_count': len(blocks),
                'text_length': len(full_text),
                'method': 'tesseract',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"[OCR][tesseract] Error: {e}")
            return {
                'page_number': page_number,
                'blocks': [],
                'full_text': '',
                'avg_confidence': 0,
                'block_count': 0,
                'text_length': 0,
                'method': 'tesseract',
                'success': False,
                'error': str(e)
            }
    
    def _redact_pii(self, text: str) -> str:
        """Redact PII patterns from text."""
        # Email addresses
        text = re.sub(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            '[EMAIL_REDACTED]',
            text
        )
        
        # Phone numbers (various formats)
        text = re.sub(
            r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            '[PHONE_REDACTED]',
            text
        )
        
        # Indian phone numbers
        text = re.sub(
            r'\+91[-.\s]?\d{10}',
            '[PHONE_REDACTED]',
            text
        )
        
        return text
    
    def chunk_and_embed(
        self,
        doc_id: str,
        ocr_results: List[Dict[str, Any]],
        doc_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Chunk OCR results and generate embeddings.
        
        Returns:
            {
                'chunks': List[dict],
                'chunk_count': int,
                'total_tokens': int
            }
        """
        chunks = []
        total_tokens = 0
        
        for page_result in ocr_results:
            if not page_result.get('success'):
                continue
            
            page_num = page_result['page_number']
            full_text = page_result.get('full_text', '')
            blocks = page_result.get('blocks', [])
            
            # Chunk the page text
            page_chunks = self._chunk_text(
                text=full_text,
                page_number=page_num,
                blocks=blocks,
                doc_id=doc_id
            )
            
            for chunk in page_chunks:
                total_tokens += chunk['token_count']
            
            chunks.extend(page_chunks)
        
        # Generate embeddings
        if self.embedder and chunks:
            texts = [c['text'] for c in chunks]
            embeddings = self.embedder.encode(texts, show_progress_bar=False)
            
            for i, chunk in enumerate(chunks):
                chunk['embedding'] = embeddings[i].tolist()
        
        return {
            'chunks': chunks,
            'chunk_count': len(chunks),
            'total_tokens': total_tokens
        }
    
    def _chunk_text(
        self,
        text: str,
        page_number: int,
        blocks: List[Dict],
        doc_id: str,
        chunk_size: int = 500,
        overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        Maintains page and block provenance.
        """
        import uuid
        
        if not text.strip():
            return []
        
        # Simple word-based chunking
        words = text.split()
        chunks = []
        
        i = 0
        chunk_index = 0
        
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Find approximate bbox from blocks
            bbox = self._find_chunk_bbox(chunk_text, blocks)
            
            chunks.append({
                'id': str(uuid.uuid4()),
                'document_id': doc_id,
                'page_number': page_number,
                'chunk_index': chunk_index,
                'text': chunk_text,
                'preview_text': chunk_text[:200],
                'bbox': bbox,
                'token_count': len(chunk_words),
                'content_type': 'text'
            })
            
            chunk_index += 1
            i += chunk_size - overlap
        
        return chunks
    
    def _find_chunk_bbox(
        self,
        chunk_text: str,
        blocks: List[Dict]
    ) -> Optional[List[int]]:
        """Find bounding box for a chunk from its source blocks."""
        if not blocks:
            return None
        
        # Find blocks that contain parts of the chunk text
        matching_blocks = []
        for block in blocks:
            if block.get('text') and block['text'] in chunk_text:
                matching_blocks.append(block)
        
        if not matching_blocks:
            return None
        
        # Combine bboxes
        x1 = min(b['bbox'][0] for b in matching_blocks)
        y1 = min(b['bbox'][1] for b in matching_blocks)
        x2 = max(b['bbox'][2] for b in matching_blocks)
        y2 = max(b['bbox'][3] for b in matching_blocks)
        
        return [x1, y1, x2, y2]
    
    def index_to_qdrant(
        self,
        doc_id: str,
        class_id: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Index chunks to Qdrant vector database.
        
        Returns:
            {
                'indexed_count': int,
                'collection': str
            }
        """
        try:
            from .qdrant_service import QdrantService, ChunkMetadata, SourceType
            
            qdrant = QdrantService()
            collection_name = f"class_{class_id}_docs"
            
            # Ensure collection exists
            # TODO: Create collection if not exists
            
            indexed_count = 0
            
            for chunk in chunks:
                if 'embedding' not in chunk:
                    continue
                
                metadata = ChunkMetadata(
                    document_id=doc_id,
                    chunk_id=chunk['id'],
                    chunk_index=chunk['chunk_index'],
                    chunk_text=chunk['text'],
                    source_type=SourceType.TEACHER_MATERIAL.value,
                    source_confidence=0.95,  # High trust for teacher materials
                    student_id="",
                    classroom_id=class_id,
                    page_number=chunk['page_number'],
                    title=chunk.get('preview_text', '')[:100],
                    subject="",
                    created_at=None,
                    url=""
                )
                
                # Store bbox in metadata
                if chunk.get('bbox'):
                    metadata.url = f"bbox:{chunk['bbox']}"  # Hack: store bbox in url field
                
                qdrant.index_single(
                    embedding=chunk['embedding'],
                    metadata=metadata,
                    collection_name=collection_name
                )
                
                indexed_count += 1
            
            return {
                'indexed_count': indexed_count,
                'collection': collection_name
            }
            
        except Exception as e:
            logger.error(f"[QDRANT] Indexing error: {e}")
            return {
                'indexed_count': 0,
                'collection': '',
                'error': str(e)
            }
