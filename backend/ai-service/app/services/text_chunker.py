"""
Text Chunker - Semantic-Aware Text Chunking

Features:
- Configurable chunk size (tokens)
- Overlap between chunks
- Respects sentence boundaries
- Preserves formulas/equations
- Adds rich metadata to chunks
"""
import re
import uuid
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with metadata"""
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    start_char: int
    end_char: int
    page_number: int
    section_heading: Optional[str]
    source_confidence: float
    contains_formula: bool
    formula_latex: Optional[str] = None


class TextChunker:
    """
    Semantic-aware text chunker
    
    Args:
        chunk_size: Target chunk size in tokens (default: 500)
        chunk_overlap: Overlap between chunks in tokens (default: 100)
        respect_sentences: Try to break at sentence boundaries (default: True)
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        respect_sentences: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_sentences = respect_sentences
        
        # Simple tokenizer (approximate: 1 token ≈ 4 chars for English)
        self.chars_per_token = 4
        
        # Formula patterns
        self.formula_patterns = [
            r'\$[^$]+\$',  # Inline math: $...$
            r'\$\$[^$]+\$\$',  # Display math: $$...$$
            r'\\begin\{equation\}.*?\\end\{equation\}',  # LaTeX equations
            r'\\[.*?\\]',  # LaTeX display: \[...\]
        ]
        
        # Sentence endings
        self.sentence_end_pattern = re.compile(r'[.!?]\s+')
        
        # Section heading patterns
        self.heading_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^([A-Z][A-Za-z\s]+):$',  # Title: format
            r'^(\d+\.?\s+[A-Z].+)$',  # Numbered sections
        ]
    
    def chunk_text(
        self,
        text: str,
        document_id: str,
        page_number: int = 1,
        source_confidence: float = 0.8,
        metadata: Dict[str, Any] = None
    ) -> List[Dict]:
        """
        Chunk text into semantic segments
        
        Args:
            text: Text to chunk
            document_id: Parent document ID
            page_number: Source page number
            source_confidence: OCR confidence score
            metadata: Additional metadata
        
        Returns:
            List of chunk dictionaries
        """
        if not text or not text.strip():
            return []
        
        # Normalize whitespace
        text = self._normalize_text(text)
        
        # Detect section headings
        current_heading = self._detect_heading(text)
        
        # Detect formulas
        formulas = self._extract_formulas(text)
        
        # Calculate chunk parameters in characters
        target_chars = self.chunk_size * self.chars_per_token
        overlap_chars = self.chunk_overlap * self.chars_per_token
        
        chunks = []
        start_pos = 0
        chunk_index = 0
        
        while start_pos < len(text):
            # Calculate end position
            end_pos = min(start_pos + target_chars, len(text))
            
            # Adjust to sentence boundary if enabled
            if self.respect_sentences and end_pos < len(text):
                end_pos = self._find_sentence_boundary(text, start_pos, end_pos)
            
            # Extract chunk text
            chunk_text = text[start_pos:end_pos].strip()
            
            if chunk_text:
                # Check for formulas in this chunk
                chunk_formulas = [f for f in formulas if start_pos <= f['start'] < end_pos]
                contains_formula = len(chunk_formulas) > 0
                formula_latex = chunk_formulas[0]['latex'] if chunk_formulas else None
                
                chunk = {
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "token_count": len(chunk_text) // self.chars_per_token,
                    "start_char": start_pos,
                    "end_char": end_pos,
                    "page_number": page_number,
                    "section_heading": current_heading,
                    "source_confidence": source_confidence,
                    "contains_formula": contains_formula,
                    "formula_latex": formula_latex
                }
                
                # Add any additional metadata
                if metadata:
                    chunk.update(metadata)
                
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position (with overlap)
            start_pos = end_pos - overlap_chars
            
            # Prevent infinite loop
            if start_pos >= len(text) - 10:
                break
        
        logger.debug(f"Created {len(chunks)} chunks from {len(text)} chars")
        return chunks
    
    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and clean text"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = text.replace('|', 'l')  # Common OCR mistake
        text = text.replace('0', 'O') if re.search(r'\bO[a-zA-Z]', text) else text
        
        return text.strip()
    
    def _detect_heading(self, text: str) -> Optional[str]:
        """Detect section heading from text start"""
        first_line = text.split('\n')[0] if '\n' in text else text[:100]
        
        for pattern in self.heading_patterns:
            match = re.match(pattern, first_line, re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_formulas(self, text: str) -> List[Dict]:
        """Extract mathematical formulas from text"""
        formulas = []
        
        for pattern in self.formula_patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                formulas.append({
                    'latex': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return formulas
    
    def _find_sentence_boundary(
        self,
        text: str,
        start: int,
        target_end: int
    ) -> int:
        """Find the nearest sentence boundary before target_end"""
        # Search backwards for sentence ending
        search_text = text[start:target_end]
        
        # Find all sentence endings
        matches = list(self.sentence_end_pattern.finditer(search_text))
        
        if matches:
            # Use the last sentence ending
            last_match = matches[-1]
            return start + last_match.end()
        
        # No sentence boundary found, try to break at word boundary
        space_pos = search_text.rfind(' ')
        if space_pos > len(search_text) * 0.5:  # Only if we keep at least half
            return start + space_pos + 1
        
        return target_end
    
    def chunk_with_context(
        self,
        text: str,
        document_id: str,
        context_before: str = "",
        context_after: str = "",
        **kwargs
    ) -> List[Dict]:
        """
        Chunk text with surrounding context for better embeddings
        
        Args:
            text: Main text to chunk
            context_before: Text from previous page/section
            context_after: Text from next page/section
            **kwargs: Additional arguments for chunk_text
        
        Returns:
            List of chunks with context prefixes
        """
        chunks = self.chunk_text(text, document_id, **kwargs)
        
        # Add context to first and last chunks
        if chunks and context_before:
            chunks[0]["context_prefix"] = context_before[-200:]
        
        if chunks and context_after:
            chunks[-1]["context_suffix"] = context_after[:200]
        
        return chunks


# Convenience function
def create_chunks(
    text: str,
    document_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    **kwargs
) -> List[Dict]:
    """
    Create text chunks from a document
    
    Args:
        text: Text to chunk
        document_id: Document ID
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks
        **kwargs: Additional metadata
    
    Returns:
        List of chunk dictionaries
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_text(text, document_id, **kwargs)
