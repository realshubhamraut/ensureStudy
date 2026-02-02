"""
Syllabus Hierarchy Extractor

Extracts hierarchical structure from syllabus documents:
Classroom (Subject) → Chapters/Lessons → Topics

Uses LLM (Mistral/Groq) to parse syllabus PDFs and return structured
chapter-topic hierarchy for database storage.

Different from topic_extractor.py:
- Returns chapters with nested topics
- Assigns colors to chapters for UI grouping
- Designed for classroom syllabus (teacher-uploaded)
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Chapter color palette (same as curriculum.py)
CHAPTER_COLORS = [
    '#3B82F6',  # Blue
    '#10B981',  # Emerald
    '#F59E0B',  # Amber
    '#EF4444',  # Red
    '#8B5CF6',  # Violet
    '#EC4899',  # Pink
    '#06B6D4',  # Cyan
    '#84CC16',  # Lime
    '#F97316',  # Orange
    '#14B8A6',  # Teal
]


@dataclass
class ExtractedTopic:
    """Topic within a chapter."""
    name: str
    description: Optional[str] = None
    difficulty: str = "medium"  # easy, medium, hard
    estimated_hours: float = 1.0
    key_concepts: List[str] = field(default_factory=list)
    order: int = 0


@dataclass
class ExtractedChapter:
    """Chapter/Lesson with topics."""
    name: str
    description: Optional[str] = None
    topics: List[ExtractedTopic] = field(default_factory=list)
    estimated_hours: float = 2.0
    color: str = '#3B82F6'
    order: int = 0


@dataclass
class ExtractedSyllabusHierarchy:
    """Complete syllabus hierarchy."""
    subject_name: str
    chapters: List[ExtractedChapter] = field(default_factory=list)
    total_chapters: int = 0
    total_topics: int = 0
    extraction_method: str = "unknown"


# Master prompt for hierarchical extraction
HIERARCHY_EXTRACTION_PROMPT = """You are an expert curriculum designer. Extract the hierarchical structure from this syllabus document.

**TASK**: Parse the syllabus and return a structured breakdown of:
1. **Chapters/Lessons** (major sections/units)
2. **Topics** within each chapter (specific concepts to learn)

**RULES**:
- Only include content explicitly mentioned in the syllabus
- Do NOT add topics that aren't in the document
- Keep topic names concise (3-8 words)
- Estimate difficulty based on complexity (easy/medium/hard)
- Estimate hours based on typical study time

**SUBJECT**: {subject_name}

**SYLLABUS CONTENT**:
{syllabus_text}

**RESPONSE FORMAT** (JSON only, no markdown):
{{
  "subject": "{subject_name}",
  "chapters": [
    {{
      "name": "Chapter 1: Introduction to XYZ",
      "description": "Brief overview of the chapter",
      "estimated_hours": 3.0,
      "topics": [
        {{
          "name": "Topic name here",
          "description": "What this topic covers",
          "difficulty": "easy|medium|hard",
          "estimated_hours": 1.0,
          "key_concepts": ["concept1", "concept2"]
        }}
      ]
    }}
  ]
}}

Return ONLY valid JSON, no explanations or markdown.
"""


class SyllabusHierarchyExtractor:
    """
    Extracts chapter-topic hierarchy from syllabus documents.
    Uses LLM for intelligent parsing.
    """
    
    def __init__(self):
        self._fitz = None
    
    def _ensure_fitz(self):
        """Lazy load PyMuPDF."""
        if self._fitz is None:
            try:
                import fitz
                self._fitz = fitz
            except ImportError:
                raise ImportError("PyMuPDF (fitz) required. Install with: pip install pymupdf")
    
    def extract_hierarchy(
        self, 
        pdf_path: str, 
        subject_name: str = ""
    ) -> ExtractedSyllabusHierarchy:
        """
        Extract hierarchical syllabus structure from PDF.
        
        Args:
            pdf_path: Path to syllabus PDF
            subject_name: Subject/classroom name
            
        Returns:
            ExtractedSyllabusHierarchy with chapters and topics
        """
        self._ensure_fitz()
        
        try:
            doc = self._fitz.open(pdf_path)
            
            # Extract text from all pages
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            doc.close()
            
            if not full_text.strip():
                logger.warning("PDF appears to be empty or image-based")
                return self._fallback_hierarchy(subject_name)
            
            # Truncate if too long (for LLM context limits)
            max_chars = 30000
            if len(full_text) > max_chars:
                logger.info(f"Truncating syllabus text from {len(full_text)} to {max_chars} chars")
                full_text = full_text[:max_chars]
            
            # Try LLM extraction
            hierarchy = self._extract_with_llm(full_text, subject_name)
            if hierarchy and hierarchy.chapters:
                return hierarchy
            
            # Fallback to pattern-based extraction
            hierarchy = self._extract_with_patterns(full_text, subject_name)
            if hierarchy and hierarchy.chapters:
                return hierarchy
            
            # Ultimate fallback
            return self._fallback_hierarchy(subject_name)
            
        except Exception as e:
            logger.error(f"Hierarchy extraction failed: {e}")
            return self._fallback_hierarchy(subject_name)
    
    def extract_from_text(
        self, 
        text: str, 
        subject_name: str = ""
    ) -> ExtractedSyllabusHierarchy:
        """
        Extract hierarchy from plain text (already extracted from PDF/OCR).
        """
        if not text.strip():
            return self._fallback_hierarchy(subject_name)
        
        # Truncate if needed
        max_chars = 30000
        if len(text) > max_chars:
            text = text[:max_chars]
        
        hierarchy = self._extract_with_llm(text, subject_name)
        if hierarchy and hierarchy.chapters:
            return hierarchy
        
        return self._extract_with_patterns(text, subject_name) or self._fallback_hierarchy(subject_name)
    
    def _extract_with_llm(
        self, 
        text: str, 
        subject_name: str
    ) -> Optional[ExtractedSyllabusHierarchy]:
        """
        Use LLM (Mistral/Groq) to extract hierarchy.
        Priority: Mistral > Groq > Gemini
        """
        # Try Mistral first
        result = self._try_mistral(text, subject_name)
        if result:
            return result
        
        # Try Groq
        result = self._try_groq(text, subject_name)
        if result:
            return result
        
        logger.warning("All LLM providers failed for hierarchy extraction")
        return None
    
    def _try_mistral(
        self, 
        text: str, 
        subject_name: str
    ) -> Optional[ExtractedSyllabusHierarchy]:
        """Try Mistral API for extraction."""
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            logger.debug("MISTRAL_API_KEY not set")
            return None
        
        try:
            from mistralai import Mistral
            
            client = Mistral(api_key=api_key)
            prompt = HIERARCHY_EXTRACTION_PROMPT.format(
                subject_name=subject_name or "General",
                syllabus_text=text
            )
            
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content
            return self._parse_llm_response(content, subject_name, "mistral")
            
        except Exception as e:
            logger.warning(f"Mistral extraction failed: {e}")
            return None
    
    def _try_groq(
        self, 
        text: str, 
        subject_name: str
    ) -> Optional[ExtractedSyllabusHierarchy]:
        """Try Groq API for extraction."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.debug("GROQ_API_KEY not set")
            return None
        
        try:
            from groq import Groq
            
            client = Groq(api_key=api_key)
            prompt = HIERARCHY_EXTRACTION_PROMPT.format(
                subject_name=subject_name or "General",
                syllabus_text=text
            )
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content
            return self._parse_llm_response(content, subject_name, "groq")
            
        except Exception as e:
            logger.warning(f"Groq extraction failed: {e}")
            return None
    
    def _parse_llm_response(
        self, 
        content: str, 
        subject_name: str, 
        method: str
    ) -> Optional[ExtractedSyllabusHierarchy]:
        """Parse LLM JSON response into dataclass."""
        try:
            # Clean up response
            content = content.strip()
            
            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            
            # Parse JSON
            data = json.loads(content)
            
            chapters = []
            for idx, ch_data in enumerate(data.get("chapters", [])):
                topics = []
                for t_idx, t_data in enumerate(ch_data.get("topics", [])):
                    topic = ExtractedTopic(
                        name=t_data.get("name", f"Topic {t_idx + 1}"),
                        description=t_data.get("description"),
                        difficulty=t_data.get("difficulty", "medium"),
                        estimated_hours=float(t_data.get("estimated_hours", 1.0)),
                        key_concepts=t_data.get("key_concepts", []),
                        order=t_idx
                    )
                    topics.append(topic)
                
                chapter = ExtractedChapter(
                    name=ch_data.get("name", f"Chapter {idx + 1}"),
                    description=ch_data.get("description"),
                    topics=topics,
                    estimated_hours=float(ch_data.get("estimated_hours", 2.0)),
                    color=CHAPTER_COLORS[idx % len(CHAPTER_COLORS)],
                    order=idx
                )
                chapters.append(chapter)
            
            total_topics = sum(len(ch.topics) for ch in chapters)
            
            return ExtractedSyllabusHierarchy(
                subject_name=subject_name or data.get("subject", "Unknown"),
                chapters=chapters,
                total_chapters=len(chapters),
                total_topics=total_topics,
                extraction_method=method
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return None
    
    def _extract_with_patterns(
        self, 
        text: str, 
        subject_name: str
    ) -> Optional[ExtractedSyllabusHierarchy]:
        """
        Fallback pattern-based extraction.
        Looks for chapter/unit headers and bullet points.
        """
        import re
        
        chapters = []
        
        # Pattern for chapter headers
        chapter_pattern = re.compile(
            r'(?:Chapter|Unit|Module|Lesson|Part)\s*[\d:.\-]+\s*[:\-]?\s*(.+)',
            re.IGNORECASE
        )
        
        # Pattern for topics (bullet points, numbered items)
        topic_pattern = re.compile(
            r'^\s*(?:[\d•\-\*]+[.)]\s*|\d+[.)]\s*)(.+)',
            re.MULTILINE
        )
        
        lines = text.split('\n')
        current_chapter = None
        current_topics = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for chapter header
            chapter_match = chapter_pattern.match(line)
            if chapter_match:
                # Save previous chapter
                if current_chapter:
                    chapters.append(ExtractedChapter(
                        name=current_chapter,
                        topics=[ExtractedTopic(name=t, order=i) for i, t in enumerate(current_topics)],
                        color=CHAPTER_COLORS[len(chapters) % len(CHAPTER_COLORS)],
                        order=len(chapters)
                    ))
                
                current_chapter = chapter_match.group(1).strip()
                current_topics = []
                continue
            
            # Check for topic (if we're in a chapter)
            if current_chapter:
                topic_match = topic_pattern.match(line)
                if topic_match:
                    topic_name = topic_match.group(1).strip()
                    if len(topic_name) > 5 and len(topic_name) < 200:
                        current_topics.append(topic_name)
        
        # Save last chapter
        if current_chapter:
            chapters.append(ExtractedChapter(
                name=current_chapter,
                topics=[ExtractedTopic(name=t, order=i) for i, t in enumerate(current_topics)],
                color=CHAPTER_COLORS[len(chapters) % len(CHAPTER_COLORS)],
                order=len(chapters)
            ))
        
        if chapters:
            total_topics = sum(len(ch.topics) for ch in chapters)
            return ExtractedSyllabusHierarchy(
                subject_name=subject_name or "Unknown",
                chapters=chapters,
                total_chapters=len(chapters),
                total_topics=total_topics,
                extraction_method="pattern"
            )
        
        return None
    
    def _fallback_hierarchy(self, subject_name: str) -> ExtractedSyllabusHierarchy:
        """Return empty hierarchy as fallback."""
        return ExtractedSyllabusHierarchy(
            subject_name=subject_name or "Unknown Subject",
            chapters=[],
            total_chapters=0,
            total_topics=0,
            extraction_method="fallback"
        )
    
    def hierarchy_to_dict(self, hierarchy: ExtractedSyllabusHierarchy) -> Dict[str, Any]:
        """Convert hierarchy to dictionary for API response."""
        return {
            "subject_name": hierarchy.subject_name,
            "chapters": [
                {
                    "name": ch.name,
                    "description": ch.description,
                    "color": ch.color,
                    "estimated_hours": ch.estimated_hours,
                    "order": ch.order,
                    "topics": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "difficulty": t.difficulty,
                            "estimated_hours": t.estimated_hours,
                            "key_concepts": t.key_concepts,
                            "order": t.order
                        }
                        for t in ch.topics
                    ]
                }
                for ch in hierarchy.chapters
            ],
            "total_chapters": hierarchy.total_chapters,
            "total_topics": hierarchy.total_topics,
            "extraction_method": hierarchy.extraction_method
        }


# Singleton instance
_extractor_instance: Optional[SyllabusHierarchyExtractor] = None


def get_syllabus_hierarchy_extractor() -> SyllabusHierarchyExtractor:
    """Get singleton extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = SyllabusHierarchyExtractor()
    return _extractor_instance
