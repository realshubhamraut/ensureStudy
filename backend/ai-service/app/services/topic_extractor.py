"""
Syllabus-Bound Educational Intelligence Engine

Implements the Master AI Prompt for:
1. Extracting database-ready topic hierarchies (atomic, teachable concepts)
2. Chunking content for vector storage
3. Strict syllabus boundaries - no hallucination

Priority: TOC > LLM (Groq 70B) > Pattern > Heuristics
"""
import os
import re
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Subtopic:
    """Atomic teachable concept."""
    subtopic_name: str
    definition_required: bool = False
    formula_required: bool = False
    diagram_required: bool = False
    difficulty_level: str = "medium"


@dataclass
class ExtractedTopic:
    """Represents an extracted topic/chapter with subtopics."""
    name: str
    description: Optional[str] = None
    subtopics: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    estimated_hours: float = 2.0
    page: int = 0
    level: int = 1
    source: str = "unknown"
    # Enhanced fields for syllabus-bound system
    unit: Optional[str] = None
    chapter: Optional[str] = None
    requires_formula: bool = False
    requires_diagram: bool = False


# Master prompt for ATOMIC syllabus extraction
SYLLABUS_EXTRACTION_PROMPT = """You are an expert curriculum analyst. Your task is to extract EVERY SINGLE atomic micro-topic from this syllabus.

CRITICAL INSTRUCTIONS:
1. Extract MAXIMUM topics - aim for 50-200 micro-topics
2. Each micro-topic = ONE teachable concept (e.g., "Evaporation", "Melting point", "Electron")
3. Use hierarchical structure: Unit → Chapter → Topics with many subtopics
4. Split aggressively - prefer over-splitting
5. Stay STRICTLY within the text content

CONTENT TO ANALYZE:
{content}

SUBJECT: {subject_name}

OUTPUT FORMAT (JSON ONLY):
{{
  "subject": "{subject_name}",
  "units": [
    {{
      "unit_name": "<Unit Name>",
      "chapters": [
        {{
          "chapter_name": "<Chapter Name>",
          "topics": [
            {{
              "topic_name": "<Topic>",
              "micro_topics": [
                "<Atomic concept 1>",
                "<Atomic concept 2>",
                "<Atomic concept 3>"
              ],
              "difficulty": "easy|medium|hard"
            }}
          ]
        }}
      ]
    }}
  ]
}}

EXTRACTION RULES:
- Every definition = separate micro-topic
- Every law/principle = separate micro-topic  
- Every formula = separate micro-topic
- Every process step = separate micro-topic
- Every property = separate micro-topic
- Every example = separate micro-topic as "Example: X"
- Every diagram concept = separate micro-topic

EXAMPLE for Physics "Motion" chapter:
- Rest
- Motion
- Distance  
- Displacement
- Speed
- Velocity
- Uniform motion
- Non-uniform motion
- Acceleration
- Distance-time graph
- Velocity-time graph
- Equations of motion - First equation
- Equations of motion - Second equation
- Equations of motion - Third equation
- Uniform circular motion

Return ONLY valid JSON. Extract AS MANY micro-topics as possible."""


class TopicExtractor:
    """Syllabus-bound multi-method topic extractor."""
    
    def __init__(self):
        self._fitz = None
    
    def _ensure_fitz(self):
        """Lazy load PyMuPDF."""
        if self._fitz is None:
            import fitz
            self._fitz = fitz
    
    def extract_topics(
        self,
        pdf_path: str,
        subject_name: str = "",
        prefer_ai: bool = True
    ) -> List[ExtractedTopic]:
        """Extract topics using the best available method."""
        self._ensure_fitz()
        
        doc = self._fitz.open(pdf_path)
        full_text = ""
        pages_text = []
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text").strip()
                full_text += f"\n{text}"
                pages_text.append({"page": page_num + 1, "text": text})
            
            topics = []
            
            # Method 1: PDF Table of Contents
            toc_topics = self._extract_from_toc(doc)
            if toc_topics and len(toc_topics) >= 3:
                print(f"[TOPIC-EXTRACTOR] ✓ Found {len(toc_topics)} topics from TOC")
                topics = toc_topics
            
            # Method 2: AI extraction with Master Prompt (Groq 70B)
            if not topics or (prefer_ai and len(topics) < 5):
                ai_topics = self._extract_with_master_prompt(full_text[:10000], subject_name)
                if ai_topics and len(ai_topics) >= len(topics):
                    print(f"[TOPIC-EXTRACTOR] ✓ AI extracted {len(ai_topics)} atomic topics")
                    topics = ai_topics
            
            # Method 3: Pattern-based extraction
            if not topics or len(topics) < 3:
                pattern_topics = self._extract_with_patterns(pages_text)
                if pattern_topics and len(pattern_topics) > len(topics):
                    print(f"[TOPIC-EXTRACTOR] ✓ Pattern-based: {len(pattern_topics)} topics")
                    topics = pattern_topics
            
            # Method 4: Heuristic fallback
            if not topics or len(topics) < 2:
                heuristic_topics = self._extract_with_heuristics(pages_text)
                if heuristic_topics:
                    print(f"[TOPIC-EXTRACTOR] ✓ Heuristics: {len(heuristic_topics)} topics")
                    topics = heuristic_topics
            
            if not topics:
                print("[TOPIC-EXTRACTOR] ⚠ Using page-based fallback")
                topics = self._fallback_page_topics(len(doc))
            
            return topics
            
        finally:
            doc.close()
    
    def _extract_from_toc(self, doc) -> List[ExtractedTopic]:
        """Extract from PDF's embedded Table of Contents."""
        try:
            toc = doc.get_toc()
            if not toc:
                return []
            
            topics = []
            seen = set()
            
            for level, title, page in toc:
                title = title.strip()
                title_lower = title.lower()
                
                if not title or len(title) < 2 or title_lower in seen:
                    continue
                if title_lower in ['contents', 'table of contents', 'index', 'bibliography']:
                    continue
                
                seen.add(title_lower)
                topics.append(ExtractedTopic(
                    name=title, page=page, level=level, source="toc"
                ))
            
            return topics
        except Exception as e:
            logger.warning(f"TOC extraction failed: {e}")
            return []
    
    def _extract_with_master_prompt(self, text: str, subject_name: str) -> List[ExtractedTopic]:
        """Use Master AI Prompt with Groq for atomic topic extraction."""
        topics = self._extract_with_groq(text, subject_name)
        if topics:
            return topics
        
        # Fallback to Gemini
        topics = self._extract_with_gemini(text, subject_name)
        return topics if topics else []
    
    def _extract_with_groq(self, text: str, subject_name: str) -> List[ExtractedTopic]:
        """Use Groq API with llama-3.3-70b-versatile for atomic topic extraction."""
        try:
            import httpx
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.info("[TOPIC-EXTRACTOR] No GROQ_API_KEY, skipping Groq")
                return []
            
            prompt = SYLLABUS_EXTRACTION_PROMPT.format(
                content=text[:15000],  # Send more context
                subject_name=subject_name or "General"
            )
            
            print("[TOPIC-EXTRACTOR] Calling Groq llama-3.3-70b-versatile...")
            
            response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 8000  # More tokens for 100+ topics
                },
                timeout=120.0  # Longer timeout
            )
            
            if response.status_code != 200:
                logger.warning(f"Groq API error: {response.status_code} - {response.text[:200]}")
                return []
            
            data = response.json()
            text_response = data["choices"][0]["message"]["content"].strip()
            
            # Clean JSON from markdown
            if "```" in text_response:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text_response)
                if match:
                    text_response = match.group(1)
            
            result = json.loads(text_response)
            
            # Parse hierarchical format and FLATTEN to individual micro-topics
            topics = []
            
            # Handle new hierarchical format
            for unit_data in result.get("units", []):
                unit_name = unit_data.get("unit_name", "General")
                
                for chapter_data in unit_data.get("chapters", []):
                    chapter_name = chapter_data.get("chapter_name", "")
                    
                    for topic_data in chapter_data.get("topics", []):
                        topic_name = topic_data.get("topic_name", "")
                        if not topic_name:
                            continue
                        
                        # Get all micro-topics
                        micro_topics = topic_data.get("micro_topics", [])
                        if isinstance(micro_topics, list):
                            micro_topics = [m for m in micro_topics if isinstance(m, str) and m]
                        else:
                            micro_topics = []
                        
                        difficulty = topic_data.get("difficulty", "medium")
                        
                        # FLATTEN: Create a separate topic for EACH micro-topic
                        if micro_topics:
                            for micro_topic in micro_topics:
                                topics.append(ExtractedTopic(
                                    name=micro_topic,  # Individual micro-topic as name
                                    description=f"{chapter_name} → {topic_name}",
                                    subtopics=[],
                                    difficulty=difficulty,
                                    estimated_hours=0.25,  # 15 min per micro-topic
                                    source="groq",
                                    unit=unit_name,
                                    chapter=chapter_name
                                ))
                        else:
                            # No micro-topics, use topic_name itself
                            topics.append(ExtractedTopic(
                                name=topic_name,
                                description=f"Part of {chapter_name}" if chapter_name else None,
                                subtopics=[],
                                difficulty=difficulty,
                                estimated_hours=0.5,
                                source="groq",
                                unit=unit_name,
                                chapter=chapter_name
                            ))
            
            # Fallback: Handle old format (topics with subtopics)
            if not topics and result.get("topics"):
                for topic_data in result.get("topics", []):
                    topic_name = topic_data.get("topic_name", "")
                    if not topic_name:
                        continue
                    
                    subtopics = []
                    for st in topic_data.get("subtopics", topic_data.get("micro_topics", [])):
                        if isinstance(st, dict):
                            subtopics.append(st.get("subtopic_name", st.get("name", "")))
                        elif isinstance(st, str):
                            subtopics.append(st)
                    
                    # FLATTEN: Each subtopic as separate entry
                    if subtopics:
                        for st in subtopics:
                            if st:
                                topics.append(ExtractedTopic(
                                    name=st,
                                    description=f"Part of {topic_name}",
                                    subtopics=[],
                                    difficulty=topic_data.get("difficulty", "medium"),
                                    source="groq"
                                ))
                    else:
                        topics.append(ExtractedTopic(
                            name=topic_name,
                            subtopics=[],
                            difficulty=topic_data.get("difficulty", "medium"),
                            source="groq"
                        ))
            
            print(f"[TOPIC-EXTRACTOR] Groq: {len(topics)} individual atomic topics extracted")
            return topics
            
        except json.JSONDecodeError as e:
            logger.warning(f"Groq JSON parse error: {e}")
            return []
        except Exception as e:
            logger.warning(f"Groq extraction failed: {e}")
            return []
    
    def _extract_with_gemini(self, text: str, subject_name: str) -> List[ExtractedTopic]:
        """Fallback: Use Gemini API for topic extraction."""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return []
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            prompt = SYLLABUS_EXTRACTION_PROMPT.format(
                content=text[:6000],
                subject_name=subject_name or "General"
            )
            
            response = model.generate_content(prompt)
            text_response = response.text.strip()
            
            if "```" in text_response:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text_response)
                if match:
                    text_response = match.group(1)
            
            result = json.loads(text_response)
            
            topics = []
            for topic_data in result.get("topics", []):
                topic_name = topic_data.get("topic_name", "")
                if not topic_name:
                    continue
                
                subtopic_names = []
                for st in topic_data.get("subtopics", []):
                    if isinstance(st, dict):
                        subtopic_names.append(st.get("subtopic_name", ""))
                    elif isinstance(st, str):
                        subtopic_names.append(st)
                
                topics.append(ExtractedTopic(
                    name=topic_name,
                    subtopics=[s for s in subtopic_names if s],
                    difficulty="medium",
                    estimated_hours=len(subtopic_names) * 0.5,
                    source="gemini"
                ))
            
            return topics
            
        except Exception as e:
            logger.warning(f"Gemini extraction failed: {e}")
            return []
    
    def _extract_with_patterns(self, pages_text: List[Dict]) -> List[ExtractedTopic]:
        """Pattern-based extraction using regex."""
        topics = []
        seen = set()
        
        patterns = [
            (r'^(?:CHAPTER|Chapter)\s*[\dIVXivx]+[:\.\-\s]+(.+)$', 1),
            (r'^(?:UNIT|Unit)\s*[\dIVXivx]+[:\.\-\s]+(.+)$', 1),
            (r'^(?:MODULE|Module)\s*[\dIVXivx]+[:\.\-\s]+(.+)$', 1),
            (r'^(?:LESSON|Lesson)\s*[\dIVXivx]+[:\.\-\s]+(.+)$', 1),
            (r'^(?:TOPIC|Topic)\s*[\dIVXivx]+[:\.\-\s]+(.+)$', 1),
            (r'^(\d+(?:\.\d+)?)[:\.\)]\s*([A-Z][a-zA-Z\s\-&]{3,})$', 1),
        ]
        
        for page_data in pages_text:
            text = page_data.get("text", "")
            page = page_data.get("page", 0)
            
            for line in text.split("\n"):
                line = line.strip()
                if not line or len(line) < 5 or len(line) > 150:
                    continue
                
                for pattern, level in patterns:
                    match = re.match(pattern, line)
                    if match:
                        name = match.group(1) if match.lastindex >= 1 else line
                        if pattern == r'^(\d+(?:\.\d+)?)[:\.\)]\s*([A-Z][a-zA-Z\s\-&]{3,})$':
                            name = match.group(2)
                        
                        name = name.strip()
                        name_lower = name.lower()
                        
                        if name and name_lower not in seen and len(name) > 3:
                            seen.add(name_lower)
                            topics.append(ExtractedTopic(
                                name=name, page=page, level=level, source="pattern"
                            ))
                        break
        
        return topics
    
    def _extract_with_heuristics(self, pages_text: List[Dict]) -> List[ExtractedTopic]:
        """Heuristic-based extraction."""
        topics = []
        seen = set()
        
        keywords = ['introduction', 'overview', 'fundamentals', 'basics',
                    'principles', 'theory', 'concepts', 'methods', 'analysis']
        
        for page_data in pages_text:
            text = page_data.get("text", "")
            page = page_data.get("page", 0)
            
            for line in text.split("\n"):
                line = line.strip()
                if len(line) < 5 or len(line) > 100:
                    continue
                
                line_lower = line.lower()
                is_heading = False
                
                if line.isupper() and len(line) > 5:
                    is_heading = True
                elif line.istitle() and any(kw in line_lower for kw in keywords):
                    is_heading = True
                elif re.match(r'^\d+\.?\s+[A-Z]', line) and len(line) < 60:
                    is_heading = True
                
                if is_heading:
                    name = re.sub(r'^[\d\.\s]+', '', line).strip()
                    name_lower = name.lower()
                    
                    skip = ['page', 'contents', 'index', 'figure', 'table']
                    if any(sw in name_lower for sw in skip):
                        continue
                    
                    if name and name_lower not in seen and len(name) > 3:
                        seen.add(name_lower)
                        topics.append(ExtractedTopic(
                            name=name[:80], page=page, source="heuristic"
                        ))
        
        return topics
    
    def _fallback_page_topics(self, num_pages: int) -> List[ExtractedTopic]:
        """Ultimate fallback: page-based sections."""
        topics = []
        pages_per = max(1, num_pages // 5)
        
        for i in range(0, num_pages, pages_per):
            end = min(i + pages_per, num_pages)
            topics.append(ExtractedTopic(
                name=f"Section {(i // pages_per) + 1}: Pages {i+1}-{end}",
                page=i + 1,
                source="fallback"
            ))
        
        return topics


_extractor_instance: Optional[TopicExtractor] = None


def get_topic_extractor() -> TopicExtractor:
    """Get singleton TopicExtractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = TopicExtractor()
    return _extractor_instance
