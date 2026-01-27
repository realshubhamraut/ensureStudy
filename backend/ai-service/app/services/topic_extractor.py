"""
Syllabus-Bound Educational Intelligence Engine

Implements the Master AI Prompt for:
1. Extracting database-ready topic hierarchies (atomic, teachable concepts)
2. Chunking content for vector storage
3. Strict syllabus boundaries - no hallucination

Priority: TOC > Mistral API (500K/min) > Groq (100K/day) > Gemini > Pattern > Heuristics
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
    
    def __post_init__(self):
        # Truncate name to 200 chars (DB limit) to prevent StringDataRightTruncation errors
        if self.name and len(self.name) > 200:
            self.name = self.name[:197] + "..."


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
        """Use Master AI Prompt with LLM for atomic topic extraction.
        
        Priority: Mistral API (500K tokens/min) -> Groq (100K tokens/day) -> Gemini (20 req/day)
        """
        # Primary: Mistral API (massive free tier - 500K tokens/min, 1B/month)
        topics = self._extract_with_mistral(text, subject_name)
        if topics:
            return topics
        
        # Fallback 1: Groq (100K tokens/day limit)
        topics = self._extract_with_groq(text, subject_name)
        if topics:
            return topics
        
        # Fallback 2: Gemini (20 requests/day per model)
        topics = self._extract_with_gemini(text, subject_name)
        return topics if topics else []
    
    def _extract_with_mistral(self, text: str, subject_name: str) -> List[ExtractedTopic]:
        """Use Mistral API for atomic topic extraction.
        
        Mistral free tier: 500K tokens/min, 1B tokens/month - much higher than Gemini/Groq!
        """
        try:
            import httpx
            
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                logger.info("[TOPIC-EXTRACTOR] No MISTRAL_API_KEY, skipping Mistral")
                return []
            
            prompt = SYLLABUS_EXTRACTION_PROMPT.format(
                content=text[:20000],  # Mistral can handle large context
                subject_name=subject_name or "General"
            )
            
            print("[TOPIC-EXTRACTOR] Calling Mistral API (open-mistral-nemo)...")
            
            content = None
            
            # Try up to 2 times with increasing max_tokens
            for attempt, max_tokens in enumerate([8000, 16000]):
                response = httpx.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "open-mistral-nemo",  # Free tier model, good for structured extraction
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": max_tokens
                    },
                    timeout=180.0  # Longer timeout for larger responses
                )
                
                if response.status_code == 429:
                    print(f"[TOPIC-EXTRACTOR] Mistral rate limited: {response.text[:200]}")
                    return []
                
                if response.status_code != 200:
                    print(f"[TOPIC-EXTRACTOR] Mistral API error: {response.status_code} - {response.text[:200]}")
                    return []
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Clean JSON from markdown code blocks
                if "```" in content:
                    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                    if match:
                        content = match.group(1)
                
                # Try to parse JSON
                try:
                    data = json.loads(content)
                    break  # Success!
                except json.JSONDecodeError as e:
                    print(f"[TOPIC-EXTRACTOR] Mistral JSON error (attempt {attempt + 1}): {e}")
                    
                    # Try to repair truncated JSON by adding closing brackets
                    repaired = content.rstrip()
                    
                    # Count open/close brackets
                    open_braces = repaired.count('{') - repaired.count('}')
                    open_brackets = repaired.count('[') - repaired.count(']')
                    
                    # Add missing closures
                    repaired += ']' * max(0, open_brackets)
                    repaired += '}' * max(0, open_braces)
                    
                    try:
                        data = json.loads(repaired)
                        print("[TOPIC-EXTRACTOR] ✓ Repaired truncated JSON")
                        break
                    except json.JSONDecodeError:
                        if attempt == 0:
                            print("[TOPIC-EXTRACTOR] Retrying with more tokens...")
                            continue  # Retry with more tokens
                        else:
                            raise  # Give up after 2 attempts
            
            if not content:
                return []
            
            # Parse and flatten topics (same logic as Groq/Gemini)
            topics = []
            
            for unit_data in data.get("units", []):
                unit_name = unit_data.get("unit_name", "General")
                
                for chapter_data in unit_data.get("chapters", []):
                    chapter_name = chapter_data.get("chapter_name", "")
                    
                    for topic_data in chapter_data.get("topics", []):
                        topic_name = topic_data.get("topic_name", "")
                        if not topic_name:
                            continue
                        
                        micro_topics = topic_data.get("micro_topics", [])
                        if isinstance(micro_topics, list):
                            micro_topics = [m for m in micro_topics if isinstance(m, str) and m]
                        else:
                            micro_topics = []
                        
                        difficulty = topic_data.get("difficulty", "medium")
                        
                        # FLATTEN: Create separate topic for each micro-topic
                        if micro_topics:
                            for micro_topic in micro_topics:
                                topics.append(ExtractedTopic(
                                    name=micro_topic,
                                    description=f"{chapter_name} → {topic_name}",
                                    subtopics=[],
                                    difficulty=difficulty,
                                    estimated_hours=0.25,
                                    source="mistral",
                                    unit=unit_name,
                                    chapter=chapter_name
                                ))
                        else:
                            topics.append(ExtractedTopic(
                                name=topic_name,
                                description=f"Part of {chapter_name}" if chapter_name else None,
                                subtopics=[],
                                difficulty=difficulty,
                                estimated_hours=0.5,
                                source="mistral",
                                unit=unit_name,
                                chapter=chapter_name
                            ))
            
            # Fallback: Handle flat topics format
            if not topics and data.get("topics"):
                for topic_data in data.get("topics", []):
                    topic_name = topic_data.get("topic_name", topic_data.get("name", ""))
                    if not topic_name:
                        continue
                    
                    subtopics = topic_data.get("subtopics", topic_data.get("micro_topics", []))
                    if isinstance(subtopics, list):
                        subtopics = [s if isinstance(s, str) else s.get("subtopic_name", s.get("name", "")) 
                                   for s in subtopics if s]
                    else:
                        subtopics = []
                    
                    # FLATTEN each subtopic
                    if subtopics:
                        for st in subtopics:
                            if st:
                                topics.append(ExtractedTopic(
                                    name=st,
                                    description=f"Part of {topic_name}",
                                    subtopics=[],
                                    difficulty="medium",
                                    estimated_hours=0.25,
                                    source="mistral"
                                ))
                    else:
                        topics.append(ExtractedTopic(
                            name=topic_name,
                            description=None,
                            subtopics=[],
                            difficulty="medium",
                            estimated_hours=0.5,
                            source="mistral"
                        ))
            
            if topics:
                print(f"[TOPIC-EXTRACTOR] ✓ Mistral extracted {len(topics)} atomic topics")
            
            return topics
            
        except json.JSONDecodeError as e:
            print(f"[TOPIC-EXTRACTOR] Mistral JSON parse error: {e}")
            return []
        except Exception as e:
            print(f"[TOPIC-EXTRACTOR] Mistral extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
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
        """Use Gemini for topic extraction with model fallbacks and retry logic.
        
        Each Gemini model has its own 20 requests/day quota on FREE tier.
        We try multiple models to maximize available quota.
        """
        try:
            from google import genai
            from google.genai import types
            from google.genai.errors import ClientError
            import time
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.info("[TOPIC-EXTRACTOR] No GEMINI_API_KEY, skipping Gemini")
                return []
            
            client = genai.Client(api_key=api_key)
            
            prompt = SYLLABUS_EXTRACTION_PROMPT.format(
                content=text[:20000],  # Gemini can handle more context
                subject_name=subject_name or "General"
            )
            
            # Try multiple models - each has separate FREE tier quota (20 req/day each)
            models_to_try = [
                "gemini-2.5-flash",
                "gemini-2.0-flash", 
                "gemini-1.5-flash",
            ]
            
            text_response = None
            
            for model_name in models_to_try:
                print(f"[TOPIC-EXTRACTOR] Trying {model_name}...")
                
                # Retry up to 2 times with backoff for rate limits
                for attempt in range(2):
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.2,
                                max_output_tokens=8000,
                            )
                        )
                        
                        # Success! Get the text
                        if response and response.text:
                            text_response = response.text.strip()
                            print(f"[TOPIC-EXTRACTOR] ✓ {model_name} returned {len(text_response)} chars")
                            break
                        else:
                            print(f"[TOPIC-EXTRACTOR] {model_name} returned empty response")
                            break  # Empty response, try next model
                            
                    except ClientError as e:
                        error_str = str(e)
                        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                            # Rate limited on this model
                            if attempt == 0:
                                # Try waiting if API suggests a retry delay
                                import re
                                delay_match = re.search(r'retry in (\d+(?:\.\d+)?)', error_str, re.IGNORECASE)
                                if delay_match:
                                    delay = min(float(delay_match.group(1)), 15)  # Cap at 15 seconds
                                    print(f"[TOPIC-EXTRACTOR] Rate limited on {model_name}, waiting {delay:.1f}s...")
                                    time.sleep(delay)
                                    continue  # Retry same model
                            
                            print(f"[TOPIC-EXTRACTOR] {model_name} rate limited, trying next model...")
                            break  # Try next model
                        else:
                            print(f"[TOPIC-EXTRACTOR] {model_name} error: {e}")
                            break  # Other error, try next model
                    except Exception as e:
                        print(f"[TOPIC-EXTRACTOR] {model_name} unexpected error: {e}")
                        break  # Try next model
                
                if text_response:
                    break  # Success, stop trying models
            
            if not text_response:
                print("[TOPIC-EXTRACTOR] All Gemini models failed or rate limited")
                return []
            
            # Clean JSON from markdown code blocks
            if "```" in text_response:
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text_response)
                if match:
                    text_response = match.group(1)
            
            result = json.loads(text_response)
            
            # Parse hierarchical format and FLATTEN to individual micro-topics (same as Groq)
            topics = []
            
            # Handle hierarchical format: units -> chapters -> topics -> micro_topics
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
                                    name=micro_topic,
                                    description=f"{chapter_name} → {topic_name}",
                                    subtopics=[],
                                    difficulty=difficulty,
                                    estimated_hours=0.25,
                                    source="gemini",
                                    unit=unit_name,
                                    chapter=chapter_name
                                ))
                        else:
                            topics.append(ExtractedTopic(
                                name=topic_name,
                                description=f"Part of {chapter_name}" if chapter_name else None,
                                subtopics=[],
                                difficulty=difficulty,
                                estimated_hours=0.5,
                                source="gemini",
                                unit=unit_name,
                                chapter=chapter_name
                            ))
            
            # Fallback: Handle flat topics format
            if not topics and result.get("topics"):
                for topic_data in result.get("topics", []):
                    topic_name = topic_data.get("topic_name", topic_data.get("name", ""))
                    if not topic_name:
                        continue
                    
                    subtopics = topic_data.get("subtopics", topic_data.get("micro_topics", []))
                    if isinstance(subtopics, list):
                        subtopics = [s if isinstance(s, str) else s.get("subtopic_name", s.get("name", "")) 
                                   for s in subtopics if s]
                    else:
                        subtopics = []
                    
                    # FLATTEN each subtopic
                    if subtopics:
                        for st in subtopics:
                            if st:
                                topics.append(ExtractedTopic(
                                    name=st,
                                    description=f"Part of {topic_name}",
                                    subtopics=[],
                                    difficulty=topic_data.get("difficulty", "medium"),
                                    source="gemini"
                                ))
                    else:
                        topics.append(ExtractedTopic(
                            name=topic_name,
                            subtopics=[],
                            difficulty=topic_data.get("difficulty", "medium"),
                            source="gemini"
                        ))
            
            print(f"[TOPIC-EXTRACTOR] ✓ Gemini: {len(topics)} atomic topics extracted")
            return topics
            
        except json.JSONDecodeError as e:
            logger.warning(f"Gemini JSON parse error: {e}")
            return []
        except Exception as e:
            logger.warning(f"Gemini extraction failed: {e}")
            import traceback
            traceback.print_exc()
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
