"""
Syllabus Extractor Service

Processes syllabus PDFs to:
1. Store chunks in 'syllabus_content' Qdrant collection
2. Extract topics/lessons using LLM
3. Populate curriculum models (Subject, Topic, Subtopic)

Pipeline:
  PDF → Text Extraction → Chunking → Qdrant Storage
  Chunks → LLM Analysis → Topic Hierarchy → Database
"""
import os
import re
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Collection name for syllabus content
SYLLABUS_COLLECTION = "syllabus_content"


@dataclass
class ExtractedTopic:
    """Extracted topic structure from syllabus"""
    name: str
    description: Optional[str]
    subtopics: List[str]
    difficulty: str = "medium"
    estimated_hours: float = 2.0
    keywords: List[str] = None
    page_numbers: List[int] = None
    
    def __post_init__(self):
        # Truncate name to 200 chars (DB limit) to prevent StringDataRightTruncation errors
        if self.name and len(self.name) > 200:
            self.name = self.name[:197] + "..."


@dataclass
class ExtractionResult:
    """Result of syllabus extraction"""
    success: bool
    syllabus_id: str
    chunks_stored: int
    topics_extracted: int
    lessons_created: int
    processing_time_ms: int
    error: Optional[str] = None


class SyllabusExtractor:
    """
    Extract and process syllabus documents.
    
    Features:
    - PDF text extraction with chapter detection
    - Semantic chunking for vector storage
    - LLM-based topic extraction
    - Curriculum model population
    """
    
    def __init__(self):
        self.pdf_processor = None
        self.embedding_model = None
        self.qdrant_client = None
        self.llm = None
        self._initialized = False
        
    def _ensure_services(self):
        """Lazy load all required services"""
        if self._initialized:
            return
            
        try:
            # Import PDF processor
            from .pdf_processor import PDFProcessor
            self.pdf_processor = PDFProcessor()
            
            # Import embedding model
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            print(f"[SYLLABUS] Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            
            # Import Qdrant client
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            
            qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
            self._ensure_collection()
            
            # Import LLM
            from .llm_provider import get_llm
            self.llm = get_llm()
            
            self._initialized = True
            logger.info("[SYLLABUS] Services initialized successfully")
             
        except Exception as e:
            logger.error(f"[SYLLABUS] Failed to initialize services: {e}")
            raise
    
    def _ensure_collection(self):
        """Create Qdrant collection for syllabus content if it doesn't exist"""
        from qdrant_client.models import Distance, VectorParams
        
        collections = self.qdrant_client.get_collections().collections
        exists = any(c.name == SYLLABUS_COLLECTION for c in collections)
        
        if not exists:
            self.qdrant_client.create_collection(
                collection_name=SYLLABUS_COLLECTION,
                vectors_config=VectorParams(
                    size=384,  # MiniLM embedding size
                    distance=Distance.COSINE
                )
            )
            logger.info(f"[SYLLABUS] Created Qdrant collection: {SYLLABUS_COLLECTION}")
        else:
            logger.info(f"[SYLLABUS] Using existing collection: {SYLLABUS_COLLECTION}")
    
    async def process_syllabus(
        self,
        syllabus_id: str,
        pdf_path: str,
        classroom_id: str,
        subject_name: str,
        title: Optional[str] = None
    ) -> ExtractionResult:
        """
        Process a syllabus PDF and extract topics.
        
        Args:
            syllabus_id: Database ID of the syllabus record
            pdf_path: Path to the PDF file
            classroom_id: Classroom this syllabus belongs to
            subject_name: Subject name (e.g., "Physics")
            title: Optional syllabus title
            
        Returns:
            ExtractionResult with processing details
        """
        start_time = datetime.utcnow()
        
        try:
            self._ensure_services()
            
            print(f"\n{'='*60}")
            print(f"[SYLLABUS] Processing: {pdf_path}")
            print(f"[SYLLABUS] Subject: {subject_name}, Classroom: {classroom_id}")
            print(f"{'='*60}")
            
            # Step 1: Extract text and detect chapters
            print("[SYLLABUS] Step 1: Extracting text from PDF...")
            extraction_result = self.pdf_processor.extract_from_file(pdf_path)
            
            if not extraction_result.success:
                raise Exception(extraction_result.error or "Failed to extract PDF")
            
            # Get full text and create chunks from pages
            full_text = self.pdf_processor.get_full_text(extraction_result)
            pages_text = self.pdf_processor.get_pages_text(extraction_result)
            
            # First pass: Detect all chapter/topic headings
            all_headings = []
            heading_keywords = ["chapter", "unit", "module", "topic", "section", "part", "lesson"]
            
            for page_data in pages_text:
                page_text = page_data.get("text", "")
                page_num = page_data.get("page_number", 0)
                
                for line in page_text.split("\n"):
                    line_clean = line.strip()
                    if not line_clean or len(line_clean) < 3 or len(line_clean) > 150:
                        continue
                    
                    # Check if it looks like a heading
                    is_heading = False
                    
                    # Method 1: Contains heading keywords
                    if any(kw in line_clean.lower() for kw in heading_keywords):
                        is_heading = True
                    
                    # Method 2: Numbered like "1. Topic" or "1.1 Topic" or "I. Topic"
                    elif re.match(r'^[\dIVXivx]+[\.\)]\s*\S', line_clean):
                        is_heading = True
                    
                    # Method 3: All caps and short (likely a title)
                    elif line_clean.isupper() and 5 < len(line_clean) < 60:
                        is_heading = True
                    
                    if is_heading:
                        # Clean the heading name
                        name = re.sub(r'^[\dIVXivx]+[\.\):\s]+', '', line_clean)  # Remove numbering
                        name = name.strip()
                        if name and len(name) > 3:
                            all_headings.append({
                                "name": name,
                                "page": page_num,
                                "original": line_clean
                            })
            
            # Deduplicate headings by name
            seen_names = set()
            chapters = []
            for h in all_headings:
                name_lower = h["name"].lower()
                if name_lower not in seen_names:
                    seen_names.add(name_lower)
                    chapters.append(h)
            
            print(f"[SYLLABUS] Found {len(chapters)} unique headings/chapters")
            
            # Create chunks with proper topic assignment
            all_chunks = []
            current_chapter = "Introduction"
            
            for page_data in pages_text:
                page_text = page_data.get("text", "")
                page_num = page_data.get("page_number", 0)
                
                # Check if this page has a new chapter heading
                for ch in chapters:
                    if ch["page"] == page_num:
                        current_chapter = ch["name"]
                        break
                
                # Chunk by paragraphs
                paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
                for para in paragraphs:
                    if len(para) > 50:
                        all_chunks.append({
                            "content": para,
                            "page": page_num,
                            "chapter": current_chapter
                        })
            
            print(f"[SYLLABUS] ✓ Extracted {len(full_text)} chars, {len(chapters)} chapters, {len(all_chunks)} chunks")
            
            # Step 2: Store chunks in Qdrant
            print("[SYLLABUS] Step 2: Storing chunks in Qdrant...")
            chunks_stored = await self._store_chunks_in_qdrant(
                chunks=all_chunks,
                syllabus_id=syllabus_id,
                classroom_id=classroom_id,
                subject_name=subject_name
            )
            print(f"[SYLLABUS] ✓ Stored {chunks_stored} chunks in '{SYLLABUS_COLLECTION}'")
            
            # Step 3: Extract topics using robust multi-method extractor
            print("[SYLLABUS] Step 3: Extracting topics with robust extractor...")
            from .topic_extractor import get_topic_extractor
            
            topic_extractor = get_topic_extractor()
            extracted_topics = topic_extractor.extract_topics(
                pdf_path=pdf_path,
                subject_name=subject_name,
                prefer_ai=True  # Use Gemini if available
            )
            
            # Convert to ExtractedTopic format
            topics = []
            for et in extracted_topics:
                topics.append(ExtractedTopic(
                    name=et.name,
                    description=et.description,
                    subtopics=et.subtopics,
                    difficulty=et.difficulty,
                    estimated_hours=et.estimated_hours
                ))
            
            print(f"[SYLLABUS] ✓ Extracted {len(topics)} topics (source: {extracted_topics[0].source if extracted_topics else 'none'})")
            
            # Step 4: Populate curriculum database
            print("[SYLLABUS] Step 4: Populating curriculum models...")
            lessons_created = await self._populate_curriculum(
                topics=topics,
                syllabus_id=syllabus_id,
                classroom_id=classroom_id,
                subject_name=subject_name
            )
            print(f"[SYLLABUS] ✓ Created {lessons_created} topic records")
            
            elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            print(f"\n{'='*60}")
            print(f"[SYLLABUS] ✅ COMPLETE in {elapsed_ms}ms")
            print(f"[SYLLABUS]   Chunks: {chunks_stored}, Topics: {len(topics)}")
            print(f"{'='*60}\n")
            
            return ExtractionResult(
                success=True,
                syllabus_id=syllabus_id,
                chunks_stored=chunks_stored,
                topics_extracted=len(topics),
                lessons_created=lessons_created,
                processing_time_ms=elapsed_ms
            )
            
        except Exception as e:
            logger.error(f"[SYLLABUS] Processing failed: {e}")
            elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return ExtractionResult(
                success=False,
                syllabus_id=syllabus_id,
                chunks_stored=0,
                topics_extracted=0,
                lessons_created=0,
                processing_time_ms=elapsed_ms,
                error=str(e)
            )
    
    async def _store_chunks_in_qdrant(
        self,
        chunks: List[Dict],
        syllabus_id: str,
        classroom_id: str,
        subject_name: str
    ) -> int:
        """Store text chunks in Qdrant collection"""
        from qdrant_client.models import PointStruct
        import uuid
        
        if not chunks:
            return 0
        
        points = []
        
        for i, chunk in enumerate(chunks):
            # Get chunk text
            if isinstance(chunk, dict):
                text = chunk.get("text", chunk.get("content", ""))
                chapter = chunk.get("chapter", "")
                page = chunk.get("page_number", 0)
            else:
                text = str(chunk)
                chapter = ""
                page = 0
            
            if not text or len(text) < 20:
                continue
            
            # Generate embedding
            embedding = self.embedding_model.encode(text).tolist()
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "syllabus_id": syllabus_id,
                    "classroom_id": classroom_id,
                    "subject": subject_name,
                    "chunk_text": text[:2000],  # Limit payload size
                    "chapter": chapter,
                    "page_number": page,
                    "chunk_index": i,
                    "source_type": "syllabus",
                    "indexed_at": datetime.utcnow().isoformat()
                }
            )
            points.append(point)
        
        # Batch upsert to Qdrant
        if points:
            self.qdrant_client.upsert(
                collection_name=SYLLABUS_COLLECTION,
                points=points
            )
        
        return len(points)
    
    async def _extract_topics_with_llm(
        self,
        full_text: str,
        chapters: List[Dict],
        subject_name: str
    ) -> List[ExtractedTopic]:
        """Use LLM to extract topics from syllabus content"""
        
        # First try using Gemini API directly (more reliable)
        try:
            topics = await self._extract_with_gemini(full_text, chapters, subject_name)
            if topics:
                return topics
        except Exception as e:
            logger.warning(f"[SYLLABUS] Gemini extraction failed: {e}")
        
        # Fallback: Try regular LLM
        try:
            topics = self._extract_with_default_llm(full_text, chapters, subject_name)
            if topics:
                return topics
        except Exception as e:
            logger.warning(f"[SYLLABUS] Default LLM failed: {e}")
        
        # Final fallback: Extract from detected chapters
        logger.info("[SYLLABUS] Using chapter-based fallback for topics")
        return self._extract_from_chapters(chapters, full_text)
    
    async def _extract_with_gemini(
        self,
        full_text: str,
        chapters: List[Dict],
        subject_name: str
    ) -> List[ExtractedTopic]:
        """Use Gemini API for topic extraction"""
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare content
        chapter_list = "\n".join([f"- {ch.get('name', ch.get('title', ''))}" for ch in chapters if ch]) if chapters else ""
        content = f"Chapters:\n{chapter_list}\n\nContent:\n{full_text[:4000]}"
        
        prompt = f"""Extract the main topics from this {subject_name} syllabus.

{content}

Return a JSON array of topics:
[{{"name": "Topic Name", "description": "Brief description", "subtopics": ["sub1"], "difficulty": "easy/medium/hard", "estimated_hours": 2}}]

Return ONLY valid JSON, no markdown or other text."""

        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean JSON
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        
        topics_data = json.loads(text)
        
        topics = []
        for t in topics_data:
            if isinstance(t, dict) and t.get("name"):
                topics.append(ExtractedTopic(
                    name=t["name"],
                    description=t.get("description"),
                    subtopics=t.get("subtopics", []),
                    difficulty=t.get("difficulty", "medium"),
                    estimated_hours=float(t.get("estimated_hours", 2)),
                    keywords=t.get("keywords", [])
                ))
        
        return topics
    
    def _extract_with_default_llm(
        self,
        full_text: str,
        chapters: List[Dict],
        subject_name: str
    ) -> List[ExtractedTopic]:
        """Try extracting with default LLM"""
        chapter_list = "\n".join([f"- {ch.get('name', ch.get('title', ''))}" for ch in chapters if ch]) if chapters else ""
        content = f"Chapters:\n{chapter_list}\n\nContent:\n{full_text[:3000]}"
        
        prompt = f"""Extract topics from this {subject_name} syllabus.
{content}

Return JSON array: [{{"name": "Topic", "description": "...", "difficulty": "medium", "estimated_hours": 2}}]
JSON only:"""

        response = self.llm.invoke(prompt)
        text = response.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        
        topics_data = json.loads(text)
        return [
            ExtractedTopic(
                name=t["name"],
                description=t.get("description"),
                subtopics=t.get("subtopics", []),
                difficulty=t.get("difficulty", "medium"),
                estimated_hours=float(t.get("estimated_hours", 2))
            )
            for t in topics_data if isinstance(t, dict) and t.get("name")
        ]
    
    def _extract_from_chapters(
        self,
        chapters: List[Dict],
        full_text: str
    ) -> List[ExtractedTopic]:
        """Fallback: Create topics from detected chapter headings"""
        topics = []
        seen = set()
        
        # From chapters list
        for ch in chapters:
            name = ch.get("name", ch.get("title", ""))
            if name and name not in seen and len(name) > 3:
                seen.add(name)
                topics.append(ExtractedTopic(
                    name=name,
                    description=None,
                    subtopics=[],
                    difficulty="medium",
                    estimated_hours=2.0
                ))
        
        # If no chapters found, extract from text headings
        if not topics:
            lines = full_text.split("\n")
            for line in lines:
                line = line.strip()
                # Detect headings: short lines, title case or all caps
                if 5 < len(line) < 80 and (line.isupper() or line.istitle()):
                    if any(kw in line.lower() for kw in ["chapter", "unit", "module", "topic", "lesson", "section"]):
                        name = line.strip("0123456789.:- ")
                        if name and name not in seen:
                            seen.add(name)
                            topics.append(ExtractedTopic(
                                name=name,
                                description=None,
                                subtopics=[],
                                difficulty="medium",
                                estimated_hours=2.0
                            ))
        
        return topics
    
    async def _populate_curriculum(
        self,
        topics: List[ExtractedTopic],
        syllabus_id: str,
        classroom_id: str,
        subject_name: str
    ) -> int:
        """
        Populate curriculum database models with extracted topics.
        
        Makes HTTP calls to Core Service API to create:
        1. Subject (if doesn't exist)
        2. Topics linked to subject
        3. Subtopics linked to topics
        4. Links syllabus to subject
        """
        import httpx
        
        created_count = 0
        core_service_url = os.getenv("CORE_SERVICE_URL", "http://localhost:9000")
        
        logger.info(f"[SYLLABUS] Populating curriculum for subject: {subject_name}")
        logger.info(f"[SYLLABUS] Core service URL: {core_service_url}")
        logger.info(f"[SYLLABUS] Topics to create: {len(topics)}")
        
        try:
            # Use verify=False for local development with self-signed certificates
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                # Step 1: Create or get Subject
                subject_id = None
                
                # First try to find existing subject
                try:
                    resp = await client.get(
                        f"{core_service_url}/api/topics/subjects",
                        headers={"Content-Type": "application/json", "X-Service-Key": "internal-ai-service"}
                    )
                    if resp.status_code == 200:
                        subjects_data = resp.json()
                        for s in subjects_data.get("subjects", []):
                            if s.get("name", "").lower() == subject_name.lower():
                                subject_id = s.get("id")
                                logger.info(f"[SYLLABUS] Found existing subject: {subject_name} (ID: {subject_id})")
                                break
                except Exception as e:
                    logger.warning(f"[SYLLABUS] Failed to fetch subjects: {e}")
                
                # Create subject if not found
                if not subject_id:
                    logger.info(f"[SYLLABUS] Creating new subject: {subject_name}")
                    try:
                        resp = await client.post(
                            f"{core_service_url}/api/topics/subjects",
                            json={
                                "name": subject_name,
                                "description": f"Subject extracted from syllabus for classroom {classroom_id}",
                                "icon": self._get_subject_icon(subject_name),
                                "color": self._get_subject_color(subject_name),
                                "is_active": True
                            },
                            headers={"Content-Type": "application/json", "X-Service-Key": "internal-ai-service"}
                        )
                        if resp.status_code in [200, 201]:
                            subject_data = resp.json().get("subject", {})
                            subject_id = subject_data.get("id")
                            logger.info(f"[SYLLABUS] ✓ Created subject: {subject_name} (ID: {subject_id})")
                        elif resp.status_code == 400 and "already exists" in resp.text.lower():
                            # Subject already exists, fetch it from the list again
                            logger.info(f"[SYLLABUS] Subject already exists, fetching existing ID...")
                            subjects_resp = await client.get(
                                f"{core_service_url}/api/topics/subjects",
                                headers={"X-Service-Key": "internal-ai-service"}
                            )
                            if subjects_resp.status_code == 200:
                                subjects_list = subjects_resp.json().get("subjects", [])
                                for subj in subjects_list:
                                    if subj.get("name", "").lower() == subject_name.lower():
                                        subject_id = subj.get("id")
                                        logger.info(f"[SYLLABUS] ✓ Found existing subject: {subject_name} (ID: {subject_id})")
                                        break
                        else:
                            logger.error(f"[SYLLABUS] Failed to create subject: {resp.status_code} - {resp.text}")
                    except Exception as e:
                        logger.error(f"[SYLLABUS] Error creating subject: {e}")
                
                if not subject_id:
                    logger.error("[SYLLABUS] Cannot proceed without subject ID")
                    return 0
                
                # Step 2: Update syllabus to link to subject
                try:
                    resp = await client.put(
                        f"{core_service_url}/api/topics/syllabus/{syllabus_id}",
                        json={"subject_id": subject_id},
                        headers={"Content-Type": "application/json", "X-Service-Key": "internal-ai-service"}
                    )
                    if resp.status_code in [200, 201]:
                        logger.info(f"[SYLLABUS] ✓ Linked syllabus to subject")
                    else:
                        logger.warning(f"[SYLLABUS] Could not link syllabus to subject: {resp.status_code}")
                except Exception as e:
                    logger.warning(f"[SYLLABUS] Failed to link syllabus: {e}")
                
                # Step 3: Create topics and subtopics
                for i, topic in enumerate(topics):
                    try:
                        logger.info(f"[SYLLABUS] Creating topic {i+1}/{len(topics)}: {topic.name}")
                        
                        # Create topic (POST /api/topics/ with subject_id)
                        topic_resp = await client.post(
                            f"{core_service_url}/api/topics/",
                            json={
                                "name": topic.name,
                                "description": topic.description or f"Topic from {subject_name} syllabus",
                                "subject_id": subject_id,
                                "difficulty": topic.difficulty,
                                "estimated_hours": topic.estimated_hours,
                                "order": i + 1,
                                "is_active": True
                            },
                            headers={"Content-Type": "application/json", "X-Service-Key": "internal-ai-service"}
                        )
                        
                        if topic_resp.status_code in [200, 201]:
                            topic_data = topic_resp.json().get("topic", {})
                            topic_id = topic_data.get("id")
                            logger.info(f"[SYLLABUS] ✓ Created topic: {topic.name} (ID: {topic_id})")
                            created_count += 1
                            
                            # Create subtopics
                            for j, subtopic_name in enumerate(topic.subtopics):
                                try:
                                    subtopic_resp = await client.post(
                                        f"{core_service_url}/api/topics/{topic_id}/subtopics",
                                        json={
                                            "name": subtopic_name,
                                            "description": f"Subtopic of {topic.name}",
                                            "order": j + 1,
                                            "is_active": True
                                        },
                                        headers={"Content-Type": "application/json", "X-Service-Key": "internal-ai-service"}
                                    )
                                    if subtopic_resp.status_code in [200, 201]:
                                        logger.info(f"[SYLLABUS]   ✓ Created subtopic: {subtopic_name}")
                                    else:
                                        logger.warning(f"[SYLLABUS]   ✗ Failed to create subtopic: {subtopic_name}")
                                except Exception as e:
                                    logger.warning(f"[SYLLABUS]   ✗ Error creating subtopic {subtopic_name}: {e}")
                        else:
                            logger.error(f"[SYLLABUS] ✗ Failed to create topic: {topic.name} - {topic_resp.status_code}")
                            
                    except Exception as e:
                        logger.error(f"[SYLLABUS] Error creating topic {topic.name}: {e}")
                
                logger.info(f"[SYLLABUS] ✅ Curriculum population complete: {created_count} topics created")
                
        except Exception as e:
            logger.error(f"[SYLLABUS] Failed to populate curriculum: {e}")
            import traceback
            traceback.print_exc()
        
        return created_count
    
    def _get_subject_icon(self, subject_name: str) -> str:
        """Get emoji icon for subject"""
        icons = {
            "physics": "⚡",
            "chemistry": "🧪",
            "biology": "🧬",
            "mathematics": "📐",
            "math": "📐",
            "english": "📚",
            "history": "🏛️",
            "geography": "🌍",
            "computer": "💻",
            "science": "🔬"
        }
        return icons.get(subject_name.lower().split()[0], "📖")
    
    def _get_subject_color(self, subject_name: str) -> str:
        """Get color for subject"""
        colors = {
            "physics": "#3B82F6",
            "chemistry": "#10B981",
            "biology": "#8B5CF6",
            "mathematics": "#F59E0B",
            "math": "#F59E0B",
            "english": "#EC4899",
            "history": "#6366F1",
            "geography": "#14B8A6",
            "computer": "#6B7280",
            "science": "#EF4444"
        }
        return colors.get(subject_name.lower().split()[0], "#6B7280")
    
    def search_syllabus_content(
        self,
        query: str,
        classroom_id: Optional[str] = None,
        subject: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search syllabus content in Qdrant.
        
        Args:
            query: Search query
            classroom_id: Optional classroom filter
            subject: Optional subject filter
            top_k: Number of results
            
        Returns:
            List of matching chunks with metadata
        """
        self._ensure_services()
        
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Build filter
        must_conditions = []
        
        if classroom_id:
            must_conditions.append(FieldCondition(
                key="classroom_id",
                match=MatchValue(value=classroom_id)
            ))
        
        if subject:
            must_conditions.append(FieldCondition(
                key="subject",
                match=MatchValue(value=subject)
            ))
        
        query_filter = Filter(must=must_conditions) if must_conditions else None
        
        # Search using query_points (newer Qdrant API)
        try:
            results = self.qdrant_client.query_points(
                collection_name=SYLLABUS_COLLECTION,
                query=query_embedding,
                query_filter=query_filter,
                limit=top_k
            )
            points = results.points if hasattr(results, 'points') else results
        except AttributeError:
            # Fallback for older Qdrant client
            results = self.qdrant_client.search(
                collection_name=SYLLABUS_COLLECTION,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=top_k
            )
            points = results
        
        # Format results
        return [
            {
                "id": str(r.id),
                "score": getattr(r, 'score', 0),
                "text": r.payload.get("chunk_text", ""),
                "chapter": r.payload.get("chapter", ""),
                "content": r.payload.get("chunk_text", ""),
                "subject": r.payload.get("subject", ""),
                "syllabus_id": r.payload.get("syllabus_id", ""),
                "page_number": r.payload.get("page_number", 0)
            }
            for r in points
        ]


# Singleton instance
_extractor_instance: Optional[SyllabusExtractor] = None


def get_syllabus_extractor() -> SyllabusExtractor:
    """Get or create the syllabus extractor singleton."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = SyllabusExtractor()
    return _extractor_instance
