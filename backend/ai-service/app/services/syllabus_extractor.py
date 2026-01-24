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
            processed = self.pdf_processor.process_pdf(pdf_path)
            
            full_text = processed.get("full_text", "")
            chapters = processed.get("chapters", [])
            all_chunks = processed.get("all_chunks", [])
            
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
            
            # Step 3: Extract topics using LLM
            print("[SYLLABUS] Step 3: Extracting topics using LLM...")
            topics = await self._extract_topics_with_llm(
                full_text=full_text,
                chapters=chapters,
                subject_name=subject_name
            )
            print(f"[SYLLABUS] ✓ Extracted {len(topics)} topics")
            
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
        
        # Prepare content for LLM - use chapters if available, else text sample
        if chapters:
            chapter_list = "\n".join([f"- {ch.get('title', 'Untitled')}" for ch in chapters])
            content_for_llm = f"Chapters detected:\n{chapter_list}\n\nSample content:\n{full_text[:3000]}"
        else:
            content_for_llm = full_text[:4000]
        
        prompt = f"""Analyze this syllabus for {subject_name} and extract the main topics/lessons.

SYLLABUS CONTENT:
{content_for_llm}

Return a JSON array of topics. Each topic should have:
- name: The topic/lesson name
- description: Brief description (1-2 sentences)
- subtopics: Array of subtopic strings
- difficulty: "easy", "medium", or "hard"
- estimated_hours: Estimated study hours (number)
- keywords: Array of key terms

Example format:
[
  {{
    "name": "Units & Measurements",
    "description": "Physical quantities, units, and measurement techniques",
    "subtopics": ["SI Units", "Dimensional Analysis", "Errors in Measurement"],
    "difficulty": "easy",
    "estimated_hours": 2,
    "keywords": ["units", "measurement", "dimensions"]
  }}
]

Return ONLY the JSON array, no other text."""

        try:
            response = self.llm.invoke(prompt)
            
            # Parse JSON from response
            text = response.strip()
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            
            topics_data = json.loads(text)
            
            # Convert to ExtractedTopic objects
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
            
        except json.JSONDecodeError as e:
            logger.warning(f"[SYLLABUS] Failed to parse LLM response as JSON: {e}")
            
            # Fallback: Extract topics from chapter titles
            topics = []
            for ch in chapters:
                title = ch.get("title", "")
                if title and len(title) > 3:
                    topics.append(ExtractedTopic(
                        name=title,
                        description=None,
                        subtopics=[],
                        difficulty="medium",
                        estimated_hours=2.0
                    ))
            
            return topics
        
        except Exception as e:
            logger.error(f"[SYLLABUS] LLM extraction error: {e}")
            return []
    
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
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: Create or get Subject
                subject_id = None
                
                # First try to find existing subject
                try:
                    resp = await client.get(
                        f"{core_service_url}/api/topics/subjects",
                        headers={"Content-Type": "application/json"}
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
                            headers={"Content-Type": "application/json"}
                        )
                        if resp.status_code in [200, 201]:
                            subject_data = resp.json().get("subject", {})
                            subject_id = subject_data.get("id")
                            logger.info(f"[SYLLABUS] ✓ Created subject: {subject_name} (ID: {subject_id})")
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
                        headers={"Content-Type": "application/json"}
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
                        
                        # Create topic
                        topic_resp = await client.post(
                            f"{core_service_url}/api/topics/topics",
                            json={
                                "name": topic.name,
                                "description": topic.description or f"Topic from {subject_name} syllabus",
                                "subject_id": subject_id,
                                "difficulty": topic.difficulty,
                                "estimated_hours": topic.estimated_hours,
                                "order": i + 1,
                                "is_active": True
                            },
                            headers={"Content-Type": "application/json"}
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
                                        f"{core_service_url}/api/topics/subtopics",
                                        json={
                                            "name": subtopic_name,
                                            "description": f"Subtopic of {topic.name}",
                                            "topic_id": topic_id,
                                            "order": j + 1,
                                            "is_active": True
                                        },
                                        headers={"Content-Type": "application/json"}
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
        
        # Search
        results = self.qdrant_client.search(
            collection_name=SYLLABUS_COLLECTION,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=top_k
        )
        
        # Format results
        return [
            {
                "id": str(r.id),
                "score": r.score,
                "text": r.payload.get("chunk_text", ""),
                "chapter": r.payload.get("chapter", ""),
                "subject": r.payload.get("subject", ""),
                "syllabus_id": r.payload.get("syllabus_id", ""),
                "page_number": r.payload.get("page_number", 0)
            }
            for r in results
        ]


# Singleton instance
_extractor_instance: Optional[SyllabusExtractor] = None


def get_syllabus_extractor() -> SyllabusExtractor:
    """Get or create the syllabus extractor singleton."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = SyllabusExtractor()
    return _extractor_instance
