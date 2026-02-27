# Page 87: Syllabus & Topic Extraction

> 820-line syllabus processing pipeline: PDF extraction → semantic chunking → Qdrant storage → LLM topic extraction → curriculum model population.

---

## 87.1 Pipeline Architecture

```mermaid\nflowchart TB\n    PDF[\"📄 Syllabus PDF\"] --> SE\n\n    subgraph SE[\"SyllabusExtractor — 820 lines, 33KB\"]\n        direction TB\n        PS[\"process_syllabus()\"]\n        SC[\"_store_chunks()\"] -->|\"Vectors\"| QD[\"Qdrant<br/>'syllabus_content'\"]\n        ET[\"_extract_topics()\"] -->|\"API call\"| LLM[\"LLM<br/>Gemini / Groq\"]\n        PC[\"_populate_curriculum()\"] -->|\"HTTP\"| CS[\"Core Service API\"]\n        PS --> SC --> ET --> PC\n    end\n\n    SE --> TE[\"topic_extractor.py<br/>36KB — Deep topic analysis<br/>Prerequisites, Bloom's taxonomy\"]\n    SE --> SHE[\"syllabus_hierarchy_extractor.py<br/>16KB — Nested hierarchy builder<br/>Subject→Unit→Chapter→Topic\"]\n\n    style PDF fill:#3b82f6,color:#fff\n    style QD fill:#ef4444,color:#fff\n    style LLM fill:#f59e0b,color:#000\n    style CS fill:#10b981,color:#fff\n```

### Source Files

| File | Lines | Size | Role |
|------|-------|------|------|
| `services/syllabus_extractor.py` | 820 | 33KB | Main pipeline |
| `services/topic_extractor.py` | — | 37KB | Deep topic analysis |
| `services/syllabus_hierarchy_extractor.py` | — | 16KB | Nested hierarchy |
| `api/routes/syllabus.py` | — | 17KB | Syllabus API |
| `api/routes/classroom_syllabus.py` | — | 15KB | Classroom-specific APIs |
| `core-service/app/routes/topics.py` | — | 36KB | Topic CRUD |
| `api/routes/topic_scores.py` | — | 11KB | Topic scoring APIs |

---

## 87.2 SyllabusExtractor — Main Pipeline

### Data Models

```python
@dataclass
class ExtractedTopic:
    name: str
    description: Optional[str]
    subtopics: List[str]
    difficulty: str = "medium"
    estimated_hours: float = 2.0
    keywords: List[str] = None
    page_numbers: List[int] = None

@dataclass
class ExtractionResult:
    success: bool
    syllabus_id: str
    chunks_stored: int
    topics_extracted: int
    lessons_created: int
    processing_time_ms: int
    error: Optional[str] = None
```

### Main Method

```python
class SyllabusExtractor:
    async def process_syllabus(
        self,
        syllabus_id: str,
        pdf_path: str,
        classroom_id: str,
        subject_name: str,
        title: Optional[str] = None
    ) -> ExtractionResult:
        """
        Full pipeline:
        1. Extract text from PDF (with chapter detection)
        2. Chunk text semantically
        3. Store chunks in Qdrant "syllabus_content" collection
        4. Extract topics using LLM
        5. Populate curriculum models via Core Service API
        """
```

---

## 87.3 Qdrant Storage

Chunks stored in `syllabus_content` collection with metadata:

```python
def _store_chunks_in_qdrant(self, chunks, syllabus_id, classroom_id, subject_name):
    # Vector: all-MiniLM-L6-v2 embedding (384-dim)
    # Payload: {
    #     syllabus_id, classroom_id, subject_name,
    #     chunk_index, page_number, chapter_title,
    #     text_preview (first 200 chars)
    # }
```

### Search

```python
def search_syllabus_content(self, query, classroom_id=None, subject=None, top_k=5):
    """Semantic search across syllabus chunks with filters"""
```

---

## 87.4 LLM Topic Extraction

Three fallback strategies:

| Priority | Method | Model |
|----------|--------|-------|
| 1 | `_extract_with_gemini()` | Google Gemini API |
| 2 | `_extract_with_default_llm()` | Groq `llama-3.3-70b` |
| 3 | `_extract_from_chapters()` | Regex-based chapter heading detection |

### LLM Prompt

```
Given this syllabus content for {subject_name}, extract a structured
topic hierarchy in JSON format:
[{
    "name": "Topic Name",
    "description": "Brief description",
    "subtopics": ["Subtopic 1", "Subtopic 2"],
    "difficulty": "easy|medium|hard",
    "estimated_hours": 2.0,
    "keywords": ["keyword1", "keyword2"]
}]
```

---

## 87.5 Curriculum Population

Makes HTTP calls to Core Service to create database records:

```python
def _populate_curriculum(self, topics, syllabus_id, classroom_id, subject_name):
    # Step 1: Create Subject (if not exists)
    POST /api/classrooms/{classroom_id}/subjects
    → {name, icon, color, syllabus_id}
    
    # Step 2: Create Topics linked to subject
    POST /api/classrooms/{classroom_id}/topics
    → {name, description, subject_id, difficulty, estimated_hours}
    
    # Step 3: Create Subtopics linked to topics
    POST /api/classrooms/{classroom_id}/topics/{topic_id}/subtopics
    → {name, description}
    
    # Step 4: Link syllabus to subject
    PUT /api/syllabi/{syllabus_id}
    → {subject_id}
```

### Subject Theming

Auto-assigns icons and colors based on subject name:
```python
def _get_subject_icon(self, subject_name):
    # "math" → "📐", "physics" → "⚛️", "chemistry" → "🧪"
    
def _get_subject_color(self, subject_name):
    # "math" → "#4F46E5", "physics" → "#7C3AED"
```

---

## 87.6 TopicExtractor Deep Analysis

### Source: `services/topic_extractor.py` (36KB — largest service file!)

Goes beyond basic extraction to provide:

- **Prerequisite mapping**: Which topics depend on others
- **Learning objective extraction**: What students should know after each topic
- **Bloom's taxonomy classification**: Remember, Understand, Apply, Analyze, Evaluate, Create
- **Cross-reference detection**: Links between topics in different subjects
- **Difficulty estimation**: Based on vocabulary complexity and concept density

---

## 87.7 Syllabus Hierarchy Extractor

### Source: `services/syllabus_hierarchy_extractor.py` (16KB)

Builds nested hierarchy: **Subject → Unit → Chapter → Topic → Subtopic**

```python
class SyllabusHierarchyExtractor:
    def extract_hierarchy(self, full_text, chapters):
        """
        Uses LLM to create nested structure:
        {
            "units": [{
                "name": "Unit 1: Mechanics",
                "chapters": [{
                    "name": "Newton's Laws",
                    "topics": [{
                        "name": "First Law",
                        "subtopics": ["Inertia", "Equilibrium"]
                    }]
                }]
            }]
        }
        """
```
