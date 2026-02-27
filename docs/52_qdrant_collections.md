# Page 52: Qdrant Vector Collections — Schema & Operations

---

## 52.1 Overview

ensureStudy uses **Qdrant** as its vector database, managing **6+ collections** for different embedding types: classroom materials, meeting transcripts, student notes, web resources, syllabus content, and general documents.

---

## 52.2 Collection Inventory

| Collection | Dimension | Distance | Source | Purpose |
|-----------|-----------|----------|--------|---------|
| `classroom_materials` | 768 | Cosine | Uploaded PDFs, PPTXs | RAG for tutor chat |
| `meeting_chunks` | 768 | Cosine | Transcribed meetings | Meeting Q&A |
| `student_notes` | 768 | Cosine | Personal notes | Note search |
| `web_resources` | 768 | Cosine | Crawled web content | Web resource search |
| `syllabus_content` | 768 | Cosine | Extracted syllabi | Curriculum planning |
| `documents` | 768 | Cosine | General documents | Document search |

All collections use the `all-mpnet-base-v2` embedding model (768 dimensions).

---

## 52.3 Collection Creation

### Source: `services/qdrant_service.py`

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
    
    def create_collection(self, name: str):
        self.client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE
            )
        )
```

---

## 52.4 Point Schema

Each vector point stores:

```python
PointStruct(
    id=str(uuid4()),
    vector=embedding,          # 768-dim float array
    payload={
        "text": chunk_text,    # Original text content
        "source": "file.pdf",  # Source filename
        "page": 3,             # Page number (PDFs)
        "classroom_id": "...", # Classroom reference
        "user_id": "...",      # Owner (for notes)
        "timestamp": "...",    # Insertion time
        "chunk_index": 5,      # Position in document
        "metadata": {}         # Additional metadata
    }
)
```

---

## 52.5 Search Operations

### Semantic Search

```python
def search(self, collection: str, query: str, limit: int = 5, 
           filters: dict = None) -> list:
    query_vector = self.embedding_model.encode(query).tolist()
    
    filter_conditions = None
    if filters:
        filter_conditions = Filter(
            must=[
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
        )
    
    results = self.client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=limit,
        query_filter=filter_conditions,
        with_payload=True,
        score_threshold=0.5    # Minimum relevance
    )
    
    return [
        {
            "text": r.payload["text"],
            "score": r.score,
            "source": r.payload.get("source"),
            "metadata": r.payload.get("metadata", {})
        }
        for r in results
    ]
```

### Filtered Search Examples

```python
# Search within a specific classroom
results = qdrant.search(
    collection="classroom_materials",
    query="explain photosynthesis",
    filters={"classroom_id": "cls_123"}
)

# Search user's personal notes
results = qdrant.search(
    collection="student_notes",
    query="neural networks",
    filters={"user_id": "usr_456"}
)

# Search meeting transcript
results = qdrant.search(
    collection="meeting_chunks",
    query="what was discussed about algorithms",
    filters={"meeting_id": "mtg_789"}
)
```

---

## 52.6 Indexing Pipelines

### Document Indexing (`services/web_ingest_service.py`)

```
Document → Extract Text → Chunk (500 chars) → Embed → Upsert to Qdrant
```

### Meeting Indexing (`services/meeting_embedding_service.py`)

```
Transcript → Split by segments → Embed with timestamps → Upsert
```

### Notes Indexing (`services/notes_embedding.py`)

```
Note text → Chunk → Embed → Upsert with user_id filter
```

### Web Resource Indexing (`services/web_cache_service.py`)

```
URL → Fetch → Extract (trafilatura) → Chunk → Embed → Upsert
```

---

## 52.7 Collection Management

```python
# List all collections
collections = client.get_collections()

# Get collection info
info = client.get_collection("classroom_materials")
# → points_count, vectors_count, segments_count

# Delete collection
client.delete_collection("temp_collection")

# Delete specific points
client.delete(
    collection_name="classroom_materials",
    points_selector=FilterSelector(
        filter=Filter(
            must=[FieldCondition(key="classroom_id", match=MatchValue(value="cls_123"))]
        )
    )
)
```

---

## 52.8 Performance Characteristics

| Metric | Value |
|--------|-------|
| Embedding time | ~50ms per chunk (CPU) |
| Search latency | ~5-15ms per query |
| Index size | ~1 KB per vector point |
| Max collection size | Limited by RAM |
| Recommended points | <1M per collection |

### Docker Volume

```yaml
qdrant:
    volumes:
        - qdrant_data:/qdrant/storage    # Persistent vector storage
```
