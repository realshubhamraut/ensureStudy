# Page 67: Text Chunking & Embedding Strategies

---

## 67.1 Overview

ensureStudy uses **semantic-aware text chunking** to split documents into vector-searchable pieces. The chunking strategy directly impacts RAG quality — chunks must be large enough for context but small enough for precise retrieval.

---

## 67.2 Chunking Configuration

```python
# Default chunking parameters
CHUNK_SIZE = 500          # Characters per chunk
CHUNK_OVERLAP = 50        # Character overlap between chunks
SEPARATOR = "\n\n"        # Preferred split point
MIN_CHUNK_SIZE = 100      # Minimum viable chunk
MAX_CHUNK_SIZE = 1000     # Hard limit
```

---

## 67.3 Chunking Strategies

### Strategy 1: Character-Based (Default)

```python
class CharacterChunker:
    def chunk(self, text: str) -> list:
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            
            # Try to break at paragraph
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + MIN_CHUNK_SIZE:
                end = para_break
            else:
                # Try sentence boundary
                sent_break = text.rfind(". ", start, end)
                if sent_break > start + MIN_CHUNK_SIZE:
                    end = sent_break + 1
            
            chunks.append(text[start:end].strip())
            start = end - CHUNK_OVERLAP
        
        return [c for c in chunks if len(c) >= MIN_CHUNK_SIZE]
```

### Strategy 2: Recursive Text Splitter (LangChain)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len
)

chunks = splitter.split_text(document_text)
```

### Strategy 3: Semantic Chunking

```python
class SemanticChunker:
    """Split based on topic boundaries using embeddings"""
    
    def chunk(self, text: str) -> list:
        sentences = self._split_sentences(text)
        embeddings = self.model.encode(sentences)
        
        # Find semantic boundaries
        boundaries = []
        for i in range(1, len(embeddings)):
            similarity = cosine_similarity(embeddings[i-1], embeddings[i])
            if similarity < BOUNDARY_THRESHOLD:
                boundaries.append(i)
        
        # Group sentences between boundaries
        chunks = self._group_by_boundaries(sentences, boundaries)
        return chunks
```

---

## 67.4 Content-Type-Specific Chunking

| Content Type | Strategy | Chunk Size | Notes |
|-------------|----------|-----------|-------|
| PDF text | Recursive | 500 chars | Respects paragraphs |
| PPTX slides | Per-slide | 1 slide | Each slide = 1 chunk |
| Meeting transcript | Time-based | 2-min segments | Timestamped chunks |
| Web pages | Section-based | By `<h2>` tags | Respects HTML structure |
| Code files | Function-based | Per function | AST-aware splitting |
| Notes | Paragraph-based | 300 chars | Shorter for precision |

---

## 67.5 Embedding Model

### Model: `all-mpnet-base-v2` (sentence-transformers)

| Property | Value |
|----------|-------|
| Dimensions | 768 |
| Max Sequence | 384 tokens |
| Speed | ~50ms per chunk (CPU) |
| Quality | State-of-the-art for its size |
| Size | ~420 MB |
| Training | 1B+ sentence pairs |

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

# Single embedding
vector = model.encode("What is photosynthesis?")

# Batch embedding (faster)
vectors = model.encode(chunks, batch_size=32, show_progress_bar=True)
```

---

## 67.6 Chunk Metadata

Every chunk stored in Qdrant carries metadata:

```python
{
    "text": "Photosynthesis is the process by which...",
    "source": "biology_chapter5.pdf",
    "page": 12,
    "chunk_index": 3,
    "total_chunks": 45,
    "classroom_id": "cls_123",
    "subject": "Biology",
    "created_at": "2025-02-15T10:30:00Z",
    "word_count": 87,
    "has_formula": false,
    "has_table": false
}
```

---

## 67.7 Retrieval Quality Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| Top-5 recall | >80% | ~85% |
| MRR (Mean Reciprocal Rank) | >0.6 | ~0.72 |
| Latency (search) | <50ms | ~15ms |
| Relevance threshold | >0.5 | Cosine similarity |

---

## 67.8 Chunking Pipeline

```mermaid
flowchart TB
    subgraph MAIN["Chunking Pipeline "]
        direction TB
        N0["Document Text"]
        N1["Split into chunks (500 chars, 50 overlap)"]
        N2["Filter empty/too-small chunks"]
        N3["Enrich metadata (page, source, subject)"]
        N4["Batch embed (sentence-transformers)"]
        N5["Upsert to Qdrant with payload"]
        N6["Store chunk references in PostgreSQL"]
    end

    style MAIN fill:#3b82f6,color:#fff
```
