-- ============================================================================
-- ensureStudy Qdrant Vector Database Schema
-- ============================================================================
-- Schema definitions for all Qdrant collections used in the application
-- 
-- Vector Database: Qdrant
-- Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
--                  or OpenAI/custom (1536 dimensions)
-- ============================================================================

-- ============================================================================
-- COLLECTION: ensure_study_documents (Main RAG Collection)
-- ============================================================================
-- Purpose: Primary collection for document embeddings used in RAG retrieval
-- Vector Size: 384 (all-MiniLM-L6-v2)
-- Distance: COSINE

/*
qdrant.create_collection(
    collection_name="ensure_study_documents",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    ),
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=20000,
        memmap_threshold=20000
    ),
    hnsw_config=HnswConfigDiff(
        m=16,
        ef_construct=100
    )
)
*/

-- Payload Schema:
-- {
--     "text": "First 1000 chars of chunk text",
--     "full_text": "Complete chunk text",
--     "source": "textbook|notes|video|web",
--     "page": 1,
--     "subject": "mathematics|physics|chemistry|history|...",
--     "topic": "Pythagorean Theorem",
--     "difficulty": "easy|medium|hard",
--     "type": "textbook|notes|video_transcript|web_article",
--     "chunk_index": 0
-- }

-- Payload Indices:
CREATE INDEX classroom_id ON ensure_study_documents (classroom_id) AS KEYWORD;
CREATE INDEX student_id ON ensure_study_documents (student_id) AS KEYWORD;
CREATE INDEX document_id ON ensure_study_documents (document_id) AS KEYWORD;
CREATE INDEX source_type ON ensure_study_documents (source_type) AS KEYWORD;
CREATE INDEX subject ON ensure_study_documents (subject) AS KEYWORD;
CREATE INDEX contains_formula ON ensure_study_documents (contains_formula) AS BOOL;


-- ============================================================================
-- COLLECTION: classroom_materials (Classroom-specific RAG)
-- ============================================================================
-- Purpose: Classroom-specific document chunks for teacher-uploaded materials
-- Vector Size: 384
-- Distance: COSINE

/*
qdrant.create_collection(
    collection_name="classroom_materials",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    ),
    optimizers_config=OptimizersConfigDiff(
        memmap_threshold=20000,
        indexing_threshold=10000
    ),
    hnsw_config=HnswConfigDiff(
        m=16,
        ef_construct=100,
        full_scan_threshold=10000
    ),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,
            always_ram=True
        )
    )
)
*/

-- Payload Schema:
-- {
--     "classroom_id": "uuid",
--     "document_id": "uuid",
--     "student_id": null,  -- For classroom materials
--     "title": "Document title",
--     "text": "Chunk text content",
--     "page_number": 1,
--     "chunk_index": 0,
--     "source_type": "classroom",
--     "subject": "mathematics",
--     "contains_formula": false,
--     "bbox": [x1, y1, x2, y2],  -- Bounding box for PDF location
--     "created_at": "2024-01-01T00:00:00Z"
-- }

-- Payload Indices:
CREATE INDEX classroom_id ON classroom_materials (classroom_id) AS KEYWORD;
CREATE INDEX document_id ON classroom_materials (document_id) AS KEYWORD;
CREATE INDEX source_type ON classroom_materials (source_type) AS KEYWORD;
CREATE INDEX subject ON classroom_materials (subject) AS KEYWORD;


-- ============================================================================
-- COLLECTION: web_cache (Semantic Web Content Cache)
-- ============================================================================
-- Purpose: Cache for web-crawled content to avoid repeated crawls
-- Vector Size: 384
-- Distance: COSINE

/*
qdrant.create_collection(
    collection_name="web_cache",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)
*/

-- Payload Schema:
-- {
--     "query": "Original search query",
--     "query_hash": "sha256 hash of normalized query",
--     "answer": "Cached answer text",
--     "sources": ["url1", "url2"],
--     "confidence": 0.95,
--     "created_at": "2024-01-01T00:00:00Z",
--     "expires_at": "2024-01-08T00:00:00Z"
-- }


-- ============================================================================
-- COLLECTION: student_notes (Student Notes Embeddings)
-- ============================================================================
-- Purpose: Personal student notes for personalized retrieval
-- Vector Size: 384
-- Distance: COSINE

/*
qdrant.create_collection(
    collection_name="student_notes",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)
*/

-- Payload Schema:
-- {
--     "student_id": "uuid",
--     "note_id": "uuid",
--     "title": "Note title",
--     "text": "Note content chunk",
--     "subject": "mathematics",
--     "topics": ["algebra", "equations"],
--     "created_at": "2024-01-01T00:00:00Z"
-- }

-- Payload Indices:
CREATE INDEX student_id ON student_notes (student_id) AS KEYWORD;
CREATE INDEX subject ON student_notes (subject) AS KEYWORD;


-- ============================================================================
-- COLLECTION: youtube_transcripts (YouTube Video Transcripts)
-- ============================================================================
-- Purpose: Cached YouTube video transcripts for educational content
-- Vector Size: 384
-- Distance: COSINE

-- Payload Schema:
-- {
--     "video_id": "dQw4w9WgXcQ",
--     "title": "Video title",
--     "channel": "Channel name",
--     "text": "Transcript chunk",
--     "timestamp_start": 120.5,  -- seconds
--     "timestamp_end": 180.0,
--     "language": "en"
-- }


-- ============================================================================
-- ENVIRONMENT VARIABLES FOR QDRANT
-- ============================================================================
-- QDRANT_HOST=localhost
-- QDRANT_PORT=6333
-- QDRANT_API_KEY=<optional>
-- QDRANT_COLLECTION_NAME=ensure_study_documents
-- EMBEDDING_DIMENSIONS=384
