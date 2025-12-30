-- Document Ingestion Schema Migration
-- Creates tables for document storage, pages, chunks, and quality reports

-- Enable UUID extension if not exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents table: main record for uploaded files
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::TEXT,
    class_id VARCHAR(36) NOT NULL,
    title VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    s3_path VARCHAR(500) NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    file_size INTEGER,
    mime_type VARCHAR(100),
    uploaded_by VARCHAR(36) NOT NULL,
    uploaded_at TIMESTAMP DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'uploaded',  -- uploaded, processing, indexed, error
    requires_manual_review BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Foreign keys (commented for flexibility - can be added later)
    -- FOREIGN KEY (class_id) REFERENCES classrooms(id) ON DELETE CASCADE,
    -- FOREIGN KEY (uploaded_by) REFERENCES users(id)
    
    CONSTRAINT unique_doc_hash_version UNIQUE (file_hash, version)
);

-- Document pages: per-page OCR results
CREATE TABLE IF NOT EXISTS document_pages (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::TEXT,
    document_id VARCHAR(36) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    s3_page_json_path VARCHAR(500),
    s3_page_image_path VARCHAR(500),
    ocr_confidence FLOAT,
    text_length INTEGER DEFAULT 0,
    block_count INTEGER DEFAULT 0,
    ocr_method VARCHAR(20),  -- nanonets, tesseract, paddleocr
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_doc_page UNIQUE (document_id, page_number)
);

-- Document chunks: chunked text for RAG
CREATE TABLE IF NOT EXISTS document_chunks (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::TEXT,
    document_id VARCHAR(36) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    block_id VARCHAR(36),
    chunk_index INTEGER NOT NULL,
    preview_text VARCHAR(200),
    full_text TEXT,
    bbox_json TEXT,  -- JSON array: [x1, y1, x2, y2]
    qdrant_id VARCHAR(36),
    token_count INTEGER,
    content_type VARCHAR(20) DEFAULT 'text',  -- text, formula_image, table
    embedding_hash VARCHAR(64),  -- For caching
    created_at TIMESTAMP DEFAULT NOW()
);

-- Quality reports: per-document quality metrics
CREATE TABLE IF NOT EXISTS document_quality_reports (
    id VARCHAR(36) PRIMARY KEY DEFAULT uuid_generate_v4()::TEXT,
    document_id VARCHAR(36) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    avg_ocr_confidence FLOAT,
    min_ocr_confidence FLOAT,
    pages_processed INTEGER DEFAULT 0,
    pages_failed INTEGER DEFAULT 0,
    flagged_pages INTEGER[],  -- Array of page numbers needing review
    total_chunks INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_class_id ON documents(class_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded_by ON documents(uploaded_by);
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);

CREATE INDEX IF NOT EXISTS idx_document_pages_document_id ON document_pages(document_id);
CREATE INDEX IF NOT EXISTS idx_document_pages_ocr_confidence ON document_pages(ocr_confidence);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_qdrant_id ON document_chunks(qdrant_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_page ON document_chunks(document_id, page_number);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ensure_study_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ensure_study_user;
