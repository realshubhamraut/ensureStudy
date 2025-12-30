-- ============================================================================
-- ensureStudy Database Schema - PostgreSQL
-- ============================================================================
-- Complete CREATE TABLE statements for all tables
-- 
-- Database: PostgreSQL 15+
-- Extensions Required: uuid-ossp
-- ============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- CORE TABLES (Flask Backend)
-- ============================================================================

-- Users table (managed by Flask-SQLAlchemy)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(120) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role VARCHAR(20) NOT NULL DEFAULT 'student',  -- 'student', 'teacher', 'admin'
    profile_picture VARCHAR(500),
    is_active BOOLEAN DEFAULT TRUE,
    email_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);

-- Classrooms table
CREATE TABLE IF NOT EXISTS classrooms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    code VARCHAR(10) NOT NULL UNIQUE,  -- Access code for joining
    description TEXT,
    subject VARCHAR(100),
    teacher_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    access_token VARCHAR(100),  -- Token for material downloads
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_classrooms_code ON classrooms(code);
CREATE INDEX IF NOT EXISTS idx_classrooms_teacher_id ON classrooms(teacher_id);

-- Classroom enrollments (students in classrooms)
CREATE TABLE IF NOT EXISTS classroom_enrollments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    classroom_id UUID NOT NULL REFERENCES classrooms(id) ON DELETE CASCADE,
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    enrolled_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    
    UNIQUE(classroom_id, student_id)
);

CREATE INDEX IF NOT EXISTS idx_enrollments_classroom ON classroom_enrollments(classroom_id);
CREATE INDEX IF NOT EXISTS idx_enrollments_student ON classroom_enrollments(student_id);

-- Classroom materials (uploaded files)
CREATE TABLE IF NOT EXISTS classroom_materials (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    classroom_id UUID NOT NULL REFERENCES classrooms(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    file_type VARCHAR(50),  -- 'pdf', 'document', 'video', 'youtube', 'image'
    file_url VARCHAR(500),
    file_size INTEGER,
    s3_key VARCHAR(500),
    uploaded_by UUID REFERENCES users(id),
    display_order INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Indexing columns (added via migration)
    indexing_status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'processing', 'indexed', 'error'
    indexed_at TIMESTAMP,
    chunk_count INTEGER DEFAULT 0,
    indexing_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_materials_classroom ON classroom_materials(classroom_id);
CREATE INDEX IF NOT EXISTS idx_materials_indexing_status ON classroom_materials(indexing_status);

-- ============================================================================
-- DOCUMENT INGESTION TABLES
-- ============================================================================

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
    status VARCHAR(20) DEFAULT 'uploaded',  -- 'uploaded', 'processing', 'indexed', 'error'
    requires_manual_review BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_doc_hash_version UNIQUE (file_hash, version)
);

CREATE INDEX IF NOT EXISTS idx_documents_class_id ON documents(class_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded_by ON documents(uploaded_by);
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);

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
    ocr_method VARCHAR(20),  -- 'nanonets', 'tesseract', 'paddleocr'
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_doc_page UNIQUE (document_id, page_number)
);

CREATE INDEX IF NOT EXISTS idx_document_pages_document_id ON document_pages(document_id);
CREATE INDEX IF NOT EXISTS idx_document_pages_ocr_confidence ON document_pages(ocr_confidence);

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
    content_type VARCHAR(20) DEFAULT 'text',  -- 'text', 'formula_image', 'table'
    embedding_hash VARCHAR(64),  -- For caching
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_qdrant_id ON document_chunks(qdrant_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_page ON document_chunks(document_id, page_number);

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

-- ============================================================================
-- AI TUTOR SESSION TABLES
-- ============================================================================

-- Tutor sessions: session tracking for resource chaining
CREATE TABLE IF NOT EXISTS tutor_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    classroom_id UUID REFERENCES classrooms(id) ON DELETE SET NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_active_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP,
    config JSONB DEFAULT '{"ttl_hours": 24, "max_resources": 25, "relatedness_threshold": 0.65, "relatedness_lookback": 3}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Session Intelligence columns (added via migration)
    last_topic_vector JSONB,  -- Centroid of related turn embeddings
    last_decision VARCHAR(20) DEFAULT 'new_topic',  -- 'related' or 'new_topic'
    consecutive_borderline INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_tutor_sessions_user_id ON tutor_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_tutor_sessions_classroom_id ON tutor_sessions(classroom_id);
CREATE INDEX IF NOT EXISTS idx_tutor_sessions_last_active ON tutor_sessions(last_active_at);

-- Session turns: query history with embeddings
CREATE TABLE IF NOT EXISTS session_turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES tutor_sessions(id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    question TEXT NOT NULL,
    question_hash VARCHAR(64) NOT NULL,
    question_embedding JSONB,
    related_to_previous BOOLEAN DEFAULT FALSE,
    relatedness_score REAL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    UNIQUE(session_id, turn_number)
);

CREATE INDEX IF NOT EXISTS idx_session_turns_session_id ON session_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_session_turns_question_hash ON session_turns(question_hash);

-- Session resources: discovered resources with dedup metadata
CREATE TABLE IF NOT EXISTS session_resources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES tutor_sessions(id) ON DELETE CASCADE,
    resource_type VARCHAR(20) NOT NULL,  -- 'video', 'article', 'pdf', 'note'
    source VARCHAR(20) NOT NULL,  -- 'youtube', 'web', 'classroom', 'notes'
    url TEXT,
    canonical_url TEXT,
    content_hash VARCHAR(64),
    title VARCHAR(500),
    preview_summary VARCHAR(300),
    qdrant_collection VARCHAR(100),
    qdrant_point_ids JSONB DEFAULT '[]'::jsonb,
    inline_render BOOLEAN DEFAULT FALSE,
    inline_html TEXT,
    page_number INTEGER,
    bbox JSONB,
    signed_url TEXT,
    inserted_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_referenced_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_session_resources_session_id ON session_resources(session_id);
CREATE INDEX IF NOT EXISTS idx_session_resources_canonical_url ON session_resources(canonical_url);
CREATE INDEX IF NOT EXISTS idx_session_resources_content_hash ON session_resources(content_hash);
CREATE INDEX IF NOT EXISTS idx_session_resources_last_ref ON session_resources(last_referenced_at);

-- ============================================================================
-- EVALUATION/EXAM TABLES (if applicable)
-- ============================================================================

-- Note: Additional tables for exams, question papers, answer submissions
-- may exist in the Flask models. Run `SHOW TABLES` on live DB to confirm.

-- ============================================================================
-- PERMISSIONS
-- ============================================================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ensure_study_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ensure_study_user;
