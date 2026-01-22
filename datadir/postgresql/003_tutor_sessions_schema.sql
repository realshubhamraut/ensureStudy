-- ============================================================================
-- ensureStudy PostgreSQL Schema - AI Tutor Sessions
-- ============================================================================
-- Session tracking, conversation history, resource management
-- ============================================================================

-- ============================================================================
-- TUTOR SESSIONS (Main session container)
-- ============================================================================
CREATE TABLE IF NOT EXISTS tutor_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    classroom_id UUID REFERENCES classrooms(id) ON DELETE SET NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_active_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP,
    config JSONB DEFAULT '{
        "ttl_hours": 24,
        "max_resources": 25,
        "relatedness_threshold": 0.65,
        "relatedness_lookback": 3
    }'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Session Intelligence columns
    last_topic_vector JSONB,  -- Centroid of related turn embeddings
    last_decision VARCHAR(20) DEFAULT 'new_topic',  -- 'related' or 'new_topic'
    consecutive_borderline INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_tutor_sessions_user_id ON tutor_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_tutor_sessions_classroom_id ON tutor_sessions(classroom_id);
CREATE INDEX IF NOT EXISTS idx_tutor_sessions_last_active ON tutor_sessions(last_active_at);

-- ============================================================================
-- SESSION TURNS (Query history with embeddings)
-- ============================================================================
CREATE TABLE IF NOT EXISTS session_turns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES tutor_sessions(id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    question TEXT NOT NULL,
    question_hash VARCHAR(64) NOT NULL,
    question_embedding JSONB,  -- Store embedding for relatedness calculation
    answer TEXT,
    answer_sources JSONB DEFAULT '[]'::jsonb,
    related_to_previous BOOLEAN DEFAULT FALSE,
    relatedness_score REAL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    UNIQUE(session_id, turn_number)
);

CREATE INDEX IF NOT EXISTS idx_session_turns_session_id ON session_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_session_turns_question_hash ON session_turns(question_hash);

-- ============================================================================
-- SESSION RESOURCES (Discovered resources with dedup metadata)
-- ============================================================================
CREATE TABLE IF NOT EXISTS session_resources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES tutor_sessions(id) ON DELETE CASCADE,
    resource_type VARCHAR(20) NOT NULL,  -- 'video', 'article', 'pdf', 'note'
    source VARCHAR(20) NOT NULL,  -- 'youtube', 'web', 'classroom', 'notes'
    url TEXT,
    canonical_url TEXT,  -- Normalized URL for deduplication
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
-- TOPIC ANCHORS (For topic tracking across conversations)
-- ============================================================================
CREATE TABLE IF NOT EXISTS topic_anchors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES tutor_sessions(id) ON DELETE CASCADE,
    canonical_title VARCHAR(255) NOT NULL,
    canonical_seed VARCHAR(255),
    topic_embedding JSONB,
    subject_scope JSONB DEFAULT '[]'::jsonb,
    locked_entities JSONB DEFAULT '[]'::jsonb,
    source VARCHAR(50),  -- 'user_query', 'extracted', 'inferred'
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMP,
    turn_count INTEGER DEFAULT 0,
    end_reason VARCHAR(50)  -- 'new_topic', 'session_end', 'timeout'
);

CREATE INDEX IF NOT EXISTS idx_topic_anchors_session ON topic_anchors(session_id);
CREATE INDEX IF NOT EXISTS idx_topic_anchors_active ON topic_anchors(session_id, ended_at) WHERE ended_at IS NULL;
