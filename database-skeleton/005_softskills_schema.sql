-- ============================================
-- Soft Skills Schema
-- PostgreSQL Tables for Soft Skills Evaluation
-- ============================================
-- 
-- This schema supports real-time evaluation of communication skills:
-- - Fluency (WPM, pauses, fillers)
-- - Grammar (errors, sentence structure)
-- - Eye Contact (gaze direction, head pose)
-- - Hand Gestures (movement, stability)
-- - Posture (lean, alignment)
--
-- Run migration: python scripts/migrate_softskills.py --up
-- ============================================

-- Sessions table: Main container for evaluation sessions
CREATE TABLE IF NOT EXISTS softskills_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    module_type VARCHAR(30) NOT NULL DEFAULT 'communication',  -- 'communication', 'mock_interview'
    avatar_id VARCHAR(20),                                      -- Avatar used in session
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    duration_seconds INTEGER,
    prompt_count INTEGER DEFAULT 0,
    is_complete BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'
);

-- Responses: Per-prompt data
CREATE TABLE IF NOT EXISTS softskills_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES softskills_sessions(id) ON DELETE CASCADE,
    prompt_index INTEGER NOT NULL,
    prompt_text TEXT NOT NULL,
    transcript TEXT,                     -- Speech-to-text result
    audio_duration_seconds FLOAT,
    video_duration_seconds FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(session_id, prompt_index)
);

-- Fluency Metrics: Speech analysis per response
CREATE TABLE IF NOT EXISTS fluency_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id UUID NOT NULL REFERENCES softskills_responses(id) ON DELETE CASCADE,
    words_per_minute FLOAT,
    word_count INTEGER DEFAULT 0,
    pause_count INTEGER DEFAULT 0,
    pause_ratio FLOAT,                   -- pause_time / total_time
    filler_word_count INTEGER DEFAULT 0,
    filler_words_detected TEXT[],        -- ['um', 'uh', 'like', ...]
    score FLOAT,                         -- 0-100
    created_at TIMESTAMP DEFAULT NOW()
);

-- Grammar Metrics: Text analysis per response
CREATE TABLE IF NOT EXISTS grammar_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id UUID NOT NULL REFERENCES softskills_responses(id) ON DELETE CASCADE,
    error_count INTEGER DEFAULT 0,
    sentence_count INTEGER DEFAULT 0,
    avg_sentence_length FLOAT,
    vocabulary_richness FLOAT,           -- unique_words / total_words
    errors_detected JSONB DEFAULT '[]',  -- [{"type": "grammar", "message": "..."}]
    score FLOAT,                         -- 0-100
    created_at TIMESTAMP DEFAULT NOW()
);

-- Visual Metrics: Aggregated video analysis per response
CREATE TABLE IF NOT EXISTS visual_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id UUID NOT NULL REFERENCES softskills_responses(id) ON DELETE CASCADE,
    frames_analyzed INTEGER DEFAULT 0,
    
    -- Eye Contact
    eye_contact_score FLOAT,
    gaze_center_ratio FLOAT,             -- % frames looking at camera
    gaze_left_ratio FLOAT,
    gaze_right_ratio FLOAT,
    
    -- Head Pose
    head_forward_ratio FLOAT,
    avg_head_pitch FLOAT,
    avg_head_yaw FLOAT,
    head_stability_score FLOAT,
    
    -- Hands
    hand_gesture_score FLOAT,
    hands_visible_ratio FLOAT,
    hand_movement_frequency FLOAT,
    
    -- Posture
    posture_score FLOAT,
    upper_body_stability FLOAT,
    shoulder_alignment_score FLOAT,
    is_upright_ratio FLOAT,
    
    -- Expressions
    expression_score FLOAT,
    smile_ratio FLOAT,
    neutral_ratio FLOAT,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Frame Analysis: Raw per-frame detection data (optional, for debugging)
CREATE TABLE IF NOT EXISTS frame_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id UUID NOT NULL REFERENCES softskills_responses(id) ON DELETE CASCADE,
    frame_number INTEGER NOT NULL,
    timestamp_ms INTEGER,
    
    -- Face
    face_detected BOOLEAN DEFAULT FALSE,
    gaze_direction VARCHAR(20),          -- 'center', 'left', 'right'
    gaze_ratio FLOAT,
    head_pose_pitch FLOAT,
    head_pose_yaw FLOAT,
    head_pose_roll FLOAT,
    
    -- Hands
    hands_visible BOOLEAN DEFAULT FALSE,
    num_hands INTEGER DEFAULT 0,
    hand_positions JSONB,
    
    -- Body
    body_detected BOOLEAN DEFAULT FALSE,
    shoulder_tilt FLOAT,
    body_lean FLOAT,
    shoulder_center JSONB,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Final Scores: Weighted overall scores per session
CREATE TABLE IF NOT EXISTS softskills_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES softskills_sessions(id) ON DELETE CASCADE,
    
    -- Individual Scores (0-100)
    fluency_score FLOAT,
    grammar_score FLOAT,
    eye_contact_score FLOAT,
    hand_gesture_score FLOAT,
    posture_score FLOAT,
    expression_score FLOAT,
    
    -- Weighted Overall
    overall_score FLOAT,
    
    -- Weights used
    weights JSONB DEFAULT '{"fluency": 0.30, "grammar": 0.20, "eye_contact": 0.15, "gesture": 0.10, "posture": 0.10, "expression": 0.10}',
    
    -- Feedback
    strengths TEXT[],
    areas_for_improvement TEXT[],
    detailed_feedback TEXT,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Benchmarks: Configurable thresholds
CREATE TABLE IF NOT EXISTS softskills_benchmarks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR(50) NOT NULL UNIQUE,
    excellent_threshold FLOAT,
    good_threshold FLOAT,
    fair_threshold FLOAT,
    description TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- Indexes
-- ============================================
CREATE INDEX IF NOT EXISTS idx_softskills_sessions_user ON softskills_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_softskills_sessions_started ON softskills_sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_softskills_responses_session ON softskills_responses(session_id);
CREATE INDEX IF NOT EXISTS idx_fluency_metrics_response ON fluency_metrics(response_id);
CREATE INDEX IF NOT EXISTS idx_grammar_metrics_response ON grammar_metrics(response_id);
CREATE INDEX IF NOT EXISTS idx_visual_metrics_response ON visual_metrics(response_id);
CREATE INDEX IF NOT EXISTS idx_frame_analysis_response ON frame_analysis(response_id);
CREATE INDEX IF NOT EXISTS idx_frame_analysis_timestamp ON frame_analysis(response_id, timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_softskills_scores_session ON softskills_scores(session_id);
CREATE INDEX IF NOT EXISTS idx_softskills_scores_overall ON softskills_scores(overall_score DESC);

-- ============================================
-- Default Benchmarks
-- ============================================
INSERT INTO softskills_benchmarks (metric_name, excellent_threshold, good_threshold, fair_threshold, description)
VALUES
    ('wpm', 140, 120, 100, 'Words per minute - optimal speaking rate'),
    ('filler_ratio', 0.02, 0.05, 0.10, 'Filler words as ratio of total words'),
    ('eye_contact', 85, 70, 50, 'Eye contact score (0-100)'),
    ('posture_stability', 90, 75, 60, 'Posture stability score (0-100)'),
    ('shoulder_tilt', 3, 5, 8, 'Shoulder tilt in degrees (lower is better)'),
    ('body_lean', 5, 10, 15, 'Body lean in degrees (lower is better)')
ON CONFLICT (metric_name) DO NOTHING;

-- ============================================
-- Entity Relationship Diagram
-- ============================================
/*
┌──────────────────────┐
│ softskills_sessions  │
│──────────────────────│
│ PK id                │
│ FK user_id           │
│    module_type       │
│    avatar_id         │
│    is_complete       │
└───────────┬──────────┘
            │ 1:N
            ▼
┌──────────────────────┐
│ softskills_responses │
│──────────────────────│
│ PK id                │
│ FK session_id        │
│    prompt_index      │
│    transcript        │
└───────────┬──────────┘
            │ 1:1 (each)
    ┌───────┼───────┬──────────┐
    ▼       ▼       ▼          ▼
┌───────┐ ┌───────┐ ┌────────┐ ┌──────────┐
│fluency│ │grammar│ │visual  │ │frame_    │
│metrics│ │metrics│ │metrics │ │analysis  │
└───────┘ └───────┘ └────────┘ └──────────┘

┌──────────────────────┐
│ softskills_scores    │
│──────────────────────│
│ PK id                │
│ FK session_id        │
│    overall_score     │
│    weights           │
│    feedback          │
└──────────────────────┘
*/
