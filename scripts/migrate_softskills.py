#!/usr/bin/env python3
"""
Soft Skills Database Migration

Creates the database tables required for the soft skills evaluation module:
- softskills_sessions: Session metadata
- softskills_responses: Per-prompt transcripts and metadata
- fluency_metrics: WPM, pauses, filler word data
- grammar_metrics: Grammar error analysis
- visual_metrics: Aggregated eye/hand/posture scores
- frame_analysis: Raw per-frame detection data
- softskills_scores: Final weighted scores

Usage:
    python scripts/migrate_softskills.py --up      # Apply migration
    python scripts/migrate_softskills.py --down    # Rollback migration
    python scripts/migrate_softskills.py --check   # Check if tables exist
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncpg
from dotenv import load_dotenv

# Load environment
load_dotenv(PROJECT_ROOT / ".env")


# =============================================================================
# SQL Migrations
# =============================================================================

UP_MIGRATION = """
-- ============================================
-- Soft Skills Sessions
-- ============================================
CREATE TABLE IF NOT EXISTS softskills_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    module_type VARCHAR(30) NOT NULL DEFAULT 'communication',
    avatar_id VARCHAR(20),
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    duration_seconds INTEGER,
    prompt_count INTEGER DEFAULT 0,
    is_complete BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_softskills_sessions_user 
    ON softskills_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_softskills_sessions_started 
    ON softskills_sessions(started_at DESC);

-- ============================================
-- Session Responses (per prompt)
-- ============================================
CREATE TABLE IF NOT EXISTS softskills_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES softskills_sessions(id) ON DELETE CASCADE,
    prompt_index INTEGER NOT NULL,
    prompt_text TEXT NOT NULL,
    transcript TEXT,
    audio_duration_seconds FLOAT,
    video_duration_seconds FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(session_id, prompt_index)
);

CREATE INDEX IF NOT EXISTS idx_softskills_responses_session 
    ON softskills_responses(session_id);

-- ============================================
-- Fluency Metrics
-- ============================================
CREATE TABLE IF NOT EXISTS fluency_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id UUID NOT NULL REFERENCES softskills_responses(id) ON DELETE CASCADE,
    words_per_minute FLOAT,
    word_count INTEGER DEFAULT 0,
    pause_count INTEGER DEFAULT 0,
    pause_ratio FLOAT,
    filler_word_count INTEGER DEFAULT 0,
    filler_words_detected TEXT[],
    score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fluency_metrics_response 
    ON fluency_metrics(response_id);

-- ============================================
-- Grammar Metrics
-- ============================================
CREATE TABLE IF NOT EXISTS grammar_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id UUID NOT NULL REFERENCES softskills_responses(id) ON DELETE CASCADE,
    error_count INTEGER DEFAULT 0,
    sentence_count INTEGER DEFAULT 0,
    avg_sentence_length FLOAT,
    vocabulary_richness FLOAT,
    errors_detected JSONB DEFAULT '[]',
    score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_grammar_metrics_response 
    ON grammar_metrics(response_id);

-- ============================================
-- Visual Metrics (aggregated per response)
-- ============================================
CREATE TABLE IF NOT EXISTS visual_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id UUID NOT NULL REFERENCES softskills_responses(id) ON DELETE CASCADE,
    frames_analyzed INTEGER DEFAULT 0,
    
    -- Eye Contact
    eye_contact_score FLOAT,
    gaze_center_ratio FLOAT,
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

CREATE INDEX IF NOT EXISTS idx_visual_metrics_response 
    ON visual_metrics(response_id);

-- ============================================
-- Frame Analysis (raw per-frame data)
-- ============================================
CREATE TABLE IF NOT EXISTS frame_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id UUID NOT NULL REFERENCES softskills_responses(id) ON DELETE CASCADE,
    frame_number INTEGER NOT NULL,
    timestamp_ms INTEGER,
    
    -- Face Detection
    face_detected BOOLEAN DEFAULT FALSE,
    num_faces INTEGER DEFAULT 0,
    gaze_direction VARCHAR(20),
    gaze_ratio FLOAT,
    head_pose_pitch FLOAT,
    head_pose_yaw FLOAT,
    head_pose_roll FLOAT,
    
    -- Hand Detection
    hands_visible BOOLEAN DEFAULT FALSE,
    num_hands INTEGER DEFAULT 0,
    hand_positions JSONB,
    
    -- Body Pose
    body_detected BOOLEAN DEFAULT FALSE,
    shoulder_tilt FLOAT,
    body_lean FLOAT,
    shoulder_center JSONB,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_frame_analysis_response 
    ON frame_analysis(response_id);
CREATE INDEX IF NOT EXISTS idx_frame_analysis_timestamp 
    ON frame_analysis(response_id, timestamp_ms);

-- ============================================
-- Final Scores
-- ============================================
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

CREATE INDEX IF NOT EXISTS idx_softskills_scores_session 
    ON softskills_scores(session_id);
CREATE INDEX IF NOT EXISTS idx_softskills_scores_overall 
    ON softskills_scores(overall_score DESC);

-- ============================================
-- Benchmark Thresholds (configuration)
-- ============================================
CREATE TABLE IF NOT EXISTS softskills_benchmarks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR(50) NOT NULL UNIQUE,
    excellent_threshold FLOAT,
    good_threshold FLOAT,
    fair_threshold FLOAT,
    description TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Insert default benchmarks
INSERT INTO softskills_benchmarks (metric_name, excellent_threshold, good_threshold, fair_threshold, description)
VALUES
    ('wpm', 140, 120, 100, 'Words per minute optimal range'),
    ('filler_ratio', 0.02, 0.05, 0.10, 'Filler words as ratio of total words'),
    ('eye_contact', 85, 70, 50, 'Eye contact score threshold'),
    ('posture_stability', 90, 75, 60, 'Posture stability score'),
    ('shoulder_tilt', 3, 5, 8, 'Shoulder tilt in degrees (lower is better)'),
    ('body_lean', 5, 10, 15, 'Body lean in degrees (lower is better)')
ON CONFLICT (metric_name) DO UPDATE SET
    excellent_threshold = EXCLUDED.excellent_threshold,
    good_threshold = EXCLUDED.good_threshold,
    fair_threshold = EXCLUDED.fair_threshold,
    description = EXCLUDED.description,
    updated_at = NOW();

-- Success message
SELECT 'Soft Skills tables created successfully' AS status;
"""

DOWN_MIGRATION = """
-- Drop tables in reverse order (respecting foreign keys)
DROP TABLE IF EXISTS softskills_benchmarks CASCADE;
DROP TABLE IF EXISTS softskills_scores CASCADE;
DROP TABLE IF EXISTS frame_analysis CASCADE;
DROP TABLE IF EXISTS visual_metrics CASCADE;
DROP TABLE IF EXISTS grammar_metrics CASCADE;
DROP TABLE IF EXISTS fluency_metrics CASCADE;
DROP TABLE IF EXISTS softskills_responses CASCADE;
DROP TABLE IF EXISTS softskills_sessions CASCADE;

SELECT 'Soft Skills tables dropped successfully' AS status;
"""

CHECK_TABLES = """
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN (
    'softskills_sessions',
    'softskills_responses',
    'fluency_metrics',
    'grammar_metrics',
    'visual_metrics',
    'frame_analysis',
    'softskills_scores',
    'softskills_benchmarks'
)
ORDER BY table_name;
"""


# =============================================================================
# Migration Functions
# =============================================================================

async def get_connection():
    """Get database connection from environment."""
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        # Build from individual components
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")
        database = os.getenv("POSTGRES_DB", "ensure_study")
        database_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    return await asyncpg.connect(database_url)


async def run_migration(sql: str, description: str):
    """Execute a migration SQL script."""
    conn = None
    try:
        print(f"\n{'='*50}")
        print(f"Running: {description}")
        print(f"{'='*50}\n")
        
        conn = await get_connection()
        
        # Execute the migration
        await conn.execute(sql)
        
        print(f"✅ {description} completed successfully")
        
    except asyncpg.exceptions.UndefinedTableError as e:
        print(f"❌ Error: Table not found - {e}")
        print("   Make sure the 'users' table exists (required for foreign key)")
        return False
    except asyncpg.exceptions.DuplicateObjectError as e:
        print(f"⚠️  Warning: {e}")
        print("   Objects may already exist")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        if conn:
            await conn.close()
    
    return True


async def check_tables():
    """Check which soft skills tables exist."""
    conn = None
    try:
        conn = await get_connection()
        rows = await conn.fetch(CHECK_TABLES)
        
        expected_tables = [
            'softskills_sessions',
            'softskills_responses',
            'fluency_metrics',
            'grammar_metrics',
            'visual_metrics',
            'frame_analysis',
            'softskills_scores',
            'softskills_benchmarks'
        ]
        
        existing = [row['table_name'] for row in rows]
        
        print("\n" + "="*50)
        print("Soft Skills Tables Status")
        print("="*50 + "\n")
        
        for table in expected_tables:
            if table in existing:
                print(f"  ✅ {table}")
            else:
                print(f"  ❌ {table} (missing)")
        
        print(f"\n{len(existing)}/{len(expected_tables)} tables exist")
        
        return len(existing) == len(expected_tables)
        
    finally:
        if conn:
            await conn.close()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Soft Skills Database Migration")
    parser.add_argument("--up", action="store_true", help="Apply migration (create tables)")
    parser.add_argument("--down", action="store_true", help="Rollback migration (drop tables)")
    parser.add_argument("--check", action="store_true", help="Check if tables exist")
    
    args = parser.parse_args()
    
    if args.up:
        success = await run_migration(UP_MIGRATION, "Create Soft Skills Tables")
        if success:
            await check_tables()
    elif args.down:
        # Confirm before dropping
        confirm = input("⚠️  This will DROP all soft skills tables. Continue? [y/N]: ")
        if confirm.lower() == 'y':
            await run_migration(DOWN_MIGRATION, "Drop Soft Skills Tables")
        else:
            print("Rollback cancelled.")
    elif args.check:
        await check_tables()
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
