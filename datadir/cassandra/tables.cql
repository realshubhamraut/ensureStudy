-- ============================================================================
-- Cassandra Tables for ensureStudy
-- ============================================================================
-- Real-time meeting analytics and event streaming
-- Run with: cqlsh -f datadir/cassandra/tables.cql
-- ============================================================================

USE ensure_study_analytics;

-- ============================================================================
-- MEETING ANALYTICS (Real-time during meetings)
-- ============================================================================

-- Real-time participant engagement metrics
CREATE TABLE IF NOT EXISTS meeting_engagement (
    meeting_id UUID,
    timestamp TIMESTAMP,
    participant_id UUID,
    metric_type TEXT,  -- 'attention', 'participation', 'emotion'
    metric_value FLOAT,
    metadata MAP<TEXT, TEXT>,
    PRIMARY KEY ((meeting_id), timestamp, participant_id)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Aggregated meeting stats (per minute)
CREATE TABLE IF NOT EXISTS meeting_stats_by_minute (
    meeting_id UUID,
    minute_bucket TIMESTAMP,
    participant_count INT,
    avg_attention_score FLOAT,
    active_speakers INT,
    chat_message_count INT,
    screen_share_active BOOLEAN,
    PRIMARY KEY ((meeting_id), minute_bucket)
) WITH CLUSTERING ORDER BY (minute_bucket DESC);

-- ============================================================================
-- USER ACTIVITY STREAMS
-- ============================================================================

-- User activity events (for analytics dashboard)
CREATE TABLE IF NOT EXISTS user_activity (
    user_id UUID,
    activity_date DATE,
    timestamp TIMESTAMP,
    activity_type TEXT,  -- 'login', 'study', 'assessment', 'meeting'
    details MAP<TEXT, TEXT>,
    duration_seconds INT,
    PRIMARY KEY ((user_id, activity_date), timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Study sessions by user
CREATE TABLE IF NOT EXISTS study_sessions (
    user_id UUID,
    session_date DATE,
    session_start TIMESTAMP,
    session_end TIMESTAMP,
    subject TEXT,
    topic TEXT,
    duration_minutes INT,
    questions_asked INT,
    resources_accessed INT,
    PRIMARY KEY ((user_id, session_date), session_start)
) WITH CLUSTERING ORDER BY (session_start DESC);

-- ============================================================================
-- PROCTORING EVENTS (Real-time exam monitoring)
-- ============================================================================

-- Proctoring violations/events
CREATE TABLE IF NOT EXISTS proctor_events (
    exam_session_id UUID,
    timestamp TIMESTAMP,
    event_type TEXT,  -- 'face_not_detected', 'multiple_faces', 'phone_detected', 'tab_switch'
    severity TEXT,    -- 'low', 'medium', 'high', 'critical'
    confidence FLOAT,
    frame_number INT,
    metadata MAP<TEXT, TEXT>,
    PRIMARY KEY ((exam_session_id), timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- Proctor session summaries
CREATE TABLE IF NOT EXISTS proctor_session_summary (
    exam_session_id UUID,
    user_id UUID,
    exam_id UUID,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    total_violations INT,
    violation_counts MAP<TEXT, INT>,  -- {'face_not_detected': 5, 'phone_detected': 1}
    risk_score FLOAT,
    decision TEXT,  -- 'approved', 'flagged', 'rejected'
    PRIMARY KEY (exam_session_id)
);

-- ============================================================================
-- KAFKA EVENT LOG (Event sourcing)
-- ============================================================================

-- All events log (for replay/audit)
CREATE TABLE IF NOT EXISTS event_log (
    partition_key TEXT,  -- Usually date or topic
    event_time TIMESTAMP,
    event_id UUID,
    event_type TEXT,
    aggregate_id UUID,
    payload TEXT,  -- JSON serialized
    PRIMARY KEY ((partition_key), event_time, event_id)
) WITH CLUSTERING ORDER BY (event_time DESC);

-- ============================================================================
-- TIME-SERIES KEYSPACE TABLES
-- ============================================================================

USE ensure_study_timeseries;

-- Learning progress over time
CREATE TABLE IF NOT EXISTS learning_progress (
    user_id UUID,
    subject TEXT,
    time_bucket TIMESTAMP,  -- Hourly buckets
    questions_attempted INT,
    questions_correct INT,
    time_spent_seconds INT,
    topics_covered SET<TEXT>,
    PRIMARY KEY ((user_id, subject), time_bucket)
) WITH CLUSTERING ORDER BY (time_bucket DESC);

-- Platform-wide metrics (for admin dashboard)
CREATE TABLE IF NOT EXISTS platform_metrics (
    metric_date DATE,
    hour INT,
    metric_name TEXT,
    metric_value DOUBLE,
    PRIMARY KEY ((metric_date), hour, metric_name)
) WITH CLUSTERING ORDER BY (hour DESC);
