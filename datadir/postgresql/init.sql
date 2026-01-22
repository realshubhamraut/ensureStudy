-- ============================================================================
-- ensureStudy PostgreSQL Init Script
-- ============================================================================
-- This script runs when the PostgreSQL container is first created.
-- Docker automatically creates user/password/database from environment vars.
-- ============================================================================

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE ensure_study TO ensure_study_user;

-- Note: Full schema is loaded from the individual schema files:
-- - 001_core_schema.sql
-- - 002_documents_schema.sql  
-- - 003_tutor_sessions_schema.sql
-- - 004_softskills_schema.sql
--
-- Run with: psql $DATABASE_URL -f datadir/postgresql/00X_*.sql
