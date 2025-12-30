-- PostgreSQL initialization script for ensureStudy
-- This script runs when the PostgreSQL container is first created

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- The database and user are created automatically by Docker environment variables:
-- POSTGRES_USER: ensure_study_user
-- POSTGRES_PASSWORD: secure_password_123
-- POSTGRES_DB: ensure_study

-- Grant privileges (already done by default, but explicit is good)
GRANT ALL PRIVILEGES ON DATABASE ensure_study TO ensure_study_user;
