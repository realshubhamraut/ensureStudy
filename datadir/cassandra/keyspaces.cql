-- ============================================================================
-- Cassandra Keyspaces for ensureStudy
-- ============================================================================
-- Real-time analytics and time-series data
-- Run with: cqlsh -f datadir/cassandra/keyspaces.cql
-- ============================================================================

-- Create keyspace for analytics
CREATE KEYSPACE IF NOT EXISTS ensure_study_analytics
WITH replication = {
    'class': 'SimpleStrategy',
    'replication_factor': 1
}
AND durable_writes = true;

-- Create keyspace for time-series data
CREATE KEYSPACE IF NOT EXISTS ensure_study_timeseries
WITH replication = {
    'class': 'SimpleStrategy',
    'replication_factor': 1
}
AND durable_writes = true;

-- Note: In production, use NetworkTopologyStrategy:
-- CREATE KEYSPACE ensure_study_analytics
-- WITH replication = {
--     'class': 'NetworkTopologyStrategy',
--     'dc1': 3
-- };
