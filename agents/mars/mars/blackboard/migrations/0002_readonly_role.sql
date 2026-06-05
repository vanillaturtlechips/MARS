-- Migration: 0002_readonly_role
-- Creates a read-only Postgres role used by the investigator tools.
-- The investigator's tools MUST run under this role so they physically
-- cannot write to the blackboard — enforced at the DB level, not by convention.
--
-- Apply manually (requires superuser / CREATEROLE privilege):
--   psql $DB_DSN -f mars/blackboard/migrations/0002_readonly_role.sql

DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mars_reader') THEN
    EXECUTE format(
      'CREATE ROLE mars_reader WITH LOGIN PASSWORD %L',
      ''
    );
  END IF;
END
$$;

GRANT CONNECT ON DATABASE warehouse TO mars_reader;
GRANT USAGE ON SCHEMA public TO mars_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mars_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO mars_reader;

INSERT INTO schema_migrations (version) VALUES ('0002_readonly_role')
ON CONFLICT DO NOTHING;
