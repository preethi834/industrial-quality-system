-- Runs once, only when the pg_data volume is first created.
-- Safety net so the extension exists even before migrations run; Alembic
-- (infrastructure/database, revision 0001) is the authority for the schema.
CREATE EXTENSION IF NOT EXISTS vector;
