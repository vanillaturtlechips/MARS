#!/usr/bin/env bash
# Postgres 14 + pgvector + warehouse DB for the MARS supervisory brain
# (Phase A / full-loop). Native install (no docker). Idempotent.
#
#   bash deploy/runpod/setup_postgres.sh
set -e

DB_USER="${DB_USER:-songhwa}"
DB_PASS="${DB_PASS:-csh110427dg93}"
DB_NAME="${DB_NAME:-warehouse}"

echo "==[1/4] apt packages =="
apt-get update -qq
apt-get install -y postgresql postgresql-server-dev-all build-essential git make gcc >/dev/null

echo "==[2/4] pgvector (build + install) =="
if ! find /usr/lib/postgresql -name 'vector.so' 2>/dev/null | grep -q .; then
    rm -rf /tmp/pgvector
    git clone --branch v0.8.0 --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector
    make -C /tmp/pgvector with_llvm=no
    make -C /tmp/pgvector with_llvm=no install
else
    echo "  vector.so already installed — skipping build"
fi

echo "==[3/4] start postgres =="
service postgresql start || pg_ctlcluster "$(ls /etc/postgresql)" main start || true
sleep 3

echo "==[4/4] role + db + extension (idempotent) =="
su - postgres -c "psql -tc \"SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'\" | grep -q 1" \
    || su - postgres -c "psql -c \"CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}' SUPERUSER;\""
su - postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'\" | grep -q 1" \
    || su - postgres -c "psql -c \"CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};\""
su - postgres -c "psql -d ${DB_NAME} -c 'CREATE EXTENSION IF NOT EXISTS vector;'"

echo
echo "verify:"
su - postgres -c "psql -d ${DB_NAME} -c \"SELECT extname FROM pg_extension WHERE extname='vector';\""
echo "DONE. DSN: postgresql://${DB_USER}:${DB_PASS}@localhost:5432/${DB_NAME}"
