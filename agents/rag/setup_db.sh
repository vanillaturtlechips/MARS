#!/bin/bash
# MARS DB 초기 세팅 — PostgreSQL + pgvector
# 실행: bash agents/rag/setup_db.sh

set -e

echo "[1/4] PostgreSQL 서비스 시작..."
sudo systemctl start postgresql
sudo systemctl enable postgresql

echo "[2/4] DB 사용자 및 데이터베이스 생성..."
sudo -u postgres psql <<EOF
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mars') THEN
    CREATE USER mars WITH PASSWORD 'mars';
  END IF;
END
\$\$;

SELECT 'CREATE DATABASE mars_db OWNER mars'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mars_db')\gexec
GRANT ALL PRIVILEGES ON DATABASE mars_db TO mars;
EOF

echo "[3/4] pgvector 확장 설치..."
sudo -u postgres psql -d mars_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql -d mars_db -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"

echo "[4/4] 스키마 생성..."
PGPASSWORD=mars psql -h localhost -U mars -d mars_db -f "$(dirname "$0")/schema.sql"

echo ""
echo "✅ MARS DB 세팅 완료"
echo "   접속: psql -h localhost -U mars -d mars_db"
echo "   URL:  postgresql://mars:mars@localhost:5432/mars_db"
