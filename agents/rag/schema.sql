-- MARS RAG Knowledge Base Schema
-- PostgreSQL + pgvector

CREATE EXTENSION IF NOT EXISTS vector;

-- 창고 맵: 선반 위치 + 상품 정보
CREATE TABLE IF NOT EXISTS warehouse_map (
    id          SERIAL PRIMARY KEY,
    shelf_id    VARCHAR(20) UNIQUE NOT NULL,   -- "A-03-12" (구역-열-행)
    zone        VARCHAR(10) NOT NULL,           -- "A", "B", "3구역"
    x           FLOAT NOT NULL,                -- 월드 좌표
    y           FLOAT NOT NULL,
    product_id  VARCHAR(50),
    product_name VARCHAR(100),
    quantity    INT DEFAULT 0,
    embedding   vector(1536),                  -- 상품 설명 임베딩
    updated_at  TIMESTAMP DEFAULT NOW()
);

-- 임무 기록: 성공/실패 패턴 학습
CREATE TABLE IF NOT EXISTS mission_logs (
    id          SERIAL PRIMARY KEY,
    mission_id  UUID DEFAULT gen_random_uuid(),
    robot_id    INT NOT NULL,
    task_type   VARCHAR(50) NOT NULL,          -- "pickup", "delivery", "navigation"
    from_shelf  VARCHAR(20),
    to_location VARCHAR(50),
    success     BOOLEAN NOT NULL,
    duration_s  FLOAT,                         -- 소요 시간 (초)
    fail_reason VARCHAR(200),
    embedding   vector(1536),                  -- 임무 설명 임베딩 (유사 임무 검색용)
    created_at  TIMESTAMP DEFAULT NOW()
);

-- 행동 라이브러리: 검증된 시퀀스
CREATE TABLE IF NOT EXISTS action_library (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    task_type   VARCHAR(50),
    steps       JSONB NOT NULL,                -- 행동 시퀀스
    success_rate FLOAT DEFAULT 0.0,
    use_count   INT DEFAULT 0,
    embedding   vector(1536),
    created_at  TIMESTAMP DEFAULT NOW()
);

-- 로봇 상태: 실시간 현황
CREATE TABLE IF NOT EXISTS robot_status (
    robot_id    INT PRIMARY KEY,
    name        VARCHAR(50),
    x           FLOAT,
    y           FLOAT,
    battery_pct FLOAT,
    status      VARCHAR(20) DEFAULT 'idle',    -- "idle", "busy", "charging", "error"
    current_mission UUID,
    updated_at  TIMESTAMP DEFAULT NOW()
);

-- 인덱스: 벡터 검색 속도
CREATE INDEX IF NOT EXISTS idx_warehouse_embedding
    ON warehouse_map USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_mission_embedding
    ON mission_logs USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_action_embedding
    ON action_library USING ivfflat (embedding vector_cosine_ops);

-- 인덱스: SQL 필터 속도
CREATE INDEX IF NOT EXISTS idx_warehouse_zone ON warehouse_map (zone);
CREATE INDEX IF NOT EXISTS idx_mission_robot  ON mission_logs (robot_id, success);
CREATE INDEX IF NOT EXISTS idx_mission_time   ON mission_logs (created_at DESC);
