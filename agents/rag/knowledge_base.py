"""MARS RAG Knowledge Base — pgvector 기반 창고 지식 저장소."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import psycopg2
from pgvector.psycopg2 import register_vector
from sqlalchemy import create_engine, text

DB_URL = os.environ.get(
    "MARS_DB_URL",
    "postgresql://mars:mars@localhost:5432/mars_db"
)


# ── 연결 ──────────────────────────────────────────────────────────────

def get_conn():
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    return conn

def get_engine():
    return create_engine(DB_URL)


# ── 창고 맵 ───────────────────────────────────────────────────────────

@dataclass
class ShelfInfo:
    shelf_id: str
    zone: str
    x: float
    y: float
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    quantity: int = 0


def upsert_shelf(shelf: ShelfInfo, embedding: list[float] | None = None):
    """선반 정보 추가/업데이트."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO warehouse_map
                    (shelf_id, zone, x, y, product_id, product_name, quantity, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (shelf_id) DO UPDATE SET
                    zone = EXCLUDED.zone,
                    x = EXCLUDED.x, y = EXCLUDED.y,
                    product_id = EXCLUDED.product_id,
                    product_name = EXCLUDED.product_name,
                    quantity = EXCLUDED.quantity,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW()
            """, (
                shelf.shelf_id, shelf.zone, shelf.x, shelf.y,
                shelf.product_id, shelf.product_name, shelf.quantity,
                embedding
            ))
        conn.commit()


def find_shelf_by_product(query_embedding: list[float], zone: str | None = None, limit: int = 5) -> list[dict]:
    """상품 설명 벡터로 선반 검색. zone 필터 선택."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            if zone:
                cur.execute("""
                    SELECT shelf_id, zone, x, y, product_name, quantity,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM warehouse_map
                    WHERE zone = %s AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, zone, query_embedding, limit))
            else:
                cur.execute("""
                    SELECT shelf_id, zone, x, y, product_name, quantity,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM warehouse_map
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, limit))

            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_shelves_in_zone(zone: str) -> list[dict]:
    """특정 구역의 전체 선반 목록 (순수 SQL)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT shelf_id, x, y, product_name, quantity
                FROM warehouse_map
                WHERE zone = %s
                ORDER BY shelf_id
            """, (zone,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


# ── 임무 기록 ─────────────────────────────────────────────────────────

def log_mission(robot_id: int, task_type: str, from_shelf: str,
                to_location: str, success: bool, duration_s: float,
                fail_reason: str | None = None,
                embedding: list[float] | None = None):
    """임무 결과 기록."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO mission_logs
                    (robot_id, task_type, from_shelf, to_location,
                     success, duration_s, fail_reason, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (robot_id, task_type, from_shelf, to_location,
                  success, duration_s, fail_reason, embedding))
        conn.commit()


def find_similar_missions(query_embedding: list[float],
                          robot_id: int | None = None,
                          success_only: bool = False,
                          limit: int = 5) -> list[dict]:
    """유사 임무 이력 검색. SQL 필터 + 벡터 검색 혼합."""
    filters = ["embedding IS NOT NULL"]
    params: list = [query_embedding]

    if robot_id is not None:
        filters.append(f"robot_id = %s")
        params.append(robot_id)
    if success_only:
        filters.append("success = true")

    where = "WHERE " + " AND ".join(filters)
    params += [query_embedding, limit]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT robot_id, task_type, from_shelf, to_location,
                       success, duration_s, fail_reason, created_at,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM mission_logs
                {where}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_robot_fail_patterns(robot_id: int, limit: int = 10) -> list[dict]:
    """특정 로봇의 최근 실패 패턴 (순수 SQL)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT task_type, from_shelf, fail_reason, created_at
                FROM mission_logs
                WHERE robot_id = %s AND success = false
                ORDER BY created_at DESC
                LIMIT %s
            """, (robot_id, limit))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


# ── 행동 라이브러리 ───────────────────────────────────────────────────

def find_action(query_embedding: list[float], task_type: str | None = None,
                limit: int = 3) -> list[dict]:
    """유사 행동 시퀀스 검색."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            if task_type:
                cur.execute("""
                    SELECT name, description, steps, success_rate, use_count,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM action_library
                    WHERE task_type = %s AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, task_type, query_embedding, limit))
            else:
                cur.execute("""
                    SELECT name, description, steps, success_rate, use_count,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM action_library
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, limit))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


# ── 로봇 상태 ─────────────────────────────────────────────────────────

def update_robot_status(robot_id: int, x: float, y: float,
                         battery_pct: float, status: str,
                         current_mission: str | None = None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO robot_status
                    (robot_id, x, y, battery_pct, status, current_mission)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (robot_id) DO UPDATE SET
                    x = EXCLUDED.x, y = EXCLUDED.y,
                    battery_pct = EXCLUDED.battery_pct,
                    status = EXCLUDED.status,
                    current_mission = EXCLUDED.current_mission,
                    updated_at = NOW()
            """, (robot_id, x, y, battery_pct, status, current_mission))
        conn.commit()


def get_available_robots() -> list[dict]:
    """현재 유휴 상태 로봇 목록."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT robot_id, x, y, battery_pct
                FROM robot_status
                WHERE status = 'idle' AND battery_pct > 20
                ORDER BY battery_pct DESC
            """)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


# ── DB 초기화 ─────────────────────────────────────────────────────────

def init_db():
    """스키마 생성 (schema.sql 실행)."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path) as f:
        sql = f.read()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    print("[mars/rag] DB 초기화 완료")


if __name__ == "__main__":
    init_db()
