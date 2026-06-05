"""
Shared configuration — loaded from environment variables (via .env).
All numeric defaults are recorded in DECISIONS.md.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root
load_dotenv(Path(__file__).parent.parent / ".env")


def _require(key: str) -> str:
    v = os.environ.get(key)
    if not v:
        raise RuntimeError(f"Required environment variable {key!r} is not set")
    return v


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_DSN: str = os.environ.get("DB_DSN", "postgresql://user:password@localhost:5432/warehouse")
REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# ---------------------------------------------------------------------------
# LLM — OpenAI is the active provider; Anthropic kept as fallback.
# ---------------------------------------------------------------------------
# Active LLM provider: "openai" | "anthropic" | "mock"
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "openai")

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
# Chat completion model for structured-output agents and the investigator tool loop.
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.0"))

# Anthropic kept as a fallback — not active by default.
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Embeddings: OpenAI text-embedding-3-small (1536 dims).
# EMBEDDING_DIM MUST match the vector(N) column in 0001_initial.sql.
# Switching models requires dropping + recreating the HNSW index and re-embedding.
OPENAI_EMBEDDING_MODEL: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM: int = int(os.environ.get("EMBEDDING_DIM", "1536"))

# Voyage kept as a fallback — not active by default.
VOYAGE_API_KEY: str = os.environ.get("VOYAGE_API_KEY", "")
VOYAGE_EMBEDDING_MODEL: str = os.environ.get("VOYAGE_EMBEDDING_MODEL", "voyage-3")

# ---------------------------------------------------------------------------
# Database — read-only role for investigator tools
# ---------------------------------------------------------------------------
# Connect string for the mars_reader role (read-only).
# Falls back to DB_DSN if not set (loses the physical write guard — only for local dev).
DB_READONLY_DSN: str = os.environ.get(
    "DB_READONLY_DSN",
    os.environ.get("DB_READONLY_DSN", "postgresql://mars_reader:changeme@localhost:5432/warehouse"),
)

# ---------------------------------------------------------------------------
# Investigator tool budget
# ---------------------------------------------------------------------------
INVESTIGATOR_MAX_TOOL_CALLS: int = int(os.environ.get("INVESTIGATOR_MAX_TOOL_CALLS", "6"))
INVESTIGATOR_MAX_ITERATIONS: int = int(os.environ.get("INVESTIGATOR_MAX_ITERATIONS", "5"))
INVESTIGATOR_TIMEOUT_SEC: float = float(os.environ.get("INVESTIGATOR_TIMEOUT_SEC", "30.0"))

# ---------------------------------------------------------------------------
# Aggregator thresholds
# ---------------------------------------------------------------------------
BATTERY_CRITICAL_PCT: float = float(os.environ.get("BATTERY_CRITICAL_PCT", "15.0"))
BATTERY_LOW_PCT: float = float(os.environ.get("BATTERY_LOW_PCT", "30.0"))
BATTERY_MIN_DISPATCH_PCT: float = float(os.environ.get("BATTERY_MIN_DISPATCH_PCT", "20.0"))
BATTERY_TOP_UP_PCT: float = float(os.environ.get("BATTERY_TOP_UP_PCT", "80.0"))
BATTERY_TARGET_CHARGE_PCT: float = float(os.environ.get("BATTERY_TARGET_CHARGE_PCT", "80.0"))

# ---------------------------------------------------------------------------
# Router thresholds
# ---------------------------------------------------------------------------
FAST_PATH_BUDGET: int = int(os.environ.get("FAST_PATH_BUDGET", "2"))
ROUTER_SCOPE_HINT: int = int(os.environ.get("ROUTER_SCOPE_HINT", "2"))
FAILURE_WINDOW_SECONDS: int = int(os.environ.get("FAILURE_WINDOW_SECONDS", "900"))

# ---------------------------------------------------------------------------
# Retrieval Validator weights & floors
# ---------------------------------------------------------------------------
RV_W_META: float = float(os.environ.get("RV_W_META", "0.30"))
RV_W_REC: float = float(os.environ.get("RV_W_REC", "0.20"))
RV_W_COV: float = float(os.environ.get("RV_W_COV", "0.25"))
RV_W_SIM: float = float(os.environ.get("RV_W_SIM", "0.25"))
RV_COV_FLOOR: float = float(os.environ.get("RV_COV_FLOOR", "0.4"))
RV_COV_MISMATCH_CAP: float = float(os.environ.get("RV_COV_MISMATCH_CAP", "0.5"))
RV_RECENCY_FLOOR: float = float(os.environ.get("RV_RECENCY_FLOOR", "0.3"))
RV_STALE_CAP: float = float(os.environ.get("RV_STALE_CAP", "0.4"))
RV_ACCEPT_THRESHOLD: float = float(os.environ.get("RV_ACCEPT_THRESHOLD", "0.5"))
RV_RECENCY_HALF_LIFE_DAYS: float = float(os.environ.get("RV_RECENCY_HALF_LIFE_DAYS", "30.0"))

# ---------------------------------------------------------------------------
# Decision Validator confidence thresholds  (tau per action class)
# ---------------------------------------------------------------------------
DV_TAU_DIAGNOSIS: float = float(os.environ.get("DV_TAU_DIAGNOSIS", "0.5"))
DV_TAU_POLICY_MEDIUM: float = float(os.environ.get("DV_TAU_POLICY_MEDIUM", "0.6"))
DV_TAU_POLICY_HIGH: float = float(os.environ.get("DV_TAU_POLICY_HIGH", "0.75"))

# ---------------------------------------------------------------------------
# Strategy Trigger correlation window
# ---------------------------------------------------------------------------
STRATEGY_CORRELATION_WINDOW_SECONDS: int = int(
    os.environ.get("STRATEGY_CORRELATION_WINDOW_SECONDS", "600")
)

# ---------------------------------------------------------------------------
# Policy Guardrail
# ---------------------------------------------------------------------------
POLICY_WHITELIST: list[str] = [
    "avoid_zone",
    "delay_low_priority_missions",
    "reserve_chargers_for_critical",
    "lower_target_charge_level",
    "pre_charge_for_demand_spike",
]
POLICY_MAX_DURATION_SEC: int = int(os.environ.get("POLICY_MAX_DURATION_SEC", "7200"))
POLICY_MIN_DURATION_SEC: int = int(os.environ.get("POLICY_MIN_DURATION_SEC", "60"))
POLICY_COOLDOWN_SEC: int = int(os.environ.get("POLICY_COOLDOWN_SEC", "120"))

# ---------------------------------------------------------------------------
# Fleet monitoring
# ---------------------------------------------------------------------------
FLEET_MONITOR_INTERVAL_SEC: int = int(os.environ.get("FLEET_MONITOR_INTERVAL_SEC", "300"))
STRATEGY_BACKLOG_THRESHOLD: int = int(os.environ.get("STRATEGY_BACKLOG_THRESHOLD", "20"))
STRATEGY_CONGESTION_THRESHOLD: float = float(os.environ.get("STRATEGY_CONGESTION_THRESHOLD", "0.8"))

# ---------------------------------------------------------------------------
# Outcome Evaluator
# ---------------------------------------------------------------------------
OUTCOME_WINDOW_SEC: int = int(os.environ.get("OUTCOME_WINDOW_SEC", "900"))
