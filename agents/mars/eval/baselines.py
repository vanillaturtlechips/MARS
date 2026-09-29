"""Baselines that need no model, reported on every diagnosis run.

An accuracy figure means nothing without the floor it stands on. Two floors have
already moved the interpretation of the same number:

  - a leaked label. `search_incidents` used to return the past incident's
    `failure_type`, which for a relevant precedent is the answer to the case in
    front of the agent. Copying the top-trust precedent's label scored 68%
    against the model's 85%.
  - structured fields. Battery percentage, an e-stop flag, fault codes, how many
    failures were seeded and whether any precedent exists separate 44 of the 100
    test cases outright: every low_battery case sits at 6-11% battery while every
    other case is at 58% or above; every `unknown` case has no precedent; every
    zone_blocked case has two or more seeded failures. A five-line rule over those
    fields, with no text and no model, scores 54%.

Neither is a defect on its own — a real fleet also has cases where the battery
reading settles it. The defect is reporting 71% as if it measured reasoning over
evidence when a threshold reaches 54% of it. Both floors are therefore printed on
every run, so a change in the dataset or the tools shows up as a moving floor
instead of a quietly inflated headline.

The rule below was written against the DEV split only and is frozen. It scores
54% on dev and 54% on test, so it is not fitted to either.
"""
from __future__ import annotations

from typing import Any


def _feats(case: dict[str, Any]) -> tuple:
    t = case["trigger_event"]
    h = t.get("health_at_failure", {}) or {}
    return (h.get("battery_pct"), bool(h.get("estop_active")),
            bool(h.get("fault_codes")),
            len(case["seed_state"].get("failures", []) or []),
            len(case["seed_state"].get("incidents", []) or []))


def rule_baseline(case: dict[str, Any]) -> str:
    """Cause from structured bundle fields alone — no text, no precedent, no LLM.

    Frozen after being written on dev. Each branch is a threshold that fully
    separates one cause in this dataset; the causes it cannot reach
    (robot_internal_fault, zone_congestion) are exactly the ones that require
    reading the precedent text, and they are where the model's margin comes from.
    """
    batt, _estop, faults, n_failures, n_incidents = _feats(case)
    if batt is not None and batt < 20:
        return "low_battery"
    if n_incidents == 0:
        return "unknown"
    if n_failures >= 2:
        return "zone_blocked"
    if faults:
        return "robot_internal_fault"
    return "localization_failure"


def rule_accuracy(cases: list[dict[str, Any]]) -> tuple[int, int]:
    hit = sum(rule_baseline(c) == c["ground_truth"]["cause"] for c in cases)
    return hit, len(cases)


def separable(cases: list[dict[str, Any]]) -> tuple[int, int]:
    """Cases whose cause the rule fixes with certainty in this dataset.

    Reported alongside the easy/medium/hard tags because the two disagree: 22
    cases are tagged easy while 44 fall to a threshold.
    """
    fixed = {"low_battery", "unknown", "zone_blocked"}
    n = sum(1 for c in cases
            if rule_baseline(c) in fixed and rule_baseline(c) == c["ground_truth"]["cause"])
    return n, len(cases)
