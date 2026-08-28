"""Scoring for the interviewer hire-decision benchmark.

Pipeline: model outputs a total score ``<score>X.X</score>`` → convert to a
zone (hire / middle / reject) → compare with the gold zone derived from the
``result`` column (通过→hire, 不通過→reject).

Reward (single unified ``hard`` signal, no within-zone smoothing):
* decisive hit  (zone == gold zone)            → 1.0
* middle zone   (4 < score < 6.5)              → 0.3
* opposite zone / parse failure                → 0.0

``soft`` returns the same value as ``hard`` to satisfy the trainer contract
(``id`` / ``hard`` / ``soft``); it carries no independent semantics.
"""
from __future__ import annotations

import re

SCORE_RE = re.compile(r"<score>\s*([0-9]+(?:\.[0-9]+)?)\s*</score>", re.IGNORECASE)

HIRE_LABELS = ("通过", "录用", "hire", "pass")
REJECT_LABELS = ("不通過", "不录用", "reject", "fail")

MIDDLE_REWARD = 0.3


def parse_score(text: str | None) -> float | None:
    """Extract the total score from the model output, or None if absent."""
    if not text:
        return None
    match = SCORE_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def score_to_zone(score: float) -> str:
    """Map a total score (0~8) to hire / middle / reject per the fixed table."""
    if score >= 6.5:
        return "hire"
    if score <= 4.0:
        return "reject"
    return "middle"


def gold_to_zone(label: str | None) -> str | None:
    """Map the raw ``result`` label to its decisive zone."""
    if not label:
        return None
    text = str(label).strip()
    if text in HIRE_LABELS:
        return "hire"
    if text in REJECT_LABELS:
        return "reject"
    return None


def score_episode(prediction: str | None, ground_truth: str | None) -> tuple[float, float]:
    """Return ``(hard, soft)`` for one episode."""
    score = parse_score(prediction)
    if score is None:
        return 0.0, 0.0
    pred_zone = score_to_zone(score)
    if pred_zone == "middle":
        return MIDDLE_REWARD, MIDDLE_REWARD
    gold_zone = gold_to_zone(ground_truth)
    if gold_zone is None:
        return 0.0, 0.0
    if pred_zone == gold_zone:
        return 1.0, 1.0
    return 0.0, 0.0
