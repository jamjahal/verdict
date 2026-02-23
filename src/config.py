"""Rubric configuration loader.

Reads ``config/rubric.yaml`` and produces an ``EvalConfig`` object that drives
every configurable knob in the pipeline: dimension weights, pass thresholds,
max retries, and heuristic guard toggles.

Falls back to sensible G-Eval-inspired defaults when the config file is missing
or incomplete, so the library works out-of-the-box with zero configuration.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from .heuristic_guard import HeuristicGuard
from .judge import JudgeOutputParser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults (G-Eval-inspired, Liu et al. 2023)
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "task_completion": 0.30,
    "factual_groundedness": 0.25,
    "coherence": 0.20,
    "relevance": 0.15,
    "safety": 0.10,
}

_DEFAULT_MIN_PASS: dict[str, int] = {
    "task_completion": 3,
    "factual_groundedness": 3,
    "coherence": 3,
    "relevance": 3,
    "safety": 4,
}

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_PASS_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Fully resolved evaluation pipeline configuration.

    Attributes:
        max_retries: Maximum retry iterations before blocking.
        overall_pass_threshold: Minimum weighted score for a PASS verdict.
        weights: Dimension name → weight mapping (must sum to 1.0).
        min_pass_scores: Dimension name → minimum acceptable score (1–5).
        check_pii: Enable PII detection in heuristic guard.
        check_secrets: Enable secrets detection in heuristic guard.
        check_profanity: Enable profanity filter in heuristic guard.
    """

    max_retries: int = _DEFAULT_MAX_RETRIES
    overall_pass_threshold: float = _DEFAULT_PASS_THRESHOLD
    weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    min_pass_scores: dict[str, int] = field(default_factory=lambda: dict(_DEFAULT_MIN_PASS))
    check_pii: bool = True
    check_secrets: bool = True
    check_profanity: bool = True

    def build_guard(self) -> HeuristicGuard:
        """Construct a HeuristicGuard from this config.

        Returns:
            Configured HeuristicGuard instance.
        """
        return HeuristicGuard(
            check_pii=self.check_pii,
            check_secrets=self.check_secrets,
            check_profanity=self.check_profanity,
        )

    def build_parser(self) -> JudgeOutputParser:
        """Construct a JudgeOutputParser from this config.

        Returns:
            Configured JudgeOutputParser with rubric weights.
        """
        return JudgeOutputParser(rubric_weights=dict(self.weights))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> EvalConfig:
    """Load evaluation config from a YAML rubric file.

    If the file does not exist, returns defaults. If the file exists but
    contains invalid values, raises ``ValueError``.

    Args:
        config_path: Path to the rubric YAML file.

    Returns:
        Fully resolved EvalConfig.

    Raises:
        ValueError: If config values are invalid (e.g. weights don't sum
                    to 1.0, scores out of range, max_retries <= 0).
    """
    if not os.path.exists(config_path):
        logger.info("Config file %s not found — using defaults.", config_path)
        return EvalConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # --- Extract values with defaults ---
    max_retries = raw.get("max_retries", _DEFAULT_MAX_RETRIES)
    overall_pass_threshold = raw.get("overall_pass_threshold", _DEFAULT_PASS_THRESHOLD)

    # Dimensions
    dim_raw = raw.get("dimensions", {})
    weights: dict[str, float] = {}
    min_pass_scores: dict[str, int] = {}

    for dim_name, defaults_w in _DEFAULT_WEIGHTS.items():
        dim_config = dim_raw.get(dim_name, {})
        weights[dim_name] = dim_config.get("weight", defaults_w)
        min_pass_scores[dim_name] = dim_config.get(
            "min_pass_score", _DEFAULT_MIN_PASS[dim_name]
        )

    # Heuristic guard
    guard_raw = raw.get("heuristic_guard", {})
    check_pii = guard_raw.get("pii_detection", True)
    check_secrets = guard_raw.get("secrets_detection", True)
    check_profanity = guard_raw.get("profanity_filter", True)

    # --- Validate ---
    _validate(max_retries, overall_pass_threshold, weights, min_pass_scores)

    return EvalConfig(
        max_retries=max_retries,
        overall_pass_threshold=overall_pass_threshold,
        weights=weights,
        min_pass_scores=min_pass_scores,
        check_pii=check_pii,
        check_secrets=check_secrets,
        check_profanity=check_profanity,
    )


def _validate(
    max_retries: int,
    overall_pass_threshold: float,
    weights: dict[str, float],
    min_pass_scores: dict[str, int],
) -> None:
    """Validate config values.

    Args:
        max_retries: Must be > 0.
        overall_pass_threshold: Must be > 0.
        weights: Must sum to 1.0.
        min_pass_scores: Each value must be 1–5.

    Raises:
        ValueError: On any invalid value.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")

    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        raise ValueError(
            f"Dimension weights must sum to 1.0, got {weight_sum:.2f}"
        )

    for dim, score in min_pass_scores.items():
        if not 1 <= score <= 5:
            raise ValueError(
                f"min_pass_score for {dim}={score} out of range (must be 1–5)"
            )
