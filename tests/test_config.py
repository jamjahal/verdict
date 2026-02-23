"""Tests for config.py — Rubric configuration loader.

TDD: These tests define the interface and expected behaviour.
All tests will FAIL until src/config.py is implemented (Red phase).
"""

import os
import tempfile

import pytest
import yaml

from src.config import EvalConfig, load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_RUBRIC = {
    "max_retries": 3,
    "overall_pass_threshold": 3.0,
    "dimensions": {
        "task_completion": {"weight": 0.30, "min_pass_score": 3},
        "factual_groundedness": {"weight": 0.25, "min_pass_score": 3},
        "coherence": {"weight": 0.20, "min_pass_score": 3},
        "relevance": {"weight": 0.15, "min_pass_score": 3},
        "safety": {"weight": 0.10, "min_pass_score": 4},
    },
    "heuristic_guard": {
        "pii_detection": True,
        "secrets_detection": True,
        "profanity_filter": True,
        "format_validation": True,
    },
}

CUSTOM_RUBRIC = {
    "max_retries": 5,
    "overall_pass_threshold": 3.5,
    "dimensions": {
        "task_completion": {"weight": 0.40, "min_pass_score": 4},
        "factual_groundedness": {"weight": 0.25, "min_pass_score": 3},
        "coherence": {"weight": 0.15, "min_pass_score": 3},
        "relevance": {"weight": 0.10, "min_pass_score": 2},
        "safety": {"weight": 0.10, "min_pass_score": 4},
    },
    "heuristic_guard": {
        "pii_detection": True,
        "secrets_detection": False,
        "profanity_filter": True,
        "format_validation": False,
    },
}


@pytest.fixture
def valid_rubric_path(tmp_path):
    """Write a valid rubric YAML and return its path."""
    path = tmp_path / "rubric.yaml"
    with open(path, "w") as f:
        yaml.dump(VALID_RUBRIC, f)
    return str(path)


@pytest.fixture
def custom_rubric_path(tmp_path):
    """Write a custom rubric YAML and return its path."""
    path = tmp_path / "rubric.yaml"
    with open(path, "w") as f:
        yaml.dump(CUSTOM_RUBRIC, f)
    return str(path)


# ---------------------------------------------------------------------------
# Loading from YAML
# ---------------------------------------------------------------------------

class TestLoadConfig:

    def test_load_returns_eval_config(self, valid_rubric_path: str) -> None:
        config = load_config(valid_rubric_path)
        assert isinstance(config, EvalConfig)

    def test_load_reads_max_retries(self, valid_rubric_path: str) -> None:
        config = load_config(valid_rubric_path)
        assert config.max_retries == 3

    def test_load_reads_pass_threshold(self, valid_rubric_path: str) -> None:
        config = load_config(valid_rubric_path)
        assert config.overall_pass_threshold == 3.0

    def test_load_reads_dimension_weights(self, valid_rubric_path: str) -> None:
        config = load_config(valid_rubric_path)
        assert config.weights["task_completion"] == 0.30
        assert config.weights["safety"] == 0.10

    def test_load_reads_min_pass_scores(self, valid_rubric_path: str) -> None:
        config = load_config(valid_rubric_path)
        assert config.min_pass_scores["task_completion"] == 3
        assert config.min_pass_scores["safety"] == 4

    def test_load_reads_heuristic_settings(self, valid_rubric_path: str) -> None:
        config = load_config(valid_rubric_path)
        assert config.check_pii is True
        assert config.check_secrets is True
        assert config.check_profanity is True

    def test_load_custom_overrides(self, custom_rubric_path: str) -> None:
        config = load_config(custom_rubric_path)
        assert config.max_retries == 5
        assert config.overall_pass_threshold == 3.5
        assert config.weights["task_completion"] == 0.40
        assert config.check_secrets is False


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

class TestDefaultConfig:

    def test_default_config_when_file_missing(self) -> None:
        config = load_config("/nonexistent/path/rubric.yaml")
        assert isinstance(config, EvalConfig)
        assert config.max_retries == 3

    def test_default_weights_sum_to_one(self) -> None:
        config = load_config("/nonexistent/path/rubric.yaml")
        assert sum(config.weights.values()) == pytest.approx(1.0)

    def test_default_has_all_dimensions(self) -> None:
        config = load_config("/nonexistent/path/rubric.yaml")
        expected = {"task_completion", "factual_groundedness", "coherence", "relevance", "safety"}
        assert set(config.weights.keys()) == expected


# ---------------------------------------------------------------------------
# Building pipeline components from config
# ---------------------------------------------------------------------------

class TestConfigBuildComponents:

    def test_build_guard(self, valid_rubric_path: str) -> None:
        from src.heuristic_guard import HeuristicGuard
        config = load_config(valid_rubric_path)
        guard = config.build_guard()
        assert isinstance(guard, HeuristicGuard)
        assert guard.check_pii is True
        assert guard.check_secrets is True

    def test_build_guard_with_disabled_checks(self, custom_rubric_path: str) -> None:
        from src.heuristic_guard import HeuristicGuard
        config = load_config(custom_rubric_path)
        guard = config.build_guard()
        assert guard.check_secrets is False

    def test_build_parser(self, valid_rubric_path: str) -> None:
        from src.judge import JudgeOutputParser
        config = load_config(valid_rubric_path)
        parser = config.build_parser()
        assert isinstance(parser, JudgeOutputParser)
        assert parser.weights["task_completion"] == 0.30

    def test_build_parser_custom_weights(self, custom_rubric_path: str) -> None:
        from src.judge import JudgeOutputParser
        config = load_config(custom_rubric_path)
        parser = config.build_parser()
        assert parser.weights["task_completion"] == 0.40


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestConfigValidation:

    def test_weights_must_sum_to_one(self, tmp_path) -> None:
        bad = VALID_RUBRIC.copy()
        bad["dimensions"] = {
            "task_completion": {"weight": 0.50, "min_pass_score": 3},
            "factual_groundedness": {"weight": 0.50, "min_pass_score": 3},
            "coherence": {"weight": 0.50, "min_pass_score": 3},
            "relevance": {"weight": 0.50, "min_pass_score": 3},
            "safety": {"weight": 0.50, "min_pass_score": 4},
        }
        path = tmp_path / "rubric.yaml"
        with open(path, "w") as f:
            yaml.dump(bad, f)
        with pytest.raises(ValueError, match="sum to 1.0"):
            load_config(str(path))

    def test_min_pass_score_must_be_1_to_5(self, tmp_path) -> None:
        bad = VALID_RUBRIC.copy()
        bad["dimensions"] = {
            "task_completion": {"weight": 0.30, "min_pass_score": 6},
            "factual_groundedness": {"weight": 0.25, "min_pass_score": 3},
            "coherence": {"weight": 0.20, "min_pass_score": 3},
            "relevance": {"weight": 0.15, "min_pass_score": 3},
            "safety": {"weight": 0.10, "min_pass_score": 4},
        }
        path = tmp_path / "rubric.yaml"
        with open(path, "w") as f:
            yaml.dump(bad, f)
        with pytest.raises(ValueError, match="range"):
            load_config(str(path))

    def test_max_retries_must_be_positive(self, tmp_path) -> None:
        bad = VALID_RUBRIC.copy()
        bad["max_retries"] = 0
        path = tmp_path / "rubric.yaml"
        with open(path, "w") as f:
            yaml.dump(bad, f)
        with pytest.raises(ValueError, match="max_retries"):
            load_config(str(path))
