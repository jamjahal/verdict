"""End-to-end integration tests for the full eval pipeline.

These tests wire all components together — config loading, heuristic guard,
judge parser, retry loop, and logger — proving the pipeline works as a unit
with no mocks on internal components (only the agent and judge callables).
"""

import json
import os

import pytest
import yaml

from src.config import EvalConfig, load_config
from src.heuristic_guard import HeuristicGuard
from src.judge import JudgeOutputParser
from src.logger import EvalLogger
from src.retry_loop import RetryLoop


# ---------------------------------------------------------------------------
# Judge output fixtures
# ---------------------------------------------------------------------------

PASS_JUDGE_OUTPUT = """
EVAL_RESULT
task_completion: 4
factual_groundedness: 4
coherence: 5
relevance: 4
safety: 5
weighted_score: 4.25
verdict: PASS
critique: |
  Good response.
END_EVAL_RESULT
"""

FAIL_JUDGE_OUTPUT = """
EVAL_RESULT
task_completion: 2
factual_groundedness: 2
coherence: 3
relevance: 2
safety: 5
weighted_score: 2.35
verdict: FAIL
critique: |
  The response missed the core task. On retry, answer the question directly.
END_EVAL_RESULT
"""


# ---------------------------------------------------------------------------
# Config-driven pipeline
# ---------------------------------------------------------------------------

class TestConfigDrivenPipeline:
    """Pipeline built from rubric.yaml config, not hardcoded defaults."""

    def test_pipeline_from_real_rubric_yaml(self) -> None:
        """Load the actual config/rubric.yaml and build a working pipeline."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config", "rubric.yaml",
        )
        config = load_config(config_path)

        agent_fn = lambda prompt, critique: "The capital of France is Paris."
        judge_fn = lambda task, output: PASS_JUDGE_OUTPUT

        guard = config.build_guard()
        parser = config.build_parser()

        loop = RetryLoop(
            agent_fn=agent_fn,
            judge_fn=judge_fn,
            guard=guard,
            parser=parser,
            max_retries=config.max_retries,
        )
        result = loop.run("What is the capital of France?")

        assert result.passed is True
        assert result.attempts == 1
        assert result.blocked is False

    def test_custom_config_changes_behavior(self, tmp_path) -> None:
        """Custom rubric with max_retries=1 blocks faster."""
        rubric = {
            "max_retries": 1,
            "overall_pass_threshold": 3.0,
            "dimensions": {
                "task_completion": {"weight": 0.30, "min_pass_score": 3},
                "factual_groundedness": {"weight": 0.25, "min_pass_score": 3},
                "coherence": {"weight": 0.20, "min_pass_score": 3},
                "relevance": {"weight": 0.15, "min_pass_score": 3},
                "safety": {"weight": 0.10, "min_pass_score": 4},
            },
        }
        path = tmp_path / "rubric.yaml"
        with open(path, "w") as f:
            yaml.dump(rubric, f)

        config = load_config(str(path))
        assert config.max_retries == 1

        agent_fn = lambda prompt, critique: "Bad output."
        judge_fn = lambda task, output: FAIL_JUDGE_OUTPUT

        loop = RetryLoop(
            agent_fn=agent_fn,
            judge_fn=judge_fn,
            guard=config.build_guard(),
            parser=config.build_parser(),
            max_retries=config.max_retries,
        )
        result = loop.run("task")

        assert result.blocked is True
        assert result.attempts == 1


# ---------------------------------------------------------------------------
# Full pipeline: guard → judge → retry → logger
# ---------------------------------------------------------------------------

class TestFullPipelineWithLogger:
    """End-to-end: agent output → guard → judge → retry → log to JSONL."""

    def test_pass_on_first_attempt_logged(self, tmp_path) -> None:
        log_path = str(tmp_path / "eval_log.jsonl")
        logger = EvalLogger(log_path=log_path)

        agent_fn = lambda prompt, critique: "Clean, safe response about Python."
        judge_fn = lambda task, output: PASS_JUDGE_OUTPUT

        loop = RetryLoop(agent_fn=agent_fn, judge_fn=judge_fn)
        result = loop.run("Explain Python.")

        logger.log(task_prompt="Explain Python.", result=result)

        events = logger.read_events()
        assert len(events) == 1
        assert events[0]["passed"] is True
        assert events[0]["attempts"] == 1
        assert events[0]["scores"]["task_completion"] == 4

    def test_retry_then_pass_logged(self, tmp_path) -> None:
        log_path = str(tmp_path / "eval_log.jsonl")
        logger = EvalLogger(log_path=log_path)

        call_count = {"n": 0}

        def agent_fn(prompt, critique):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "Incomplete answer."
            return "Complete answer about Python."

        def judge_fn(task, output):
            if "Incomplete" in output:
                return FAIL_JUDGE_OUTPUT
            return PASS_JUDGE_OUTPUT

        loop = RetryLoop(agent_fn=agent_fn, judge_fn=judge_fn)
        result = loop.run("Explain Python.")

        logger.log(task_prompt="Explain Python.", result=result)

        events = logger.read_events()
        assert len(events) == 1
        assert events[0]["passed"] is True
        assert events[0]["attempts"] == 2

    def test_heuristic_block_then_pass_logged(self, tmp_path) -> None:
        """Guard blocks PII on first attempt, clean output passes on second."""
        log_path = str(tmp_path / "eval_log.jsonl")
        logger = EvalLogger(log_path=log_path)

        call_count = {"n": 0}

        def agent_fn(prompt, critique):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "Contact user@example.com for details."
            return "Contact our support team for details."

        judge_fn = lambda task, output: PASS_JUDGE_OUTPUT

        loop = RetryLoop(agent_fn=agent_fn, judge_fn=judge_fn)
        result = loop.run("How to contact support?")

        logger.log(task_prompt="How to contact support?", result=result)

        events = logger.read_events()
        assert events[0]["passed"] is True
        assert events[0]["attempts"] == 2

    def test_full_block_after_max_retries_logged(self, tmp_path) -> None:
        log_path = str(tmp_path / "eval_log.jsonl")
        logger = EvalLogger(log_path=log_path)

        agent_fn = lambda prompt, critique: "Still a bad response."
        judge_fn = lambda task, output: FAIL_JUDGE_OUTPUT

        loop = RetryLoop(agent_fn=agent_fn, judge_fn=judge_fn, max_retries=2)
        result = loop.run("Explain quantum computing.")

        logger.log(task_prompt="Explain quantum computing.", result=result)

        events = logger.read_events()
        assert events[0]["passed"] is False
        assert events[0]["blocked"] is True
        assert events[0]["attempts"] == 2
        assert events[0]["failure_reason"] is not None

    def test_multiple_evals_logged_sequentially(self, tmp_path) -> None:
        """Multiple pipeline runs produce multiple log entries."""
        log_path = str(tmp_path / "eval_log.jsonl")
        logger = EvalLogger(log_path=log_path)

        agent_fn = lambda prompt, critique: "Good response."
        judge_fn = lambda task, output: PASS_JUDGE_OUTPUT
        loop = RetryLoop(agent_fn=agent_fn, judge_fn=judge_fn)

        for task in ["Task A", "Task B", "Task C"]:
            result = loop.run(task)
            logger.log(task_prompt=task, result=result)

        events = logger.read_events()
        assert len(events) == 3
        assert [e["task_prompt"] for e in events] == ["Task A", "Task B", "Task C"]


# ---------------------------------------------------------------------------
# Guard config integration
# ---------------------------------------------------------------------------

class TestGuardConfigIntegration:

    def test_disabled_secrets_check_allows_api_key(self, tmp_path) -> None:
        """Config with secrets_detection=false lets API keys through."""
        rubric = {
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
                "secrets_detection": False,
                "profanity_filter": True,
            },
        }
        path = tmp_path / "rubric.yaml"
        with open(path, "w") as f:
            yaml.dump(rubric, f)

        config = load_config(str(path))
        guard = config.build_guard()

        # API key should pass when secrets detection is off
        result = guard.evaluate("Use key AKIAIOSFODNN7EXAMPLE")
        assert result.passed is True

    def test_enabled_secrets_check_blocks_api_key(self) -> None:
        """Default config blocks API keys."""
        config = EvalConfig()
        guard = config.build_guard()
        result = guard.evaluate("Use key AKIAIOSFODNN7EXAMPLE")
        assert result.passed is False
        assert "secrets" in result.failed_checks
