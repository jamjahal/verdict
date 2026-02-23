"""Tests for hooks/scripts/judge_eval.py - PostToolUse autonomous quality hook.

TDD: These tests define the interface and expected behaviour.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

# The hook script lives outside src/, so we import run_eval directly.
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK_SCRIPTS_DIR = os.path.join(REPO_ROOT, "hooks", "scripts")
sys.path.insert(0, HOOK_SCRIPTS_DIR)

from judge_eval import run_eval  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_RUBRIC = {
    "max_retries": 3,
    "overall_pass_threshold": 3.0,
    "judge_model": "claude-sonnet-4-5-20250929",
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
    },
}


@pytest.fixture
def config_path(tmp_path):
    """Write a valid rubric and return its path."""
    path = tmp_path / "rubric.yaml"
    with open(path, "w") as f:
        yaml.dump(VALID_RUBRIC, f)
    return str(path)


def _make_hook_input(
    tool_result: str = "This is a long enough agent output for quality assessment purposes. " * 5,
    task_prompt: str = "Write a summary of the meeting notes.",
) -> dict:
    """Build a realistic PostToolUse hook input dict."""
    return {
        "session_id": "test-session-123",
        "transcript_path": "/tmp/transcript.txt",
        "cwd": "/tmp",
        "hook_event_name": "PostToolUse",
        "tool_name": "Task",
        "tool_input": {
            "prompt": task_prompt,
            "subagent_type": "general-purpose",
            "description": "Summarise meeting notes",
        },
        "tool_result": tool_result,
    }


MOCK_PASS_RESPONSE = """EVAL_RESULT
task_completion: 4
factual_groundedness: 4
coherence: 5
relevance: 4
safety: 5
weighted_score: 4.25
verdict: PASS
critique: |
  Solid output.
END_EVAL_RESULT"""


MOCK_FAIL_RESPONSE = """EVAL_RESULT
task_completion: 2
factual_groundedness: 2
coherence: 3
relevance: 2
safety: 5
weighted_score: 2.35
verdict: FAIL
critique: |
  The output did not address the core request.
  Please re-read the task and include all required sections.
END_EVAL_RESULT"""


# ---------------------------------------------------------------------------
# Skip short / trivial outputs
# ---------------------------------------------------------------------------

class TestSkipTrivialOutputs:

    def test_skip_empty_tool_result(self, config_path: str) -> None:
        hook_input = _make_hook_input(tool_result="")
        result = run_eval(hook_input, config_path)
        assert result is None

    def test_skip_short_tool_result(self, config_path: str) -> None:
        hook_input = _make_hook_input(tool_result="OK")
        result = run_eval(hook_input, config_path)
        assert result is None

    def test_skip_under_50_chars(self, config_path: str) -> None:
        hook_input = _make_hook_input(tool_result="A" * 49)
        result = run_eval(hook_input, config_path)
        assert result is None


# ---------------------------------------------------------------------------
# Layer 1: Heuristic guard failures
# ---------------------------------------------------------------------------

class TestLayer1HeuristicGuard:

    def test_pii_detected_returns_fail_message(self, config_path: str) -> None:
        output_with_pii = (
            "The user's SSN is 123-45-6789 and their email is test@example.com. " * 3
        )
        hook_input = _make_hook_input(tool_result=output_with_pii)
        result = run_eval(hook_input, config_path)

        assert result is not None
        assert "systemMessage" in result
        assert "Layer 1" in result["systemMessage"] or "Heuristic Guard" in result["systemMessage"]
        assert "FAIL" in result["systemMessage"].upper()

    def test_secrets_detected_returns_fail_message(self, config_path: str) -> None:
        output_with_secrets = (
            "Here is the API key: sk-1234567890abcdef1234567890abcdef " * 3
        )
        hook_input = _make_hook_input(tool_result=output_with_secrets)
        result = run_eval(hook_input, config_path)

        assert result is not None
        assert "systemMessage" in result

    def test_guard_fail_does_not_call_api(self, config_path: str) -> None:
        output_with_pii = (
            "The user's SSN is 123-45-6789. " * 5
        )
        hook_input = _make_hook_input(tool_result=output_with_pii)

        with patch("judge_eval.call_judge") as mock_call:
            run_eval(hook_input, config_path)
            mock_call.assert_not_called()


# ---------------------------------------------------------------------------
# Layer 2: No API key
# ---------------------------------------------------------------------------

class TestNoApiKey:

    def test_missing_api_key_returns_skip_message(self, config_path: str) -> None:
        hook_input = _make_hook_input()
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)

        with patch.dict(os.environ, env, clear=True):
            result = run_eval(hook_input, config_path)

        assert result is not None
        assert "systemMessage" in result
        assert "SKIP" in result["systemMessage"].upper()


# ---------------------------------------------------------------------------
# Layer 2: PASS verdict
# ---------------------------------------------------------------------------

class TestLayer2Pass:

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("judge_eval.call_judge")
    def test_pass_verdict_returns_pass_message(self, mock_call_judge, config_path: str) -> None:
        mock_call_judge.return_value = MOCK_PASS_RESPONSE
        hook_input = _make_hook_input()

        result = run_eval(hook_input, config_path)

        assert result is not None
        assert "systemMessage" in result
        assert "PASS" in result["systemMessage"].upper()
        assert "4.25" in result["systemMessage"]

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("judge_eval.call_judge")
    def test_pass_does_not_include_retry_instruction(self, mock_call_judge, config_path: str) -> None:
        mock_call_judge.return_value = MOCK_PASS_RESPONSE
        hook_input = _make_hook_input()

        result = run_eval(hook_input, config_path)

        assert "retry" not in result["systemMessage"].lower()
        assert "revise" not in result["systemMessage"].lower()


# ---------------------------------------------------------------------------
# Layer 2: FAIL verdict
# ---------------------------------------------------------------------------

class TestLayer2Fail:

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("judge_eval.call_judge")
    def test_fail_verdict_returns_fail_message(self, mock_call_judge, config_path: str) -> None:
        mock_call_judge.return_value = MOCK_FAIL_RESPONSE
        hook_input = _make_hook_input()

        result = run_eval(hook_input, config_path)

        assert result is not None
        assert "systemMessage" in result
        assert "FAIL" in result["systemMessage"].upper()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("judge_eval.call_judge")
    def test_fail_includes_critique(self, mock_call_judge, config_path: str) -> None:
        mock_call_judge.return_value = MOCK_FAIL_RESPONSE
        hook_input = _make_hook_input()

        result = run_eval(hook_input, config_path)

        assert "did not address" in result["systemMessage"]

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("judge_eval.call_judge")
    def test_fail_includes_retry_instruction(self, mock_call_judge, config_path: str) -> None:
        mock_call_judge.return_value = MOCK_FAIL_RESPONSE
        hook_input = _make_hook_input()

        result = run_eval(hook_input, config_path)

        msg = result["systemMessage"].lower()
        assert "revise" in msg or "retry" in msg or "try again" in msg

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("judge_eval.call_judge")
    def test_fail_includes_score(self, mock_call_judge, config_path: str) -> None:
        mock_call_judge.return_value = MOCK_FAIL_RESPONSE
        hook_input = _make_hook_input()

        result = run_eval(hook_input, config_path)

        assert "2.35" in result["systemMessage"]


# ---------------------------------------------------------------------------
# Layer 2: API errors handled gracefully
# ---------------------------------------------------------------------------

class TestApiErrorHandling:

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("judge_eval.call_judge")
    def test_api_error_returns_error_message(self, mock_call_judge, config_path: str) -> None:
        mock_call_judge.side_effect = Exception("Connection timeout")
        hook_input = _make_hook_input()

        result = run_eval(hook_input, config_path)

        assert result is not None
        assert "systemMessage" in result
        assert "ERROR" in result["systemMessage"].upper()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("judge_eval.call_judge")
    def test_parse_error_returns_error_message(self, mock_call_judge, config_path: str) -> None:
        mock_call_judge.return_value = "This is not a valid block"
        hook_input = _make_hook_input()

        result = run_eval(hook_input, config_path)

        assert result is not None
        assert "systemMessage" in result
        assert "ERROR" in result["systemMessage"].upper()


# ---------------------------------------------------------------------------
# call_judge receives correct arguments
# ---------------------------------------------------------------------------

class TestCallJudgeArguments:

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("judge_eval.call_judge")
    def test_passes_task_prompt_from_tool_input(self, mock_call_judge, config_path: str) -> None:
        mock_call_judge.return_value = MOCK_PASS_RESPONSE
        hook_input = _make_hook_input(task_prompt="Explain quantum computing.")

        run_eval(hook_input, config_path)

        call_args = mock_call_judge.call_args
        assert "quantum computing" in str(call_args)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("judge_eval.call_judge")
    def test_passes_agent_output_from_tool_result(self, mock_call_judge, config_path: str) -> None:
        mock_call_judge.return_value = MOCK_PASS_RESPONSE
        agent_output = "Quantum computing uses qubits. " * 5
        hook_input = _make_hook_input(tool_result=agent_output)

        run_eval(hook_input, config_path)

        call_args = mock_call_judge.call_args
        assert "qubits" in str(call_args)
