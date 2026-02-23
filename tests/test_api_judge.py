"""Tests for api_judge.py — LLM-as-Judge API caller.

TDD: These tests define the interface and expected behaviour.
All tests will FAIL until src/api_judge.py is implemented (Red phase).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.api_judge import build_judge_system_prompt, build_judge_user_message, call_judge
from src.config import EvalConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> EvalConfig:
    """Return a default EvalConfig."""
    return EvalConfig()


@pytest.fixture
def custom_config() -> EvalConfig:
    """Return a config with custom judge model and weights."""
    return EvalConfig(
        judge_model="claude-opus-4-5-20251101",
        weights={
            "task_completion": 0.40,
            "factual_groundedness": 0.25,
            "coherence": 0.15,
            "relevance": 0.10,
            "safety": 0.10,
        },
        min_pass_scores={
            "task_completion": 4,
            "factual_groundedness": 3,
            "coherence": 3,
            "relevance": 3,
            "safety": 4,
        },
        overall_pass_threshold=3.5,
    )


SAMPLE_TASK = "Write a Python function to sort a list using merge sort."
SAMPLE_OUTPUT = "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    ..."


# ---------------------------------------------------------------------------
# build_judge_system_prompt
# ---------------------------------------------------------------------------

class TestBuildJudgeSystemPrompt:

    def test_returns_string(self, default_config: EvalConfig) -> None:
        prompt = build_judge_system_prompt(default_config)
        assert isinstance(prompt, str)

    def test_contains_eval_result_format(self, default_config: EvalConfig) -> None:
        prompt = build_judge_system_prompt(default_config)
        assert "EVAL_RESULT" in prompt
        assert "END_EVAL_RESULT" in prompt

    def test_contains_all_dimensions(self, default_config: EvalConfig) -> None:
        prompt = build_judge_system_prompt(default_config)
        for dim in ["task_completion", "factual_groundedness", "coherence", "relevance", "safety"]:
            assert dim in prompt

    def test_contains_weights(self, default_config: EvalConfig) -> None:
        prompt = build_judge_system_prompt(default_config)
        assert "0.3" in prompt or "30%" in prompt

    def test_contains_min_pass_scores(self, default_config: EvalConfig) -> None:
        prompt = build_judge_system_prompt(default_config)
        assert "min" in prompt.lower()

    def test_contains_pass_threshold(self, default_config: EvalConfig) -> None:
        prompt = build_judge_system_prompt(default_config)
        assert "3.0" in prompt

    def test_custom_weights_reflected(self, custom_config: EvalConfig) -> None:
        prompt = build_judge_system_prompt(custom_config)
        assert "0.4" in prompt or "40%" in prompt

    def test_contains_sycophancy_guard(self, default_config: EvalConfig) -> None:
        """The judge prompt must warn against sycophancy (inflating scores)."""
        prompt = build_judge_system_prompt(default_config)
        assert "sycophancy" in prompt.lower() or "inflate" in prompt.lower()


# ---------------------------------------------------------------------------
# build_judge_user_message
# ---------------------------------------------------------------------------

class TestBuildJudgeUserMessage:

    def test_returns_string(self) -> None:
        msg = build_judge_user_message(SAMPLE_TASK, SAMPLE_OUTPUT)
        assert isinstance(msg, str)

    def test_contains_task_prompt(self) -> None:
        msg = build_judge_user_message(SAMPLE_TASK, SAMPLE_OUTPUT)
        assert SAMPLE_TASK in msg

    def test_contains_agent_output(self) -> None:
        msg = build_judge_user_message(SAMPLE_TASK, SAMPLE_OUTPUT)
        assert SAMPLE_OUTPUT in msg

    def test_clearly_separates_task_and_output(self) -> None:
        msg = build_judge_user_message(SAMPLE_TASK, SAMPLE_OUTPUT)
        # Should have some separator/label so the judge knows which is which
        assert "task" in msg.lower() or "prompt" in msg.lower()
        assert "output" in msg.lower() or "response" in msg.lower()


# ---------------------------------------------------------------------------
# call_judge (mocked API)
# ---------------------------------------------------------------------------

MOCK_JUDGE_RESPONSE = """Here is my evaluation:

EVAL_RESULT
task_completion: 4
factual_groundedness: 4
coherence: 5
relevance: 4
safety: 5
weighted_score: 4.25
verdict: PASS
critique: |
  The implementation is solid but missing edge case handling.
END_EVAL_RESULT"""


class TestCallJudge:

    @patch("src.api_judge.anthropic")
    def test_returns_string(self, mock_anthropic, default_config: EvalConfig) -> None:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=MOCK_JUDGE_RESPONSE)]
        mock_client.messages.create.return_value = mock_response

        result = call_judge(default_config, SAMPLE_TASK, SAMPLE_OUTPUT)
        assert isinstance(result, str)

    @patch("src.api_judge.anthropic")
    def test_returns_eval_result_block(self, mock_anthropic, default_config: EvalConfig) -> None:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=MOCK_JUDGE_RESPONSE)]
        mock_client.messages.create.return_value = mock_response

        result = call_judge(default_config, SAMPLE_TASK, SAMPLE_OUTPUT)
        assert "EVAL_RESULT" in result

    @patch("src.api_judge.anthropic")
    def test_uses_config_model(self, mock_anthropic, custom_config: EvalConfig) -> None:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=MOCK_JUDGE_RESPONSE)]
        mock_client.messages.create.return_value = mock_response

        call_judge(custom_config, SAMPLE_TASK, SAMPLE_OUTPUT)

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["model"] == "claude-opus-4-5-20251101"

    @patch("src.api_judge.anthropic")
    def test_passes_system_and_user_messages(self, mock_anthropic, default_config: EvalConfig) -> None:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=MOCK_JUDGE_RESPONSE)]
        mock_client.messages.create.return_value = mock_response

        call_judge(default_config, SAMPLE_TASK, SAMPLE_OUTPUT)

        call_args = mock_client.messages.create.call_args
        assert "system" in call_args.kwargs
        assert "messages" in call_args.kwargs
        assert len(call_args.kwargs["messages"]) == 1
        assert call_args.kwargs["messages"][0]["role"] == "user"

    @patch("src.api_judge.anthropic")
    def test_sets_max_tokens(self, mock_anthropic, default_config: EvalConfig) -> None:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=MOCK_JUDGE_RESPONSE)]
        mock_client.messages.create.return_value = mock_response

        call_judge(default_config, SAMPLE_TASK, SAMPLE_OUTPUT)

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["max_tokens"] >= 1024

    @patch("src.api_judge.anthropic")
    def test_api_error_propagates(self, mock_anthropic, default_config: EvalConfig) -> None:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API rate limit")

        with pytest.raises(Exception, match="API rate limit"):
            call_judge(default_config, SAMPLE_TASK, SAMPLE_OUTPUT)


# ---------------------------------------------------------------------------
# Integration: call_judge output is parseable by JudgeOutputParser
# ---------------------------------------------------------------------------

class TestCallJudgeIntegration:

    @patch("src.api_judge.anthropic")
    def test_parseable_by_judge_output_parser(self, mock_anthropic, default_config: EvalConfig) -> None:
        from src.judge import JudgeOutputParser

        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=MOCK_JUDGE_RESPONSE)]
        mock_client.messages.create.return_value = mock_response

        raw = call_judge(default_config, SAMPLE_TASK, SAMPLE_OUTPUT)
        parser = default_config.build_parser()
        verdict = parser.parse(raw)

        assert verdict.passed is True
        assert verdict.scores.task_completion == 4
        assert verdict.scores.weighted_score == 4.25
