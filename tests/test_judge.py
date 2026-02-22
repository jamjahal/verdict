"""Tests for judge.py — LLM-as-Judge output parser.

TDD: These tests define the interface and expected behaviour.
All tests will FAIL until src/judge.py is implemented (Red phase).
"""

import pytest
from src.judge import JudgeOutputParser, JudgeVerdict, RubricScore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_PASS_OUTPUT = """
Some preamble text from the judge.

EVAL_RESULT
task_completion: 4
factual_groundedness: 4
coherence: 5
relevance: 4
safety: 5
weighted_score: 4.25
verdict: PASS
critique: |
  The response was clear and complete.
END_EVAL_RESULT
"""

VALID_FAIL_OUTPUT = """
EVAL_RESULT
task_completion: 2
factual_groundedness: 3
coherence: 3
relevance: 2
safety: 5
weighted_score: 2.6
verdict: FAIL
critique: |
  The response did not complete the core task. It addressed the format question
  but omitted the required explanation of the approach. On retry, ensure you
  directly answer why this approach was chosen.
END_EVAL_RESULT
"""

MALFORMED_NO_BLOCK = "This judge output has no EVAL_RESULT block at all."

MALFORMED_MISSING_FIELD = """
EVAL_RESULT
task_completion: 4
coherence: 5
relevance: 4
safety: 5
weighted_score: 4.0
verdict: PASS
END_EVAL_RESULT
"""


@pytest.fixture
def parser() -> JudgeOutputParser:
    return JudgeOutputParser()


# ---------------------------------------------------------------------------
# Parsing valid outputs
# ---------------------------------------------------------------------------

class TestParseValidOutputs:

    def test_parse_pass_verdict(self, parser: JudgeOutputParser) -> None:
        verdict = parser.parse(VALID_PASS_OUTPUT)
        assert isinstance(verdict, JudgeVerdict)
        assert verdict.passed is True

    def test_parse_fail_verdict(self, parser: JudgeOutputParser) -> None:
        verdict = parser.parse(VALID_FAIL_OUTPUT)
        assert verdict.passed is False

    def test_scores_populated_on_pass(self, parser: JudgeOutputParser) -> None:
        verdict = parser.parse(VALID_PASS_OUTPUT)
        assert isinstance(verdict.scores, RubricScore)
        assert verdict.scores.task_completion == 4
        assert verdict.scores.coherence == 5
        assert verdict.scores.safety == 5

    def test_scores_populated_on_fail(self, parser: JudgeOutputParser) -> None:
        verdict = parser.parse(VALID_FAIL_OUTPUT)
        assert verdict.scores.task_completion == 2
        assert verdict.scores.relevance == 2

    def test_weighted_score_parsed(self, parser: JudgeOutputParser) -> None:
        verdict = parser.parse(VALID_PASS_OUTPUT)
        assert verdict.scores.weighted_score == pytest.approx(4.25)

    def test_critique_none_on_pass(self, parser: JudgeOutputParser) -> None:
        verdict = parser.parse(VALID_PASS_OUTPUT)
        assert verdict.critique is None

    def test_critique_present_on_fail(self, parser: JudgeOutputParser) -> None:
        verdict = parser.parse(VALID_FAIL_OUTPUT)
        assert verdict.critique is not None
        assert len(verdict.critique) > 10

    def test_raw_output_preserved(self, parser: JudgeOutputParser) -> None:
        verdict = parser.parse(VALID_PASS_OUTPUT)
        assert verdict.raw_output == VALID_PASS_OUTPUT

    def test_preamble_text_ignored(self, parser: JudgeOutputParser) -> None:
        """Parser should ignore text before EVAL_RESULT block."""
        verdict = parser.parse(VALID_PASS_OUTPUT)
        assert verdict.passed is True  # preamble didn't corrupt parse


# ---------------------------------------------------------------------------
# Malformed inputs
# ---------------------------------------------------------------------------

class TestMalformedInputs:

    def test_raises_on_missing_block(self, parser: JudgeOutputParser) -> None:
        with pytest.raises(ValueError, match="EVAL_RESULT"):
            parser.parse(MALFORMED_NO_BLOCK)

    def test_raises_on_missing_required_field(self, parser: JudgeOutputParser) -> None:
        with pytest.raises(ValueError):
            parser.parse(MALFORMED_MISSING_FIELD)

    def test_raises_on_empty_string(self, parser: JudgeOutputParser) -> None:
        with pytest.raises(ValueError):
            parser.parse("")

    def test_raises_on_score_out_of_range(self, parser: JudgeOutputParser) -> None:
        bad = VALID_PASS_OUTPUT.replace("task_completion: 4", "task_completion: 6")
        with pytest.raises(ValueError, match="range"):
            parser.parse(bad)


# ---------------------------------------------------------------------------
# Custom rubric weights
# ---------------------------------------------------------------------------

class TestCustomWeights:

    def test_custom_weights_accepted(self) -> None:
        custom = {"task_completion": 0.5, "factual_groundedness": 0.5,
                  "coherence": 0.0, "relevance": 0.0, "safety": 0.0}
        parser = JudgeOutputParser(rubric_weights=custom)
        verdict = parser.parse(VALID_PASS_OUTPUT)
        assert verdict is not None

    def test_default_weights_sum_to_one(self) -> None:
        weights = JudgeOutputParser.DEFAULT_WEIGHTS
        assert sum(weights.values()) == pytest.approx(1.0)
