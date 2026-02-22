"""Layer 2 LLM-as-Judge semantic evaluator.

Invokes the judge-agent subagent and parses its structured EVAL_RESULT block
into a typed Python object. Designed to be called by the RetryLoop orchestrator.

The judge uses a G-Eval-inspired rubric (Liu et al. 2023) with configurable
dimension weights and pass thresholds loaded from rubric.yaml.

References:
    G-Eval (Liu et al., 2023) — rubric dimensions and scoring.
    MT-Bench (Zheng et al., 2023) — validates LLM-as-Judge correlation with humans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import re


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RubricScore:
    """Per-dimension score from the judge.

    Attributes:
        task_completion: Score 1-5.
        factual_groundedness: Score 1-5.
        coherence: Score 1-5.
        relevance: Score 1-5.
        safety: Score 1-5.
        weighted_score: Computed weighted overall score (0.0-5.0).
    """
    task_completion: int
    factual_groundedness: int
    coherence: int
    relevance: int
    safety: int
    weighted_score: float


@dataclass
class JudgeVerdict:
    """Full structured output from a judge evaluation.

    Attributes:
        scores: Per-dimension RubricScore.
        passed: True if all dimensions met minimum thresholds and weighted score
                exceeds overall_pass_threshold.
        critique: Natural-language critique addressed to the primary agent.
                  Injected as context on retry. None if verdict is PASS.
        raw_output: The unparsed judge output for debugging/logging.
    """
    scores: RubricScore
    passed: bool
    critique: Optional[str]
    raw_output: str


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class JudgeOutputParser:
    """Parses the structured EVAL_RESULT block from judge-agent output.

    Expected format::

        EVAL_RESULT
        task_completion: 4
        factual_groundedness: 3
        coherence: 5
        relevance: 4
        safety: 5
        weighted_score: 4.05
        verdict: PASS
        critique: |
          The response was clear but omitted X.
        END_EVAL_RESULT

    Args:
        rubric_weights: Dict mapping dimension name to float weight.
                        If None, uses defaults from G-Eval paper.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "task_completion": 0.30,
        "factual_groundedness": 0.25,
        "coherence": 0.20,
        "relevance": 0.15,
        "safety": 0.10,
    }

    def __init__(self, rubric_weights: Optional[dict[str, float]] = None) -> None:
        self.weights = rubric_weights or self.DEFAULT_WEIGHTS

    def parse(self, raw_output: str) -> JudgeVerdict:
        """Parse a judge-agent raw output string into a JudgeVerdict.

        Args:
            raw_output: The full text output from the judge-agent, which must
                        contain an EVAL_RESULT...END_EVAL_RESULT block.

        Returns:
            Parsed JudgeVerdict with scores, verdict, and critique.

        Raises:
            ValueError: If the EVAL_RESULT block is missing or malformed.
        """
        if not raw_output:
            raise ValueError("EVAL_RESULT block not found: empty input")

        block = self._extract_block(raw_output)
        scores = self._parse_scores(block)

        verdict_match = re.search(r"^verdict:\s*(PASS|FAIL)", block, re.MULTILINE)
        if not verdict_match:
            raise ValueError("Missing required field: verdict")
        passed = verdict_match.group(1) == "PASS"

        # Critique is only surfaced on FAIL — suppress it on PASS to keep
        # the contract clean (critique is the retry signal, not a footnote).
        critique = None if passed else self._parse_critique(block)

        return JudgeVerdict(
            scores=scores,
            passed=passed,
            critique=critique,
            raw_output=raw_output,
        )

    def _extract_block(self, raw_output: str) -> str:
        """Extract the EVAL_RESULT block from raw judge output.

        Args:
            raw_output: Full judge text output.

        Returns:
            Content between EVAL_RESULT and END_EVAL_RESULT markers.

        Raises:
            ValueError: If markers are not found.
        """
        match = re.search(
            r"EVAL_RESULT\n(.*?)\nEND_EVAL_RESULT",
            raw_output,
            re.DOTALL,
        )
        if not match:
            raise ValueError("EVAL_RESULT block not found in judge output")
        return match.group(1)

    def _parse_scores(self, block: str) -> RubricScore:
        """Parse dimension scores and weighted score from the EVAL_RESULT block.

        Args:
            block: The extracted EVAL_RESULT content string.

        Returns:
            RubricScore with all dimension values populated.

        Raises:
            ValueError: If any required score field is missing or out of range.
        """
        required_dims = [
            "task_completion",
            "factual_groundedness",
            "coherence",
            "relevance",
            "safety",
        ]
        scores: dict[str, int] = {}
        for dim in required_dims:
            match = re.search(rf"^{dim}:\s*(\d+)", block, re.MULTILINE)
            if not match:
                raise ValueError(f"Missing required field: {dim}")
            val = int(match.group(1))
            if not 1 <= val <= 5:
                raise ValueError(
                    f"Score {dim}={val} out of range (must be 1–5)"
                )
            scores[dim] = val

        ws_match = re.search(r"^weighted_score:\s*([\d.]+)", block, re.MULTILINE)
        if not ws_match:
            raise ValueError("Missing required field: weighted_score")

        return RubricScore(
            task_completion=scores["task_completion"],
            factual_groundedness=scores["factual_groundedness"],
            coherence=scores["coherence"],
            relevance=scores["relevance"],
            safety=scores["safety"],
            weighted_score=float(ws_match.group(1)),
        )

    def _parse_critique(self, block: str) -> Optional[str]:
        """Extract the critique text from the EVAL_RESULT block.

        Args:
            block: The extracted EVAL_RESULT content string.

        Returns:
            Critique string if present and non-empty, None otherwise.
        """
        match = re.search(
            r"^critique:\s*\|\n(.*?)(?=\n\S|\Z)",
            block,
            re.MULTILINE | re.DOTALL,
        )
        if not match:
            return None
        lines = match.group(1).splitlines()
        critique = "\n".join(line.strip() for line in lines).strip()
        return critique or None
