"""Self-Refine retry loop orchestrator.

Implements the Self-Refine pattern (Madaan et al. 2023) and Reflexion verbal
reinforcement learning (Shinn et al. 2023). On eval failure, injects the
judge's critique back into the primary agent's context explicitly, allowing
the agent to learn within the session from its mistakes.

The primary agent is eval-aware: it receives the original task prompt plus
the critique and knows it is retrying.

Retry budget is configurable (default: max_retries=3). Beyond 3 iterations,
quality degradation from reward hacking and sycophancy outweigh gains.

References:
    Self-Refine (Madaan et al., 2023)
    Reflexion (Shinn et al., 2023)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import logging

from .heuristic_guard import GuardResult, HeuristicGuard
from .judge import JudgeVerdict, JudgeOutputParser


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Callable type for the primary agent: takes (prompt, critique) -> output str.
# critique is None on the first attempt.
AgentFn = Callable[[str, Optional[str]], str]

# Callable type for the judge: takes (task_prompt, output) -> raw judge text.
JudgeFn = Callable[[str, str], str]


@dataclass
class LoopResult:
    """Result of a full retry loop execution.

    Attributes:
        final_output: The last output produced (whether passed or not).
        passed: True if the loop ended with a PASS verdict.
        attempts: Total number of agent attempts (1 = no retries needed).
        verdicts: List of JudgeVerdict objects, one per attempt.
        blocked: True if max_retries was exhausted without a PASS.
        failure_reason: Human-readable summary if blocked=True.
    """
    final_output: str
    passed: bool
    attempts: int
    verdicts: list[JudgeVerdict]
    blocked: bool = False
    failure_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class RetryLoop:
    """Orchestrates the heuristic guard → LLM judge → retry loop pipeline.

    Args:
        agent_fn: Callable that runs the primary agent. Receives the task
                  prompt and optional critique string; returns output text.
        judge_fn: Callable that runs the LLM judge. Receives task prompt and
                  agent output; returns raw judge text (EVAL_RESULT block).
        guard: HeuristicGuard instance. Uses default config if None.
        parser: JudgeOutputParser instance. Uses default config if None.
        max_retries: Maximum retry iterations. Defaults to 3.
    """

    def __init__(
        self,
        agent_fn: AgentFn,
        judge_fn: JudgeFn,
        guard: Optional[HeuristicGuard] = None,
        parser: Optional[JudgeOutputParser] = None,
        max_retries: int = 3,
    ) -> None:
        self.agent_fn = agent_fn
        self.judge_fn = judge_fn
        self.guard: HeuristicGuard = guard or HeuristicGuard()
        self.parser: JudgeOutputParser = parser or JudgeOutputParser()
        self.max_retries = max_retries

    def run(self, task_prompt: str) -> LoopResult:
        """Execute the full eval pipeline for a given task prompt.

        Runs the primary agent, evaluates with the heuristic guard and judge,
        and retries with critique injection up to max_retries times.

        Args:
            task_prompt: The original task prompt for the primary agent.

        Returns:
            LoopResult summarising the full run, including all verdicts and
            whether the loop ultimately passed or was blocked.
        """
        verdicts: list[JudgeVerdict] = []
        critique: Optional[str] = None
        final_output: str = ""

        for attempt in range(1, self.max_retries + 1):
            # Build prompt — first attempt uses raw task; retries inject critique.
            if attempt == 1:
                prompt = task_prompt
            else:
                prompt = self._build_retry_prompt(task_prompt, critique, attempt)  # type: ignore[arg-type]

            output = self.agent_fn(prompt, critique)
            final_output = output

            # Layer 1: fast deterministic guard — skip judge on failure.
            guard_result = self.guard.evaluate(output)
            if not guard_result.passed:
                critique = (
                    f"Output failed safety checks: {guard_result.details}. "
                    "Please revise to remove any PII, secrets, or disallowed content."
                )
                logger.debug("Attempt %d blocked by heuristic guard: %s", attempt, guard_result.details)
                if attempt == self.max_retries:
                    return self._handle_blocked(verdicts, final_output)
                continue

            # Layer 2: LLM judge — semantic quality evaluation.
            raw_judge_output = self.judge_fn(task_prompt, output)
            verdict = self.parser.parse(raw_judge_output)
            verdicts.append(verdict)

            if verdict.passed:
                logger.debug("Attempt %d passed judge evaluation.", attempt)
                return LoopResult(
                    final_output=final_output,
                    passed=True,
                    attempts=attempt,
                    verdicts=verdicts,
                    blocked=False,
                )

            critique = verdict.critique
            logger.debug("Attempt %d failed judge. Critique: %s", attempt, critique)

            if attempt == self.max_retries:
                return self._handle_blocked(verdicts, final_output)

        # Should never reach here, but satisfy type checker.
        return self._handle_blocked(verdicts, final_output)  # pragma: no cover

    def _build_retry_prompt(self, task_prompt: str, critique: str, attempt: int) -> str:
        """Construct the retry prompt with injected critique.

        Follows the Self-Refine pattern: the original task prompt is preserved
        and the critique is appended as explicit context. The agent knows it
        is on a retry attempt.

        Args:
            task_prompt: The original task prompt.
            critique: The judge's critique from the previous attempt.
            attempt: Current attempt number (1-indexed).

        Returns:
            Full prompt string for the next agent call.
        """
        return (
            f"{task_prompt}\n\n"
            f"--- Evaluator feedback (attempt {attempt - 1}) ---\n"
            f"{critique}\n"
            f"---\n"
            f"Please address the feedback above and try again."
        )

    def _handle_blocked(self, verdicts: list[JudgeVerdict], final_output: str) -> LoopResult:
        """Handle the case where max_retries is exhausted without a PASS.

        Logs the full trace and returns a blocked LoopResult for the
        pass/fail hook to act on.

        Args:
            verdicts: All JudgeVerdict objects from each attempt.
            final_output: The last agent output.

        Returns:
            LoopResult with blocked=True and a failure summary.
        """
        if verdicts and verdicts[-1].critique:
            last_critique = verdicts[-1].critique
        else:
            last_critique = "Safety checks failed on all attempts."

        logger.warning(
            "RetryLoop blocked after %d attempts. Last critique: %s",
            self.max_retries,
            last_critique,
        )

        return LoopResult(
            final_output=final_output,
            passed=False,
            attempts=self.max_retries,
            verdicts=verdicts,
            blocked=True,
            failure_reason=(
                f"Max retries ({self.max_retries}) exhausted. "
                f"Last critique: {last_critique}"
            ),
        )
