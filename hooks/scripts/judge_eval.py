#!/usr/bin/env python3
"""PostToolUse hook: autonomous two-layer quality pipeline for Task tool outputs.

Fired automatically after every Task tool completion. Runs the full
Verdict pipeline:

    Layer 1 - HeuristicGuard (deterministic): PII, secrets, profanity
    Layer 2 - LLM-as-Judge (Anthropic API): G-Eval rubric scoring

Outputs a JSON object with a ``systemMessage`` field that Claude sees
in context. On FAIL, the message includes the critique and an instruction
to retry - Claude interprets this and re-runs the task autonomously.

Hook input (stdin): JSON with tool_name, tool_input, tool_result.
Hook output (stdout): JSON with systemMessage.
Exit code: always 0 (non-blocking; guidance via systemMessage).
"""

from __future__ import annotations

import json
import os
import sys

# Ensure the plugin src/ directory is importable.
PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PLUGIN_ROOT)

from src.config import load_config
from src.api_judge import call_judge


# ---------------------------------------------------------------------------
# Minimum output length to trigger assessment (skip trivial results).
# ---------------------------------------------------------------------------
_MIN_OUTPUT_LENGTH = 50


def run_eval(hook_input: dict, config_path: str) -> dict | None:
    """Run the two-layer quality pipeline on a PostToolUse hook input.

    Args:
        hook_input: Parsed JSON from stdin with tool_name, tool_input,
                    and tool_result fields.
        config_path: Path to the rubric.yaml config file.

    Returns:
        Dict with ``systemMessage`` key for Claude, or None to skip
        silently (trivial outputs).
    """
    # --- Extract fields ---
    tool_input = hook_input.get("tool_input", {})
    tool_result = hook_input.get("tool_result", "")

    agent_output = str(tool_result).strip()

    # --- Skip trivial outputs ---
    if len(agent_output) < _MIN_OUTPUT_LENGTH:
        return None

    task_prompt = ""
    if isinstance(tool_input, dict):
        task_prompt = tool_input.get("prompt", "")

    # --- Load config ---
    config = load_config(config_path)

    # --- Layer 1: Heuristic Guard (deterministic) ---
    guard = config.build_guard()
    guard_result = guard.evaluate(agent_output)

    if not guard_result.passed:
        return {
            "systemMessage": (
                f"VERDICT FAILED - Layer 1 (Heuristic Guard): {guard_result.details}. "
                "The agent output contains safety violations. "
                "Please revise the output to remove all flagged content and try again."
            )
        }

    # --- Layer 2: LLM-as-Judge via Anthropic API ---
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "systemMessage": (
                "VERDICT SKIPPED - ANTHROPIC_API_KEY not set. "
                "Layer 2 (LLM Judge) requires an API key to run. "
                "Layer 1 (Heuristic Guard) passed."
            )
        }

    try:
        raw_judge_output = call_judge(config, task_prompt, agent_output)
        parser = config.build_parser()
        verdict = parser.parse(raw_judge_output)
    except Exception as exc:
        return {
            "systemMessage": (
                f"VERDICT ERROR - Layer 2 (LLM Judge) failed: {exc}. "
                "Layer 1 (Heuristic Guard) passed."
            )
        }

    if verdict.passed:
        return {
            "systemMessage": (
                f"VERDICT PASSED - Weighted score: {verdict.scores.weighted_score}/5.00. "
                "Output meets all quality thresholds."
            )
        }

    # FAIL - include critique and retry instruction
    return {
        "systemMessage": (
            f"VERDICT FAILED - Weighted score: {verdict.scores.weighted_score}/5.00. "
            f"Critique: {verdict.critique} "
            "Please revise the output addressing the feedback above and try again."
        )
    }


def main() -> None:
    """Entry point: read hook input from stdin, run assessment, print result."""
    try:
        raw_input = sys.stdin.read()
        hook_input = json.loads(raw_input) if raw_input.strip() else {}
    except json.JSONDecodeError:
        # Malformed input - skip silently.
        sys.exit(0)

    config_path = os.path.join(PLUGIN_ROOT, "config", "rubric.yaml")
    result = run_eval(hook_input, config_path)

    if result:
        print(json.dumps(result))

    sys.exit(0)


if __name__ == "__main__":
    main()
