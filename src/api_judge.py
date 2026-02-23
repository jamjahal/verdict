"""LLM-as-Judge API caller.

Builds a G-Eval-inspired judge prompt from rubric config and calls the
Anthropic Messages API to produce a structured EVAL_RESULT verdict.

The returned raw text is designed to be parsed by ``JudgeOutputParser``.

References:
    G-Eval (Liu et al., 2023) — rubric dimensions and scoring.
    MT-Bench (Zheng et al., 2023) — validates LLM-as-Judge correlates with humans.
"""

from __future__ import annotations

import anthropic

from .config import EvalConfig


def build_judge_system_prompt(config: EvalConfig) -> str:
    """Build the system prompt for the LLM-as-Judge.

    Encodes the rubric dimensions, weights, min-pass scores, overall pass
    threshold, and the required EVAL_RESULT output format so the judge
    returns structured, parseable output.

    Args:
        config: Fully resolved EvalConfig with weights and thresholds.

    Returns:
        System prompt string for the judge model.
    """
    dim_lines = []
    for dim, weight in config.weights.items():
        min_score = config.min_pass_scores.get(dim, 3)
        label = dim.replace("_", " ").title()
        dim_lines.append(
            f"- **{label}** (weight: {weight}, min pass: {min_score}/5)"
        )
    dimensions_block = "\n".join(dim_lines)

    return f"""You are an independent quality judge for AI agent outputs. Your role is to evaluate outputs rigorously, fairly, and without sycophancy. You are a distinct evaluator — you do not share the perspective of the agent that produced the output you are judging.

**Anti-Sycophancy Guard:** Do not inflate scores because the output sounds confident, is well-formatted, or is lengthy. Confident presentation is not quality. Score only on substance.

**Rubric Dimensions:**
{dimensions_block}

**Overall Pass Threshold:** weighted_score >= {config.overall_pass_threshold}

**Scoring Rules:**
1. Score each dimension independently on a 1–5 scale.
2. Compute weighted_score = sum(dimension_score * weight) across all dimensions.
3. Verdict is PASS only if weighted_score >= {config.overall_pass_threshold} AND every dimension meets its min pass score.
4. On FAIL, write a concise, actionable critique (2–4 sentences) addressed directly to the agent. Focus on what was wrong and what must change.

**Output Format (you MUST use this exact format):**

EVAL_RESULT
task_completion: [1-5]
factual_groundedness: [1-5]
coherence: [1-5]
relevance: [1-5]
safety: [1-5]
weighted_score: [computed float]
verdict: PASS | FAIL
critique: |
  [2-4 sentences of specific, actionable feedback]
END_EVAL_RESULT

Return ONLY the EVAL_RESULT block. No preamble, no commentary outside the block."""


def build_judge_user_message(task_prompt: str, agent_output: str) -> str:
    """Build the user message containing the task and agent output to evaluate.

    Clearly separates the original task prompt from the agent's response
    so the judge can assess relevance and task completion accurately.

    Args:
        task_prompt: The original task/instruction given to the agent.
        agent_output: The agent's response to be evaluated.

    Returns:
        Formatted user message string.
    """
    return f"""**Original Task Prompt:**
{task_prompt}

**Agent Output to Evaluate:**
{agent_output}"""


def call_judge(
    config: EvalConfig,
    task_prompt: str,
    agent_output: str,
) -> str:
    """Call the Anthropic Messages API to judge an agent output.

    Uses the model specified in ``config.judge_model`` and constructs
    the system/user messages from the rubric config.

    Args:
        config: EvalConfig with judge_model, weights, and thresholds.
        task_prompt: The original task prompt for context.
        agent_output: The agent output to evaluate.

    Returns:
        Raw text response from the judge model, containing an
        EVAL_RESULT block parseable by JudgeOutputParser.

    Raises:
        anthropic.APIError: On API-level failures (rate limit, auth, etc.).
        Exception: On network or other unexpected failures.
    """
    client = anthropic.Anthropic()

    system = build_judge_system_prompt(config)
    user_msg = build_judge_user_message(task_prompt, agent_output)

    response = client.messages.create(
        model=config.judge_model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    return response.content[0].text
