---
name: eval-methodology
description: Activate this skill when evaluating AI agent outputs, running quality checks on agent responses, applying the Self-Refine retry loop, interpreting judge-agent verdicts, or deciding whether an output passes quality thresholds. This skill provides the conceptual framework and workflow knowledge for the eval pipeline.
version: 0.1.0
---

# Eval Harness — Methodology

## What This Plugin Does

The eval-harness wraps any AI agent output with a two-layer quality pipeline before it reaches the user or downstream systems.

**Layer 1 — Heuristic Guard (deterministic, fast, free):**
Catches catastrophic failures before any LLM call is made. Checks for PII, exposed secrets/API keys, profanity, schema/format violations, and hard length constraints. If Layer 1 fails, the retry loop triggers immediately.

**Layer 2 — LLM Judge (semantic quality evaluation):**
The `judge-agent` evaluates the output against a G-Eval-inspired rubric (task completion, factual groundedness, coherence, relevance, safety). Returns a structured verdict with per-dimension scores and a natural-language critique.

---

## The Retry Loop (Self-Refine Pattern)

When an evaluation fails:

1. The judge's critique is injected directly into the primary agent's context alongside the original task prompt.
2. The primary agent retries — it is **eval-aware**, meaning it knows it failed and why.
3. This repeats up to `max_retries` times (default: 3, configurable in `config/rubric.yaml`).
4. After `max_retries` failures: the pass/fail hook fires, the output is blocked, and the event is logged for human review.

**Why 3 retries?** Research (Madaan et al. 2023 — Self-Refine; Shinn et al. 2023 — Reflexion) shows quality improvements plateau after 2–3 iterations. Beyond 3, reward hacking and judge-evaluatee sycophancy risks increase significantly.

---

## Rubric Configuration

The default rubric lives in `config/rubric.yaml`. To customise:

```yaml
dimensions:
  task_completion:
    weight: 0.30
    min_pass_score: 3
  factual_groundedness:
    weight: 0.25
    min_pass_score: 3
  # ... etc

max_retries: 3
overall_pass_threshold: 3.0
```

Use `/eval-harness:eval-configure` to adjust settings for the current session without editing the file.

---

## When to Invoke Evaluation

Evaluation runs automatically via the PostToolUse hook after agent outputs. Manual evaluation can be triggered with `/eval-harness:eval-run`.

Do **not** invoke the judge-agent for:
- Short conversational responses (greetings, clarifications)
- System messages or tool call results
- Outputs that have already passed evaluation in the current retry chain

---

## Interpreting Results

A `PASS` verdict means all rubric dimensions met their minimum score threshold and the weighted overall score is above `overall_pass_threshold`.

A `FAIL` verdict with a critique means the output should be retried with the critique injected. The critique is addressed directly to the primary agent — pass it verbatim.

A `FAIL` after `max_retries` exhausted means: block the output, log the full trace (all outputs, all critiques, all scores), and surface to the user for human review.

---

## Portability

The core eval logic (`src/heuristic_guard.py`, `src/judge.py`, `src/retry_loop.py`) is a standalone Python library with no Claude-specific imports. Any agent framework can `pip install` and invoke it directly. The plugin is the Claude Code integration layer.
