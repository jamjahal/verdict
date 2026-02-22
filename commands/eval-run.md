---
description: Manually run the eval pipeline on the most recent agent output
allowed-tools: Read, Bash
argument-hint: [optional: output text to evaluate directly]
---

Manually trigger the eval-harness evaluation pipeline.

@${CLAUDE_PLUGIN_ROOT}/skills/eval-methodology/SKILL.md

If $ARGUMENTS is provided, evaluate that text directly as the agent output.
If $ARGUMENTS is empty, evaluate the most recent agent output from this session.

Steps:
1. Run Layer 1 heuristic guard: execute `python3 ${CLAUDE_PLUGIN_ROOT}/hooks/scripts/heuristic_guard_check.py` with the output as input.
2. If Layer 1 passes, launch the judge-agent to evaluate the output against the active rubric.
3. Display the full EVAL_RESULT block to the user: per-dimension scores, weighted score, verdict, and critique.
4. If FAIL and retries remain, ask the user whether to trigger a retry loop or surface the critique only.
5. Log the eval event to `${CLAUDE_PLUGIN_ROOT}/logs/eval_log.jsonl`.
