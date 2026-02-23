---
description: Show eval log summary — pass rates, retry distributions, top failure dimensions
allowed-tools: Read, Bash
argument-hint: [optional: last N | today | all]
---

Generate a summary report from the eval log.

Load the eval log: !`test -f ${CLAUDE_PLUGIN_ROOT}/logs/eval_log.jsonl && wc -l ${CLAUDE_PLUGIN_ROOT}/logs/eval_log.jsonl || echo "NO_LOG"`

If no log exists, explain that no eval events have been recorded yet and suggest running `/eval-harness:eval-run` or enabling the PostToolUse hook.

Otherwise, read `${CLAUDE_PLUGIN_ROOT}/logs/eval_log.jsonl` and compute:

1. **Overall pass rate** — percentage of eval events that passed on first attempt
2. **Recovery rate** — percentage of initial failures that eventually passed after retry
3. **Retry distribution** — how often 1, 2, or 3 retries were needed
4. **Top failure dimensions** — which rubric dimensions fail most frequently
5. **Recent failures** — last 5 FAIL events with their critiques and final verdicts

Filter scope based on $ARGUMENTS:
- `last N` — most recent N events
- `today` — events from today only
- `all` (default) — full log

Present results in a clean, readable summary. Note any patterns worth attention (e.g. "Factual Groundedness is failing 60% of the time — consider improving context injection in your primary agent prompt.").
