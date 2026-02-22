---
description: Configure eval rubric, retry limit, and heuristic rules for this session
allowed-tools: Read, Write
argument-hint: [setting] [value] — e.g. max_retries 5 | dimension task_completion weight 0.4
---

Configure the eval-harness settings for this session.

@${CLAUDE_PLUGIN_ROOT}/config/rubric.yaml

Valid settings:

- `max_retries [n]` — Set the maximum number of retry iterations (default: 3)
- `threshold [score]` — Set the overall pass threshold (default: 3.0)
- `dimension [name] weight [value]` — Adjust a rubric dimension weight (must sum to 1.0 across all dimensions)
- `dimension [name] min [score]` — Adjust a dimension's minimum pass score
- `heuristic [rule] [on|off]` — Enable or disable a specific heuristic check (pii, secrets, profanity, format)
- `show` — Display current active configuration

If $ARGUMENTS is `show` or empty, display the current rubric.yaml contents in a readable format.

Otherwise, parse $ARGUMENTS, validate the setting, update the in-session config, and confirm the change. Do not write to rubric.yaml unless the user explicitly says "save" — session changes are ephemeral by default.

After any change, display the full updated configuration.
