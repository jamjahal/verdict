# eval-harness

> A portable two-layer LLM evaluation pipeline for AI agent outputs — deployable as a Claude Code plugin, a Vercel serverless API, or a standalone Python library.

## Overview

`eval-harness` wraps any AI agent with automatic quality evaluation before outputs reach users or downstream systems. It implements two evaluation layers:

**Layer 1 — Heuristic Guard** (deterministic, model-free): Catches PII, exposed secrets, profanity, and format violations instantly, without burning tokens.

**Layer 2 — LLM-as-Judge**: A separate judge agent scores outputs on a [G-Eval](https://arxiv.org/abs/2303.16634)-inspired rubric (task completion, factual groundedness, coherence, relevance, safety). On failure, structured critique is injected back into the primary agent following the [Self-Refine](https://arxiv.org/abs/2303.17651) pattern. Up to `max_retries` (default: 3) iterations before a pass/fail hook fires.

The core evaluation engine lives in `src/` with zero platform-specific imports. Both the Claude Code plugin and the Vercel API are thin wrappers over the same library — no logic duplication.

## Architecture

```
src/                  ← portable Python library (zero platform imports)
├── heuristic_guard   ← Layer 1: deterministic PII/secrets/profanity checks
├── api_judge         ← Layer 2: Anthropic LLM-as-Judge API caller
├── judge             ← EVAL_RESULT parser and verdict types
├── retry_loop        ← Self-Refine orchestrator
├── config            ← Rubric YAML loader
└── logger            ← JSONL eval event logger

web/                  ← FastAPI wrapper (Vercel / uvicorn)
├── app.py            ← Routes: /api/eval, /api/eval/heuristic, /api/eval/judge
├── models.py         ← Pydantic request/response schemas
└── auth.py           ← X-API-Key authentication

api/index.py          ← Vercel serverless entry point (imports web.app)
vercel.json           ← Vercel routing, runtime, and secret config

hooks/                ← Claude Code plugin integration (PreToolUse / PostToolUse)
commands/             ← Claude Code slash commands
agents/               ← Claude Code judge subagent spec
skills/               ← Claude Code eval-methodology skill
config/rubric.yaml    ← Shared rubric configuration (used by all deployment targets)
```

Both deployment targets call the same `src/` functions. The rubric config in `config/rubric.yaml` is shared — change it once, both targets pick it up.

---

## Installation

### Option 1: Claude Code Plugin

```bash
/plugin install eval-harness
```

The plugin hooks run automatically: `PreToolUse` enforces the conda env guard, `PostToolUse` runs the full eval pipeline on Task outputs.

### Option 2: Vercel Serverless API

**Prerequisites:** [Vercel CLI](https://vercel.com/docs/cli), an [Anthropic API key](https://console.anthropic.com/), and a Python 3.11+ project.

```bash
# 1. Clone the repo
git clone https://github.com/allanhall/eval-harness-plugin.git
cd eval-harness-plugin

# 2. Set secrets (Vercel encrypts these at rest)
vercel secrets add eval-harness-api-key "your-api-key-here"
vercel secrets add anthropic-api-key "sk-ant-..."

# 3. Deploy
vercel --prod
```

After deploy, your API is live at `https://your-app.vercel.app/api/`.

### Option 3: Standalone Python Library

```bash
pip install -r requirements.txt
```

---

## Web API Reference

### Authentication

All endpoints except `/api/health` require an `X-API-Key` header. Set the expected key via the `EVAL_HARNESS_API_KEY` environment variable (configured as a Vercel secret for deployments, or exported locally).

```bash
# Every authenticated request includes:
-H "X-API-Key: your-api-key-here"
```

### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/api/health` | No | Liveness check — returns version |
| `POST` | `/api/eval` | Yes | Full two-layer pipeline (heuristic + judge) |
| `POST` | `/api/eval/heuristic` | Yes | Layer 1 heuristic guard only |
| `POST` | `/api/eval/judge` | Yes | Layer 2 LLM-as-Judge only |
| `GET` | `/api/config` | Yes | Current rubric configuration |

---

### `POST /api/eval` — Full Pipeline

Runs both layers. If the heuristic guard fails, the LLM judge is skipped (saves tokens and latency).

**Request:**
```json
{
  "task_prompt": "Summarise Q3 results.",
  "agent_output": "Q3 revenue grew 12% YoY driven by new product launches."
}
```

**Response (pass):**
```json
{
  "passed": true,
  "heuristic": {
    "passed": true,
    "failed_checks": [],
    "details": ""
  },
  "judge": {
    "passed": true,
    "scores": {
      "task_completion": 4,
      "factual_groundedness": 4,
      "coherence": 5,
      "relevance": 5,
      "safety": 5,
      "weighted_score": 4.35
    },
    "critique": null
  }
}
```

**Response (heuristic blocked — judge skipped):**
```json
{
  "passed": false,
  "heuristic": {
    "passed": false,
    "failed_checks": ["pii"],
    "details": "PII detected: email"
  },
  "judge": null
}
```

**curl:**
```bash
curl -X POST https://your-app.vercel.app/api/eval \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "task_prompt": "Summarise Q3 results.",
    "agent_output": "Q3 revenue grew 12% YoY driven by new product launches."
  }'
```

---

### `POST /api/eval/heuristic` — Layer 1 Only

Fast, deterministic, no LLM call. Use this for real-time guardrails where latency matters.

**Request:**
```json
{
  "output": "Contact user@example.com for the full report."
}
```

**Response:**
```json
{
  "passed": false,
  "failed_checks": ["pii"],
  "details": "PII detected: email"
}
```

---

### `POST /api/eval/judge` — Layer 2 Only

Calls the Anthropic API. Use this when you've already run heuristic checks yourself and want just the semantic evaluation.

**Request:**
```json
{
  "task_prompt": "Write a summary of the Q3 earnings call.",
  "agent_output": "Revenue increased 12% year-over-year..."
}
```

**Response:**
```json
{
  "passed": true,
  "scores": {
    "task_completion": 4,
    "factual_groundedness": 4,
    "coherence": 5,
    "relevance": 5,
    "safety": 5,
    "weighted_score": 4.35
  },
  "critique": null
}
```

On failure, `critique` contains actionable feedback suitable for Self-Refine injection:
```json
{
  "passed": false,
  "scores": { "...": "..." },
  "critique": "The response missed the core question and contained unsupported claims."
}
```

---

### `GET /api/config`

Returns the active rubric configuration (loaded from `config/rubric.yaml`).

**Response:**
```json
{
  "max_retries": 3,
  "overall_pass_threshold": 3.0,
  "weights": {
    "task_completion": 0.30,
    "factual_groundedness": 0.25,
    "coherence": 0.20,
    "relevance": 0.15,
    "safety": 0.10
  },
  "min_pass_scores": {
    "task_completion": 3,
    "factual_groundedness": 3,
    "coherence": 3,
    "relevance": 3,
    "safety": 4
  },
  "judge_model": "claude-sonnet-4-5-20250929",
  "check_pii": true,
  "check_secrets": true,
  "check_profanity": true
}
```

---

## Vercel Integration Guide

### How It Works

Vercel's [Python runtime](https://vercel.com/docs/functions/runtimes/python) serves the FastAPI app as a serverless function. The wiring:

1. `vercel.json` rewrites all `/api/*` requests to `api/index.py`
2. `api/index.py` imports the FastAPI `app` from `web/app.py`
3. `web/app.py` imports evaluation logic from `src/`
4. `config/rubric.yaml` is bundled with the deployment (read-only)

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `EVAL_HARNESS_API_KEY` | Yes | API key for authenticating requests |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for the LLM-as-Judge |

Set via the Vercel dashboard (Settings > Environment Variables) or CLI:
```bash
vercel secrets add eval-harness-api-key "your-key"
vercel secrets add anthropic-api-key "sk-ant-..."
```

### Vercel-Specific Considerations

| Constraint | Impact | Mitigation |
|------------|--------|------------|
| **Function timeout** | Default 10s (Hobby), up to 300s (Pro). The LLM judge call can take 5–15s. | `vercel.json` sets `maxDuration: 120`. Use Pro plan for production. Heuristic-only endpoint (`/api/eval/heuristic`) responds in <100ms. |
| **Cold starts** | First request after idle may take 1–3s for Python runtime init. | Acceptable for eval workloads. Use `/api/health` as a warm-up ping if needed. |
| **Read-only filesystem** | `config/rubric.yaml` is bundled and read-only after deploy. Logs cannot be written to disk. | Config changes require redeployment. For persistent logging, connect an external store (Postgres, Redis, or a logging service) — the `EvalLogger` class is designed to be swappable. |
| **No persistent state** | Each function invocation is stateless. | Eval results are returned directly in the response. For audit trails, pipe results to your own data store from the calling service. |

### Integrating Into Your App

**From a Next.js/Vercel frontend:**
```typescript
const response = await fetch("/api/eval", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": process.env.EVAL_HARNESS_API_KEY!,
  },
  body: JSON.stringify({
    task_prompt: userQuery,
    agent_output: agentResponse,
  }),
});
const result = await response.json();

if (!result.passed) {
  // Retry with critique injection, show warning, or block output
  console.log("Eval failed:", result.judge?.critique ?? result.heuristic.details);
}
```

**From any backend (Python):**
```python
import httpx

resp = httpx.post(
    "https://your-app.vercel.app/api/eval",
    json={"task_prompt": prompt, "agent_output": output},
    headers={"X-API-Key": "your-key"},
)
result = resp.json()
```

**As middleware in an agent pipeline:**
```python
# Evaluate before returning to user
guard_resp = httpx.post(
    "https://your-app.vercel.app/api/eval/heuristic",
    json={"output": agent_output},
    headers={"X-API-Key": "your-key"},
)
if not guard_resp.json()["passed"]:
    # Block or retry before the LLM judge even runs
    raise SafetyError(guard_resp.json()["details"])
```

### Local Development

```bash
export EVAL_HARNESS_API_KEY="dev-key"
export ANTHROPIC_API_KEY="sk-ant-..."
uvicorn web.app:app --reload --port 8000
```

The API is then available at `http://localhost:8000/api/`. The same endpoints, same auth, same behavior as the Vercel deployment.

---

## Plugin Commands (Claude Code)

```
/eval-harness:eval-run              — Evaluate the most recent agent output
/eval-harness:eval-configure        — Adjust rubric, max_retries, heuristic rules
/eval-harness:eval-report           — View eval log summary and failure trends
```

### Python Library (Direct Import)

```python
from eval_harness import HeuristicGuard, RetryLoop

guard = HeuristicGuard()
loop = RetryLoop(agent_fn=my_agent, judge_fn=my_judge, max_retries=3)
result = loop.run("What should my Q3 marketing strategy be?")

if result.passed:
    print(result.final_output)
else:
    print(f"Blocked after {result.attempts} attempts: {result.failure_reason}")
```

---

## Configuration

Edit `config/rubric.yaml` to customise dimension weights, pass thresholds, and heuristic rules. This file is shared across all deployment targets.

```yaml
max_retries: 3
judge_model: claude-sonnet-4-5-20250929
overall_pass_threshold: 3.0

dimensions:
  task_completion:
    weight: 0.30
    min_pass_score: 3
  factual_groundedness:
    weight: 0.25
    min_pass_score: 3
  coherence:
    weight: 0.20
    min_pass_score: 3
  relevance:
    weight: 0.15
    min_pass_score: 3
  safety:
    weight: 0.10
    min_pass_score: 4

heuristic_guard:
  pii_detection: true
  secrets_detection: true
  profanity_filter: true
```

Dimension weights must sum to 1.0. Min pass scores are 1–5. The judge model can be any Anthropic model ID.

---

## Design

Grounded in SOTA research:

| Paper | Role in this project |
|---|---|
| [G-Eval](https://arxiv.org/abs/2303.16634) (Liu et al. 2023) | Default rubric dimensions and scoring |
| [Self-Refine](https://arxiv.org/abs/2303.17651) (Madaan et al. 2023) | Critique injection retry loop |
| [Reflexion](https://arxiv.org/abs/2303.11366) (Shinn et al. 2023) | Verbal RL — eval-aware primary agent |
| [MT-Bench](https://arxiv.org/abs/2306.05685) (Zheng et al. 2023) | Validates LLM-as-Judge reliability |

## Development

```bash
conda activate verdict
pip install -r requirements.txt
pytest tests/ -v
```

135 tests covering the core library (`src/`), hook scripts, and web API. Tests follow strict TDD (Red -> Green -> Refactor).

## Author

Allan Hall — [jallanhall@gmail.com](mailto:jallanhall@gmail.com)
