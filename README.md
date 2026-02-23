# eval-harness

> A portable two-layer LLM evaluation pipeline for AI agent outputs — available as a Claude Code plugin and standalone Python library.

## Overview

`eval-harness` wraps any AI agent with automatic quality evaluation before outputs reach users or downstream systems. It implements two evaluation layers:

**Layer 1 — Heuristic Guard** (deterministic, model-free): Catches PII, exposed secrets, profanity, and format violations instantly, without burning tokens.

**Layer 2 — LLM-as-Judge**: A separate judge agent scores outputs on a [G-Eval](https://arxiv.org/abs/2303.16634)-inspired rubric (task completion, factual groundedness, coherence, relevance, safety). On failure, structured critique is injected back into the primary agent following the [Self-Refine](https://arxiv.org/abs/2303.17651) pattern. Up to `max_retries` (default: 3) iterations before a pass/fail hook fires.

## Installation

**As a Claude Code plugin:**
```bash
/plugin install eval-harness
```

**As a standalone Python library:**
```bash
pip install eval-harness  # coming soon
```

## Usage

### Plugin commands

```
/eval-harness:eval-run              — Evaluate the most recent agent output
/eval-harness:eval-configure        — Adjust rubric, max_retries, heuristic rules
/eval-harness:eval-report           — View eval log summary and failure trends
```

### Python library

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

## Configuration

Edit `config/rubric.yaml` to customise dimension weights, pass thresholds, and heuristic rules:

```yaml
max_retries: 3
overall_pass_threshold: 3.0

dimensions:
  task_completion:
    weight: 0.30
    min_pass_score: 3
  # ...
```

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
conda create -n eval-harness python=3.11 && conda activate eval-harness
pip install -r requirements.txt
pytest tests/ -v
```

Tests follow strict TDD (Red → Green → Refactor). All `src/` modules start as stubs with failing tests — implement each function to make its tests pass.

## Author

Allan Hall — [jallanhall@gmail.com](mailto:jallanhall@gmail.com)
