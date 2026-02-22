# eval-harness-plugin

## What This Is
A Claude Code plugin implementing a two-layer LLM evaluation pipeline for AI agent outputs.
Layer 1: deterministic heuristic guard. Layer 2: LLM-as-Judge (G-Eval rubric) with Self-Refine retry loop.

## Architecture
- `src/` — standalone Python library, zero Claude-specific imports. Portable to any agent framework.
- Plugin wiring (agents/, commands/, hooks/, skills/) is the Claude Code integration layer on top.

## Coding Standards
**Language:** Python 3.11+. **Tests:** pytest.

**TDD — Red → Green → Refactor (strict):**
- Write failing test first. Run it — confirm it fails. Write minimal code to pass. Refactor.
- Tests import from `src/`. Never redefine logic inside test files.

**Docstrings:** Google style on all functions and classes.
**Modularity:** Single-responsibility functions. No monoliths.
**Environment:** Always work inside a conda environment. Never install to root Python.

## File Structure
```
src/          ← implementation (portable Python library)
tests/        ← mirrors src/ — pytest suite
config/       ← rubric.yaml and YAML configs
agents/       ← judge-agent subagent
skills/       ← eval-methodology skill
commands/     ← /eval:run /eval:configure /eval:report
hooks/        ← PostToolUse eval + PreToolUse env-guard
docs/         ← design docs, ADRs
```

## Key References
- G-Eval (Liu et al. 2023) — rubric framework
- Self-Refine (Madaan et al. 2023) — critique injection loop
- Reflexion (Shinn et al. 2023) — verbal RL pattern
- MT-Bench (Zheng et al. 2023) — LLM-as-Judge validation
