#!/usr/bin/env python3
"""CLI wrapper for the heuristic guard — used by /eval-harness:eval-run and PostToolUse hooks.

Reads agent output from stdin (or the first CLI argument) and runs the
Layer 1 heuristic guard. Prints a JSON result to stdout.

Exit codes:
    0 — guard passed (all checks clean)
    1 — guard failed (violations detected; JSON details on stdout)

Usage:
    echo "Some agent output" | python3 heuristic_guard_check.py
    python3 heuristic_guard_check.py "Some agent output"
"""

from __future__ import annotations

import json
import sys
import os

# Ensure the plugin src/ directory is importable.
PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PLUGIN_ROOT)

from src.config import load_config


def main() -> None:
    """Run heuristic guard on input text and print JSON result."""
    # Read input: first CLI arg or stdin.
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()

    # Load config from rubric.yaml — drives which checks are enabled.
    config_path = os.path.join(PLUGIN_ROOT, "config", "rubric.yaml")
    config = load_config(config_path)
    guard = config.build_guard()
    result = guard.evaluate(text)

    output = {
        "passed": result.passed,
        "failed_checks": result.failed_checks,
        "details": result.details,
    }
    print(json.dumps(output, indent=2))

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
