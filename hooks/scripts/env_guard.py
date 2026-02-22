#!/usr/bin/env python3
"""PreToolUse hook: blocks pip/conda install commands when no conda environment is active.

Reads tool input from stdin as JSON, inspects the bash command for install patterns,
and checks CONDA_DEFAULT_ENV. Exits 2 with a blocking message if an install is
attempted outside a conda environment.

Exit codes:
    0 — allow the tool call to proceed
    2 — block the tool call (stderr is fed back to Claude as a system message)
"""

import json
import os
import re
import sys


INSTALL_PATTERNS = [
    r"\bpip\s+install\b",
    r"\bpip3\s+install\b",
    r"\bconda\s+install\b",
    r"\bconda\s+env\s+create\b",
]


def is_install_command(command: str) -> bool:
    """Check whether a bash command contains a package install operation.

    Args:
        command: The bash command string to inspect.

    Returns:
        True if the command contains a pip or conda install pattern.
    """
    return any(re.search(pattern, command) for pattern in INSTALL_PATTERNS)


def conda_env_is_active() -> bool:
    """Check whether a conda environment is currently active.

    Returns:
        True if CONDA_DEFAULT_ENV is set and non-empty.
    """
    env = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
    return bool(env) and env != "base"


def main() -> None:
    """Main hook entrypoint. Reads stdin, checks command, blocks if unsafe."""
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        # Cannot parse input — allow through rather than false-positive block
        sys.exit(0)

    tool_input = payload.get("tool_input", {})
    command = tool_input.get("command", "")

    if not is_install_command(command):
        sys.exit(0)

    if conda_env_is_active():
        sys.exit(0)

    active_env = os.environ.get("CONDA_DEFAULT_ENV", "none")
    message = (
        f"[env-guard] Package install blocked: no active conda environment detected "
        f"(CONDA_DEFAULT_ENV='{active_env}'). "
        f"Please activate or create a conda environment before installing packages. "
        f"Example: `conda activate my-env` or `conda create -n my-env python=3.11 && conda activate my-env`."
    )
    print(message, file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
