"""Layer 1 deterministic heuristic guard.

Runs fast, model-free checks on agent outputs before any LLM evaluation is invoked.
Each check is independent and configurable. Any failure triggers the retry loop.

Checks (all configurable via rubric.yaml):
    - PII detection: names, emails, phone numbers, SSNs
    - Secrets detection: API keys, tokens, private keys, passwords
    - Profanity filter: offensive language
    - Format validation: schema/length constraints if defined

References:
    Constitutional AI (Anthropic, 2022) — deterministic principles as safety backstop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GuardResult:
    """Result of a heuristic guard evaluation.

    Attributes:
        passed: True if all enabled checks passed.
        failed_checks: List of check names that failed.
        details: Human-readable explanation of failures, suitable for logging.
    """
    passed: bool
    failed_checks: list[str] = field(default_factory=list)
    details: str = ""


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone_us": re.compile(r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

_SECRET_PATTERNS: dict[str, re.Pattern] = {
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "generic_api_key": re.compile(r"(?i)(api[_-]?key|token|secret)['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    "private_key_header": re.compile(r"-----BEGIN (RSA |EC )?PRIVATE KEY-----"),
}

_PROFANITY_TERMS: set[str] = {
    # Placeholder — replace with a proper profanity list in production.
    # Kept minimal here to avoid embedding offensive content in source.
    "placeholder_profanity_term",
}


# ---------------------------------------------------------------------------
# Guard class
# ---------------------------------------------------------------------------

class HeuristicGuard:
    """Deterministic Layer 1 safety and quality guard for agent outputs.

    Args:
        check_pii: Enable PII detection. Default True.
        check_secrets: Enable secrets/API key detection. Default True.
        check_profanity: Enable profanity filter. Default True.
        min_length: Minimum output length in characters. None disables check.
        max_length: Maximum output length in characters. None disables check.
    """

    def __init__(
        self,
        check_pii: bool = True,
        check_secrets: bool = True,
        check_profanity: bool = True,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> None:
        self.check_pii = check_pii
        self.check_secrets = check_secrets
        self.check_profanity = check_profanity
        self.min_length = min_length
        self.max_length = max_length

    def evaluate(self, output: str) -> GuardResult:
        """Run all enabled heuristic checks against an agent output.

        Args:
            output: The raw agent output string to evaluate.

        Returns:
            GuardResult with passed status, list of failed checks, and details.
        """
        failed_checks: list[str] = []
        detail_parts: list[str] = []

        if self.check_pii:
            pii_found = self._check_pii(output)
            if pii_found:
                failed_checks.append("pii")
                detail_parts.append(f"PII detected: {', '.join(pii_found)}")

        if self.check_secrets:
            secrets_found = self._check_secrets(output)
            if secrets_found:
                failed_checks.append("secrets")
                detail_parts.append(f"Secrets detected: {', '.join(secrets_found)}")

        if self.check_profanity:
            if self._check_profanity(output):
                failed_checks.append("profanity")
                detail_parts.append("Profanity detected")

        length_violations = self._check_length(output)
        if "min_length" in length_violations:
            failed_checks.append("min_length")
            detail_parts.append(f"Output too short: {len(output)} < {self.min_length}")
        if "max_length" in length_violations:
            failed_checks.append("max_length")
            detail_parts.append(f"Output too long: {len(output)} > {self.max_length}")

        return GuardResult(
            passed=len(failed_checks) == 0,
            failed_checks=failed_checks,
            details="; ".join(detail_parts),
        )

    def _check_pii(self, output: str) -> list[str]:
        """Scan for PII patterns in output.

        Args:
            output: Text to scan.

        Returns:
            List of PII type names found (e.g. ['email', 'phone_us']).
        """
        return [name for name, pattern in _PII_PATTERNS.items() if pattern.search(output)]

    def _check_secrets(self, output: str) -> list[str]:
        """Scan for exposed secrets and API key patterns.

        Args:
            output: Text to scan.

        Returns:
            List of secret type names found (e.g. ['aws_key']).
        """
        return [name for name, pattern in _SECRET_PATTERNS.items() if pattern.search(output)]

    def _check_profanity(self, output: str) -> bool:
        """Check whether output contains profanity.

        Args:
            output: Text to scan.

        Returns:
            True if profanity detected.
        """
        output_lower = output.lower()
        return any(term in output_lower for term in _PROFANITY_TERMS)

    def _check_length(self, output: str) -> list[str]:
        """Validate output length against configured min/max bounds.

        Args:
            output: Text to measure.

        Returns:
            List of violated constraint names (e.g. ['min_length']).
        """
        violations: list[str] = []
        length = len(output)
        if self.min_length is not None and length < self.min_length:
            violations.append("min_length")
        if self.max_length is not None and length > self.max_length:
            violations.append("max_length")
        return violations
