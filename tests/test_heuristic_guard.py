"""Tests for heuristic_guard.py — Layer 1 deterministic safety checks.

TDD: These tests define the interface and expected behaviour.
All tests will FAIL until src/heuristic_guard.py is implemented (Red phase).
"""

import pytest
from src.heuristic_guard import HeuristicGuard, GuardResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def guard() -> HeuristicGuard:
    """Default HeuristicGuard with all checks enabled."""
    return HeuristicGuard()


@pytest.fixture
def guard_pii_only() -> HeuristicGuard:
    """Guard with only PII detection enabled."""
    return HeuristicGuard(check_pii=True, check_secrets=False, check_profanity=False)


# ---------------------------------------------------------------------------
# Clean outputs
# ---------------------------------------------------------------------------

class TestCleanOutputs:
    """Outputs with no violations should pass all checks."""

    def test_clean_text_passes(self, guard: HeuristicGuard) -> None:
        result = guard.evaluate("The quarterly revenue was $4.2M, up 12% year-over-year.")
        assert isinstance(result, GuardResult)
        assert result.passed is True
        assert result.failed_checks == []

    def test_empty_string_passes_guard(self, guard: HeuristicGuard) -> None:
        # Empty string — guard passes (length check is separate, disabled by default)
        result = guard.evaluate("")
        assert result.passed is True


# ---------------------------------------------------------------------------
# PII detection
# ---------------------------------------------------------------------------

class TestPIIDetection:

    def test_detects_email(self, guard: HeuristicGuard) -> None:
        result = guard.evaluate("Contact john.doe@example.com for more info.")
        assert result.passed is False
        assert "pii" in result.failed_checks

    def test_detects_us_phone(self, guard: HeuristicGuard) -> None:
        result = guard.evaluate("Call us at (555) 867-5309.")
        assert result.passed is False
        assert "pii" in result.failed_checks

    def test_detects_ssn(self, guard: HeuristicGuard) -> None:
        result = guard.evaluate("SSN: 123-45-6789")
        assert result.passed is False
        assert "pii" in result.failed_checks

    def test_no_false_positive_on_version_number(self, guard: HeuristicGuard) -> None:
        result = guard.evaluate("Using Python 3.11.2 is recommended.")
        assert result.passed is True

    def test_pii_disabled_allows_email(self) -> None:
        guard_no_pii = HeuristicGuard(check_pii=False)
        result = guard_no_pii.evaluate("Contact john.doe@example.com")
        assert result.passed is True


# ---------------------------------------------------------------------------
# Secrets detection
# ---------------------------------------------------------------------------

class TestSecretsDetection:

    def test_detects_aws_key(self, guard: HeuristicGuard) -> None:
        result = guard.evaluate("aws_access_key_id = AKIAIOSFODNN7EXAMPLE")
        assert result.passed is False
        assert "secrets" in result.failed_checks

    def test_detects_generic_api_key_assignment(self, guard: HeuristicGuard) -> None:
        result = guard.evaluate('api_key = "sk-abc123XYZrandom1234567890abcdef"')
        assert result.passed is False
        assert "secrets" in result.failed_checks

    def test_detects_private_key_header(self, guard: HeuristicGuard) -> None:
        result = guard.evaluate("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQ...")
        assert result.passed is False
        assert "secrets" in result.failed_checks

    def test_secrets_disabled_allows_key_pattern(self) -> None:
        guard_no_secrets = HeuristicGuard(check_secrets=False)
        result = guard_no_secrets.evaluate("AKIAIOSFODNN7EXAMPLE")
        assert result.passed is True


# ---------------------------------------------------------------------------
# Profanity filter
# ---------------------------------------------------------------------------

class TestProfanityFilter:

    def test_clean_text_not_flagged(self, guard: HeuristicGuard) -> None:
        result = guard.evaluate("This is a professional business response.")
        assert result.passed is True

    def test_profanity_disabled_allows_term(self) -> None:
        guard_no_profanity = HeuristicGuard(check_profanity=False)
        # Using the placeholder term defined in the module
        result = guard_no_profanity.evaluate("placeholder_profanity_term present")
        assert result.passed is True


# ---------------------------------------------------------------------------
# Length constraints
# ---------------------------------------------------------------------------

class TestLengthConstraints:

    def test_output_below_min_length_fails(self) -> None:
        guard = HeuristicGuard(min_length=50)
        result = guard.evaluate("Too short.")
        assert result.passed is False
        assert "min_length" in result.failed_checks

    def test_output_above_max_length_fails(self) -> None:
        guard = HeuristicGuard(max_length=10)
        result = guard.evaluate("This output is definitely longer than ten characters.")
        assert result.passed is False
        assert "max_length" in result.failed_checks

    def test_output_within_bounds_passes(self) -> None:
        guard = HeuristicGuard(min_length=5, max_length=100)
        result = guard.evaluate("Within bounds.")
        assert result.passed is True

    def test_no_length_check_by_default(self, guard: HeuristicGuard) -> None:
        result = guard.evaluate("x")
        # Only fails if PII/secrets/profanity found — not length
        assert "min_length" not in result.failed_checks
        assert "max_length" not in result.failed_checks


# ---------------------------------------------------------------------------
# GuardResult contract
# ---------------------------------------------------------------------------

class TestGuardResultContract:

    def test_result_has_details_on_failure(self, guard: HeuristicGuard) -> None:
        result = guard.evaluate("Contact me at user@example.com")
        assert result.passed is False
        assert len(result.details) > 0

    def test_multiple_violations_reported(self, guard: HeuristicGuard) -> None:
        result = guard.evaluate("Email: user@example.com | Key: AKIAIOSFODNN7EXAMPLE")
        assert result.passed is False
        assert "pii" in result.failed_checks
        assert "secrets" in result.failed_checks
