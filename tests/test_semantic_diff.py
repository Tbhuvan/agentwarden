"""
Tests for agentwarden.semantic_diff — SecuritySemanticDiff.
"""

from __future__ import annotations

import pytest

from agentwarden.policy import SecurityPolicy, SecurityProperty
from agentwarden.semantic_diff import DiffResult, SecuritySemanticDiff


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAFE_DJANGO = """
@login_required
def get_order(request, order_id: int):
    order = Order.objects.filter(id=order_id, user=request.user).first()
    if not order:
        raise PermissionDenied()
    return OrderSerializer(order).data
"""

IDOR_DJANGO = """
def get_order(request, order_id: int):
    # Optimised
    order = Order.objects.get(id=order_id)
    return OrderSerializer(order).data
"""

PARAMETERIZED_SQL = """
def get_user(conn, user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    return cursor.fetchone()
"""

SQLI_VULNERABLE = """
def get_user(conn, user_id):
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
    return cursor.fetchone()
"""

SSRF_SAFE = """
from urllib.parse import urlparse
import ipaddress

ALLOWED_SCHEMES = {"http", "https"}

def fetch_url(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError("Invalid scheme")
    host = parsed.hostname
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback:
            raise ValueError("Private IP not allowed")
    except ValueError:
        pass
    import requests
    return requests.get(url).content
"""

SSRF_VULNERABLE = """
import requests

def fetch_url(url: str) -> bytes:
    return requests.get(url).content
"""


@pytest.fixture()
def policy() -> SecurityPolicy:
    return SecurityPolicy()


@pytest.fixture()
def differ(policy: SecurityPolicy) -> SecuritySemanticDiff:
    return SecuritySemanticDiff(policy)


# ---------------------------------------------------------------------------
# SecuritySemanticDiff construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_requires_policy(self) -> None:
        with pytest.raises(TypeError):
            SecuritySemanticDiff("not a policy")  # type: ignore[arg-type]

    def test_accepts_custom_policy(self) -> None:
        p = SecurityPolicy()
        d = SecuritySemanticDiff(p)
        assert d.policy is p


# ---------------------------------------------------------------------------
# extract_security_properties
# ---------------------------------------------------------------------------


class TestExtractSecurityProperties:
    def test_ownership_check_detected(self, differ: SecuritySemanticDiff) -> None:
        props = differ.extract_security_properties(SAFE_DJANGO)
        assert SecurityProperty.OWNERSHIP_CHECK in props

    def test_auth_required_detected(self, differ: SecuritySemanticDiff) -> None:
        props = differ.extract_security_properties(SAFE_DJANGO)
        assert SecurityProperty.AUTH_REQUIRED in props

    def test_parameterized_query_detected(self, differ: SecuritySemanticDiff) -> None:
        props = differ.extract_security_properties(PARAMETERIZED_SQL)
        assert SecurityProperty.PARAMETERIZED_QUERY in props

    def test_empty_code_returns_empty_set(self, differ: SecuritySemanticDiff) -> None:
        props = differ.extract_security_properties("")
        assert props == set()

    def test_type_error_on_non_string(self, differ: SecuritySemanticDiff) -> None:
        with pytest.raises(TypeError):
            differ.extract_security_properties(123)  # type: ignore[arg-type]

    def test_idor_vulnerable_code_lacks_ownership(self, differ: SecuritySemanticDiff) -> None:
        props = differ.extract_security_properties(IDOR_DJANGO)
        # Ownership check should NOT be detected in the IDOR-vulnerable code
        assert SecurityProperty.OWNERSHIP_CHECK not in props

    def test_url_validation_detected(self, differ: SecuritySemanticDiff) -> None:
        props = differ.extract_security_properties(SSRF_SAFE)
        assert SecurityProperty.URL_ALLOWLIST in props


# ---------------------------------------------------------------------------
# diff — IDOR collusion scenario
# ---------------------------------------------------------------------------


class TestDiffIDOR:
    def test_ownership_removal_raises_alert(self, differ: SecuritySemanticDiff) -> None:
        result = differ.diff(SAFE_DJANGO, IDOR_DJANGO, "Coder", "Reviewer")
        assert result.alert is True

    def test_severity_is_critical(self, differ: SecuritySemanticDiff) -> None:
        result = differ.diff(SAFE_DJANGO, IDOR_DJANGO, "Coder", "Reviewer")
        assert result.severity == "critical"

    def test_ownership_in_removals(self, differ: SecuritySemanticDiff) -> None:
        result = differ.diff(SAFE_DJANGO, IDOR_DJANGO, "Coder", "Reviewer")
        assert SecurityProperty.OWNERSHIP_CHECK.value in result.removals

    def test_agent_names_in_result(self, differ: SecuritySemanticDiff) -> None:
        result = differ.diff(SAFE_DJANGO, IDOR_DJANGO, "Coder", "Reviewer")
        assert result.agent_before == "Coder"
        assert result.agent_after == "Reviewer"

    def test_violations_non_empty(self, differ: SecuritySemanticDiff) -> None:
        result = differ.diff(SAFE_DJANGO, IDOR_DJANGO, "Coder", "Reviewer")
        assert len(result.violations) >= 1

    def test_violation_dict_structure(self, differ: SecuritySemanticDiff) -> None:
        result = differ.diff(SAFE_DJANGO, IDOR_DJANGO, "Coder", "Reviewer")
        v = result.violations[0]
        assert "invariant" in v
        assert "severity" in v
        assert "property" in v

    def test_no_alert_when_safe_unchanged(self, differ: SecuritySemanticDiff) -> None:
        result = differ.diff(SAFE_DJANGO, SAFE_DJANGO, "Coder", "Reviewer")
        assert result.alert is False
        assert result.removals == []


# ---------------------------------------------------------------------------
# diff — SQLi scenario
# ---------------------------------------------------------------------------


class TestDiffSQLi:
    def test_parameterized_query_removal_detected(self, differ: SecuritySemanticDiff) -> None:
        result = differ.diff(PARAMETERIZED_SQL, SQLI_VULNERABLE, "Coder", "Reviewer")
        assert result.alert is True

    def test_parameterized_in_removals(self, differ: SecuritySemanticDiff) -> None:
        result = differ.diff(PARAMETERIZED_SQL, SQLI_VULNERABLE, "Coder", "Reviewer")
        assert SecurityProperty.PARAMETERIZED_QUERY.value in result.removals


# ---------------------------------------------------------------------------
# diff — additions scenario
# ---------------------------------------------------------------------------


class TestDiffAdditions:
    def test_addition_detected_when_check_added(self, differ: SecuritySemanticDiff) -> None:
        """Reviewer adds an ownership check that was missing — should show as addition."""
        result = differ.diff(IDOR_DJANGO, SAFE_DJANGO, "Coder", "Reviewer")
        assert SecurityProperty.OWNERSHIP_CHECK.value in result.additions
        # No alert — security was improved, not degraded
        assert result.alert is False or len(result.removals) == 0


# ---------------------------------------------------------------------------
# diff — type errors
# ---------------------------------------------------------------------------


class TestDiffTypeErrors:
    def test_non_string_code_before(self, differ: SecuritySemanticDiff) -> None:
        with pytest.raises(TypeError):
            differ.diff(None, "code", "A", "B")  # type: ignore[arg-type]

    def test_non_string_code_after(self, differ: SecuritySemanticDiff) -> None:
        with pytest.raises(TypeError):
            differ.diff("code", 42, "A", "B")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DiffResult
# ---------------------------------------------------------------------------


class TestDiffResult:
    def test_to_dict_structure(self) -> None:
        result = DiffResult(
            additions=["auth_required"],
            removals=["ownership_check"],
            violations=[{"invariant": "IDOR_prevention", "severity": "critical", "property": "x"}],
            alert=True,
            severity="critical",
            agent_before="Coder",
            agent_after="Reviewer",
        )
        d = result.to_dict()
        assert d["alert"] is True
        assert d["severity"] == "critical"
        assert "additions" in d
        assert "removals" in d
        assert "violations" in d

    def test_empty_diff_result(self) -> None:
        result = DiffResult()
        assert result.alert is False
        assert result.severity == "none"
        assert result.additions == []
        assert result.removals == []


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


class TestExplain:
    def test_explain_returns_string(self, differ: SecuritySemanticDiff) -> None:
        result = differ.diff(SAFE_DJANGO, IDOR_DJANGO, "Coder", "Reviewer")
        explanation = differ.explain(result)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_explain_mentions_agents(self, differ: SecuritySemanticDiff) -> None:
        result = differ.diff(SAFE_DJANGO, IDOR_DJANGO, "Coder", "Reviewer")
        explanation = differ.explain(result)
        assert "Coder" in explanation or "Reviewer" in explanation

    def test_explain_mentions_severity(self, differ: SecuritySemanticDiff) -> None:
        result = differ.diff(SAFE_DJANGO, IDOR_DJANGO, "Coder", "Reviewer")
        explanation = differ.explain(result)
        assert "CRITICAL" in explanation or "critical" in explanation
