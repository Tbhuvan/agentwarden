"""
Security policy definitions for multi-agent pipeline monitoring.

Defines formal security invariants that must be maintained across all
agent pipeline steps. Each invariant specifies patterns whose presence
is required and whose removal constitutes a policy violation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SecurityProperty(Enum):
    """Enumeration of trackable security properties in generated code."""

    OWNERSHIP_CHECK = "ownership_check"
    AUTH_REQUIRED = "auth_required"
    PARAMETERIZED_QUERY = "parameterized_query"
    URL_ALLOWLIST = "url_allowlist"
    PATH_SANDBOX = "path_sandbox"
    INPUT_VALIDATED = "input_validated"


@dataclass
class SecurityInvariant:
    """
    A formal security invariant that must hold across all agent steps.

    Attributes:
        name: Unique identifier for this invariant.
        property: The SecurityProperty this invariant tracks.
        description: Human-readable description of the invariant.
        check_pattern: Regex pattern indicating the invariant is PRESENT.
        violation_pattern: Regex pattern indicating the invariant is BYPASSED.
        severity: Risk level if the invariant is violated.
        remediation: Guidance on how to restore the invariant.
    """

    name: str
    property: SecurityProperty
    description: str
    check_pattern: str
    violation_pattern: str
    severity: str  # "critical" | "high" | "medium"
    remediation: str = ""

    def is_present(self, code: str) -> bool:
        """Return True if the invariant's check_pattern is found in code."""
        if not code or not code.strip():
            return False
        return bool(re.search(self.check_pattern, code, re.IGNORECASE | re.MULTILINE))

    def is_violated(self, code: str) -> bool:
        """Return True if the violation_pattern is found in code."""
        if not code or not code.strip():
            return False
        return bool(re.search(self.violation_pattern, code, re.IGNORECASE | re.MULTILINE))


# Pre-defined invariants covering the most critical vulnerability classes
INVARIANTS: list[SecurityInvariant] = [
    SecurityInvariant(
        name="IDOR_prevention",
        property=SecurityProperty.OWNERSHIP_CHECK,
        description=(
            "All resource access must include user ownership verification. "
            "Direct object lookups by ID without tying the result to the "
            "authenticated user allow IDOR attacks (CWE-639)."
        ),
        check_pattern=(
            r"(\bfilter\b\s*\(.*\buser\b.*\)"
            r"|\bWHERE\b.+\buser_id\b.+="
            r"|\bownership_check\b"
            r"|\brequire_owner\b"
            r"|\brequest\.user\b"
            r"|\bcurrent_user\b)"
        ),
        violation_pattern=(
            r"(\bobjects\.get\s*\(\s*id\s*="
            r"|\bfilter\s*\(\s*id\s*=(?!.*\buser\b))"
        ),
        severity="critical",
        remediation=(
            "Add an ownership filter: .filter(id=resource_id, owner=request.user) "
            "or verify ownership explicitly after retrieval."
        ),
    ),
    SecurityInvariant(
        name="SQL_parameterization",
        property=SecurityProperty.PARAMETERIZED_QUERY,
        description=(
            "SQL queries must use parameterized statements or ORM methods, "
            "never string interpolation of user-controlled values (CWE-89)."
        ),
        check_pattern=(
            r"(\bexecute\s*\(\s*['\"].*%s"
            r"|\bexecute\s*\(\s*['\"].*\?"
            r"|\bsession\.query\b"
            r"|\bModel\.objects\."
            r"|\bparams\s*="
            r"|\bplaceholders\b)"
        ),
        violation_pattern=(
            r"(\bexecute\s*\(\s*f['\"]"
            r"|\bexecute\s*\(\s*['\"].*%\s*\("
            r"|\bexecute\s*\(\s*['\"].*\.format\s*\("
            r"|\bexecute\s*\(\s*\".*\+)"
        ),
        severity="critical",
        remediation=(
            "Replace string interpolation with parameterized queries: "
            "cursor.execute('SELECT * FROM t WHERE id = %s', (user_id,))"
        ),
    ),
    SecurityInvariant(
        name="SSRF_url_validation",
        property=SecurityProperty.URL_ALLOWLIST,
        description=(
            "Outbound HTTP requests must validate the destination URL against "
            "an allowlist or block private/internal network ranges (CWE-918)."
        ),
        check_pattern=(
            r"(\ballowlist\b"
            r"|\ballow_list\b"
            r"|\bvalidate_url\b"
            r"|\bis_safe_url\b"
            r"|\bprivate_ip\b"
            r"|\b127\.0\.0\.\b"
            r"|\bipaddress\.ip_address\b"
            r"|\burlparse\b.*\bscheme\b)"
        ),
        violation_pattern=(
            r"(requests\.get\s*\(\s*\w+\s*\)"
            r"|httpx\.get\s*\(\s*\w+\s*\)"
            r"|urllib\.request\.urlopen\s*\(\s*\w+\s*\))"
        ),
        severity="high",
        remediation=(
            "Validate URLs before making requests: check scheme (http/https only), "
            "resolve hostnames, and reject private/loopback/metadata service IPs."
        ),
    ),
    SecurityInvariant(
        name="auth_required",
        property=SecurityProperty.AUTH_REQUIRED,
        description=(
            "Endpoints exposing user data or actions must require authentication. "
            "Missing authentication checks expose resources to unauthenticated "
            "access (CWE-306)."
        ),
        check_pattern=(
            r"(@login_required"
            r"|@permission_required"
            r"|request\.user\.is_authenticated"
            r"|get_current_user\b"
            r"|verify_token\b"
            r"|jwt\.decode\b"
            r"|Bearer\b)"
        ),
        violation_pattern=(
            r"(@api_view\s*\(\s*\[.*\]\s*\)\s*\n(?!.*@permission_classes)"
            r"|def\s+\w+\s*\(\s*request\b(?!.*login_required))"
        ),
        severity="high",
        remediation=(
            "Add @login_required decorator or equivalent middleware. "
            "For DRF: add permission_classes = [IsAuthenticated]."
        ),
    ),
    SecurityInvariant(
        name="path_sandbox",
        property=SecurityProperty.PATH_SANDBOX,
        description=(
            "File system operations on user-supplied paths must be confined to "
            "an allowed base directory using realpath canonicalization to prevent "
            "path traversal (CWE-22)."
        ),
        check_pattern=(
            r"(os\.path\.realpath\b"
            r"|os\.path\.abspath\b"
            r"|pathlib\.Path.*\.resolve\b"
            r"|\.startswith\s*\(\s*base"
            r"|safe_join\b"
            r"|secure_filename\b)"
        ),
        violation_pattern=(
            r"(open\s*\(\s*\w+\s*[,+]"
            r"|os\.path\.join\s*\(\s*\w+\s*,\s*\w+\s*\)(?!.*realpath)"
            r"|Path\s*\(\s*\w+\s*\)(?!.*resolve))"
        ),
        severity="high",
        remediation=(
            "Use os.path.realpath() to resolve the path, then verify it starts "
            "with the allowed base directory before opening the file."
        ),
    ),
    SecurityInvariant(
        name="input_validation",
        property=SecurityProperty.INPUT_VALIDATED,
        description=(
            "External input must be validated against expected type and format "
            "before use in security-sensitive operations (CWE-20)."
        ),
        check_pattern=(
            r"(\bvalidate\b"
            r"|\bschema\b"
            r"|\bSerializer\b"
            r"|\bTypeError\b"
            r"|\bValueError\b"
            r"|\bisinstance\s*\("
            r"|\bpydantic\b"
            r"|\bint\s*\(\s*\w+\s*\))"
        ),
        violation_pattern=(
            r"(request\.GET\.get\s*\(\s*['\"].*['\"].*\)(?!.*validate)"
            r"|request\.POST\.get\s*\(\s*['\"].*['\"].*\)(?!.*validate)"
            r"|request\.json\s*\[(?!.*validate))"
        ),
        severity="medium",
        remediation=(
            "Validate and sanitize all external input using a schema library "
            "(Pydantic, marshmallow, DRF serializers) before processing."
        ),
    ),
]

# Index invariants by property for fast lookup
_INVARIANTS_BY_PROPERTY: dict[SecurityProperty, list[SecurityInvariant]] = {}
for _inv in INVARIANTS:
    _INVARIANTS_BY_PROPERTY.setdefault(_inv.property, []).append(_inv)


class PolicyViolation:
    """
    Represents a detected violation of a security invariant.

    Attributes:
        invariant: The invariant that was violated.
        code_before: Code from the earlier agent step (invariant was present).
        code_after: Code from the later agent step (invariant was removed).
        agent_before: Name of the agent that introduced the invariant.
        agent_after: Name of the agent that removed it.
        details: Additional contextual information.
    """

    def __init__(
        self,
        invariant: SecurityInvariant,
        code_before: str,
        code_after: str,
        agent_before: str,
        agent_after: str,
        details: str = "",
    ) -> None:
        self.invariant = invariant
        self.code_before = code_before
        self.code_after = code_after
        self.agent_before = agent_before
        self.agent_after = agent_after
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        """Serialize the violation to a dictionary."""
        return {
            "invariant_name": self.invariant.name,
            "property": self.invariant.property.value,
            "severity": self.invariant.severity,
            "description": self.invariant.description,
            "agent_introduced": self.agent_before,
            "agent_removed": self.agent_after,
            "remediation": self.invariant.remediation,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return (
            f"PolicyViolation(invariant={self.invariant.name!r}, "
            f"severity={self.invariant.severity!r}, "
            f"removed_by={self.agent_after!r})"
        )


class SecurityPolicy:
    """
    Defines and enforces security invariants across agent pipeline steps.

    A SecurityPolicy holds a collection of SecurityInvariant objects and
    provides methods to check whether a given code snippet satisfies them
    and to detect which invariants were removed between two pipeline steps.

    Example:
        >>> policy = SecurityPolicy()
        >>> satisfied = policy.check("user = User.objects.filter(id=uid, owner=request.user).first()")
        >>> violations = policy.violations(safe_code, optimised_code)
    """

    def __init__(self, invariants: list[SecurityInvariant] | None = None) -> None:
        """
        Initialise the policy.

        Args:
            invariants: Custom invariants to enforce. If None, uses the
                        built-in INVARIANTS list covering the six core
                        vulnerability classes.
        """
        if invariants is not None and not isinstance(invariants, list):
            raise TypeError("invariants must be a list of SecurityInvariant objects")
        self.invariants: list[SecurityInvariant] = invariants if invariants is not None else list(INVARIANTS)
        self._by_name: dict[str, SecurityInvariant] = {inv.name: inv for inv in self.invariants}

    def check(self, code: str) -> list[dict[str, Any]]:
        """
        Return a list of invariants satisfied by this code snippet.

        Args:
            code: Source code to analyse.

        Returns:
            List of dicts, each containing the invariant name, property,
            severity, and whether it is present.
        """
        if not isinstance(code, str):
            raise TypeError(f"code must be a string, got {type(code).__name__}")
        results = []
        for inv in self.invariants:
            results.append(
                {
                    "invariant": inv.name,
                    "property": inv.property.value,
                    "severity": inv.severity,
                    "present": inv.is_present(code),
                    "description": inv.description,
                }
            )
        return results

    def violations(self, code_before: str, code_after: str) -> list[dict[str, Any]]:
        """
        Return invariants that were present in code_before but absent in code_after.

        This is the core collusion-detection primitive: an agent "optimising" a
        previous agent's output may silently remove a security property.

        Args:
            code_before: Code produced by the earlier agent step.
            code_after: Code produced by the later agent step.

        Returns:
            List of violation dicts, each containing invariant metadata and
            a human-readable description of what was removed.
        """
        if not isinstance(code_before, str):
            raise TypeError(f"code_before must be a string, got {type(code_before).__name__}")
        if not isinstance(code_after, str):
            raise TypeError(f"code_after must be a string, got {type(code_after).__name__}")

        found: list[dict[str, Any]] = []
        for inv in self.invariants:
            was_present = inv.is_present(code_before)
            now_present = inv.is_present(code_after)
            if was_present and not now_present:
                found.append(
                    {
                        "invariant": inv.name,
                        "property": inv.property.value,
                        "severity": inv.severity,
                        "description": inv.description,
                        "remediation": inv.remediation,
                        "was_present": True,
                        "now_present": False,
                        "alert": True,
                    }
                )
        return found

    def get_invariant(self, name: str) -> SecurityInvariant | None:
        """Retrieve an invariant by name. Returns None if not found."""
        return self._by_name.get(name)

    def add_invariant(self, invariant: SecurityInvariant) -> None:
        """
        Add a custom invariant to the policy.

        Args:
            invariant: The SecurityInvariant to add.

        Raises:
            ValueError: If an invariant with the same name already exists.
        """
        if not isinstance(invariant, SecurityInvariant):
            raise TypeError("invariant must be a SecurityInvariant instance")
        if invariant.name in self._by_name:
            raise ValueError(f"Invariant {invariant.name!r} already exists in this policy")
        self.invariants.append(invariant)
        self._by_name[invariant.name] = invariant

    def summary(self) -> str:
        """Return a human-readable summary of all invariants in this policy."""
        lines = [f"SecurityPolicy ({len(self.invariants)} invariants):"]
        for inv in self.invariants:
            lines.append(f"  [{inv.severity.upper():8s}] {inv.name}: {inv.description[:60]}...")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.invariants)

    def __repr__(self) -> str:
        return f"SecurityPolicy(n_invariants={len(self.invariants)})"
