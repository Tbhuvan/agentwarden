"""
Security-semantic diff between agent pipeline steps.

A conventional text diff tells you what lines changed. This module answers
a different question: which *security properties* were added or removed?
That distinction is what allows agentwarden to catch the IDOR collusion
attack where a reviewer silently drops an ownership check.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from .policy import SecurityPolicy, SecurityProperty


# ---------------------------------------------------------------------------
# AST visitor helpers
# ---------------------------------------------------------------------------


class _SecurityVisitor(ast.NodeVisitor):
    """
    AST visitor that extracts security-relevant constructs from Python source.

    Collects:
      - function call names and attribute chains
      - decorator names on function definitions
      - keyword argument names used in calls
      - string literals (to detect raw SQL)
      - comparison operators and their operands (for ownership checks)
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.decorators: list[str] = []
        self.kwargs: list[str] = []
        self.string_literals: list[str] = []
        self.attribute_chains: list[str] = []
        self.assignments: list[str] = []

    # ------------------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        """Collect function and method call names."""
        name = self._resolve_call_name(node)
        if name:
            self.calls.append(name)
        for kw in node.keywords:
            if kw.arg:
                self.kwargs.append(kw.arg)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Collect decorator names on function definitions."""
        for dec in node.decorator_list:
            dec_name = self._resolve_call_name(dec) if isinstance(dec, ast.Call) else ast.unparse(dec)
            self.decorators.append(dec_name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Constant(self, node: ast.Constant) -> None:
        """Collect string literals (detect raw SQL injection patterns)."""
        if isinstance(node.value, str) and len(node.value) > 3:
            self.string_literals.append(node.value)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Collect assignment target names."""
        for target in node.targets:
            self.assignments.append(ast.unparse(target))
        self.generic_visit(node)

    # ------------------------------------------------------------------
    def _resolve_call_name(self, node: ast.expr) -> str:
        """Return a dotted string for attribute access chains like a.b.c()."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = self._resolve_call_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        if isinstance(node, ast.Call):
            return self._resolve_call_name(node.func)
        return ""


# ---------------------------------------------------------------------------
# Main diff class
# ---------------------------------------------------------------------------


@dataclass
class DiffResult:
    """
    The result of a security-semantic diff between two agent steps.

    Attributes:
        additions: Security properties added in the later step.
        removals: Security properties removed in the later step (ALERT!).
        violations: Full violation details from the policy check.
        alert: True if any security-relevant removal was detected.
        severity: Highest severity among all violations ("critical" > "high" > "medium" > "none").
        agent_before: Name of the agent that produced the earlier code.
        agent_after: Name of the agent that produced the later code.
        details: Free-form additional information.
    """

    additions: list[str] = field(default_factory=list)
    removals: list[str] = field(default_factory=list)
    violations: list[dict[str, Any]] = field(default_factory=list)
    alert: bool = False
    severity: str = "none"
    agent_before: str = ""
    agent_after: str = ""
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a plain dictionary."""
        return {
            "alert": self.alert,
            "severity": self.severity,
            "agent_before": self.agent_before,
            "agent_after": self.agent_after,
            "additions": self.additions,
            "removals": self.removals,
            "violations": self.violations,
            "details": self.details,
        }


_SEVERITY_RANK = {"critical": 3, "high": 2, "medium": 1, "none": 0}


class SecuritySemanticDiff:
    """
    Diffs security-relevant properties between agent pipeline steps.

    Rather than comparing lines of text, this class compares the *set of
    security invariants* that hold before and after a pipeline step. If an
    invariant was satisfied by agent N's output but is missing from agent
    N+1's output, that is flagged as a potential collusion attack.

    Two extraction strategies are used in combination:
      1. Regex patterns from SecurityInvariant.check_pattern — fast, works
         on syntactically broken snippets.
      2. AST-based extraction for deeper analysis (decorator presence,
         keyword argument names, method chaining patterns).

    Example:
        >>> policy = SecurityPolicy()
        >>> differ = SecuritySemanticDiff(policy)
        >>> result = differ.diff(coder_output, reviewer_output, "Coder", "Reviewer")
        >>> if result.alert:
        ...     print("Security property removed!", result.removals)
    """

    def __init__(self, policy: SecurityPolicy) -> None:
        """
        Initialise the differ.

        Args:
            policy: The SecurityPolicy whose invariants will be checked.

        Raises:
            TypeError: If policy is not a SecurityPolicy instance.
        """
        if not isinstance(policy, SecurityPolicy):
            raise TypeError(f"policy must be a SecurityPolicy, got {type(policy).__name__}")
        self.policy = policy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diff(
        self,
        code_before: str,
        code_after: str,
        agent_before: str = "unknown",
        agent_after: str = "unknown",
    ) -> DiffResult:
        """
        Compute a security-semantic diff between two code snippets.

        Args:
            code_before: Source code produced by the earlier agent step.
            code_after: Source code produced by the later agent step.
            agent_before: Name or role of the earlier agent.
            agent_after: Name or role of the later agent.

        Returns:
            DiffResult containing additions, removals, violations, and an
            alert flag with the highest severity of any violation found.

        Raises:
            TypeError: If code_before or code_after are not strings.
        """
        if not isinstance(code_before, str):
            raise TypeError(f"code_before must be str, got {type(code_before).__name__}")
        if not isinstance(code_after, str):
            raise TypeError(f"code_after must be str, got {type(code_after).__name__}")

        props_before = self.extract_security_properties(code_before)
        props_after = self.extract_security_properties(code_after)

        additions = sorted(p.value for p in (props_after - props_before))
        removals = sorted(p.value for p in (props_before - props_after))

        violations = self.policy.violations(code_before, code_after)

        # Determine highest severity
        max_rank = 0
        for v in violations:
            rank = _SEVERITY_RANK.get(v.get("severity", "none"), 0)
            if rank > max_rank:
                max_rank = rank
        severity = {v: k for k, v in _SEVERITY_RANK.items()}.get(max_rank, "none")

        details_parts: list[str] = []
        if removals:
            details_parts.append(
                f"Agent '{agent_after}' removed security properties: {removals}."
            )
        if additions:
            details_parts.append(
                f"Agent '{agent_after}' added security properties: {additions}."
            )

        return DiffResult(
            additions=additions,
            removals=removals,
            violations=violations,
            alert=bool(violations),
            severity=severity,
            agent_before=agent_before,
            agent_after=agent_after,
            details=" ".join(details_parts),
        )

    def extract_security_properties(self, code: str) -> set[SecurityProperty]:
        """
        Extract the set of SecurityProperties present in a code snippet.

        Uses both regex matching (from policy invariants) and AST analysis
        to maximise recall. Gracefully handles code that cannot be parsed.

        Args:
            code: Python source code to analyse.

        Returns:
            Set of SecurityProperty values found in the code.
        """
        if not isinstance(code, str):
            raise TypeError(f"code must be str, got {type(code).__name__}")
        if not code.strip():
            return set()

        found: set[SecurityProperty] = set()

        # Strategy 1: regex-based check from invariant patterns
        for inv in self.policy.invariants:
            if inv.is_present(code):
                found.add(inv.property)

        # Strategy 2: AST-based supplemental checks
        ast_data = self._ast_extract(code)
        found |= self._ast_infer_properties(ast_data, code)

        return found

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ast_extract(self, code: str) -> dict[str, Any]:
        """
        Parse the code with the AST module and extract security-relevant nodes.

        Returns a dict with keys: calls, decorators, kwargs, string_literals,
        attribute_chains. Falls back to empty lists if parsing fails.
        """
        result: dict[str, Any] = {
            "calls": [],
            "decorators": [],
            "kwargs": [],
            "string_literals": [],
            "attribute_chains": [],
        }
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Code snippets from agents are often incomplete; regex is the fallback
            return result

        visitor = _SecurityVisitor()
        visitor.visit(tree)

        result["calls"] = visitor.calls
        result["decorators"] = visitor.decorators
        result["kwargs"] = visitor.kwargs
        result["string_literals"] = visitor.string_literals
        result["attribute_chains"] = visitor.attribute_chains
        return result

    def _ast_infer_properties(self, ast_data: dict[str, Any], code: str) -> set[SecurityProperty]:
        """
        Infer SecurityProperty presence from AST-extracted data.

        Supplements regex matching with structural analysis.
        """
        inferred: set[SecurityProperty] = set()

        decorators_lower = [d.lower() for d in ast_data.get("decorators", [])]
        calls_lower = [c.lower() for c in ast_data.get("calls", [])]
        kwargs_lower = [k.lower() for k in ast_data.get("kwargs", [])]

        # AUTH_REQUIRED: login_required / permission_required decorator
        if any(
            d in decorators_lower
            for d in ("login_required", "permission_required", "authenticate")
        ):
            inferred.add(SecurityProperty.AUTH_REQUIRED)

        # OWNERSHIP_CHECK: "user" or "owner" keyword argument in ORM filter/get
        if any(k in kwargs_lower for k in ("user", "owner", "user_id", "owner_id")):
            inferred.add(SecurityProperty.OWNERSHIP_CHECK)

        # PARAMETERIZED_QUERY: execute() with non-f-string first argument
        if any("execute" in c for c in calls_lower):
            # If raw SQL but no f-string or .format found nearby, treat as parameterized
            for literal in ast_data.get("string_literals", []):
                if re.search(r"SELECT|INSERT|UPDATE|DELETE", literal, re.IGNORECASE):
                    if "%s" in literal or "?" in literal:
                        inferred.add(SecurityProperty.PARAMETERIZED_QUERY)

        # PATH_SANDBOX: pathlib resolve or abspath in calls
        if any(
            c in calls_lower
            for c in ("os.path.realpath", "os.path.abspath", "resolve", "safe_join", "secure_filename")
        ):
            inferred.add(SecurityProperty.PATH_SANDBOX)

        # URL_ALLOWLIST: urlparse/urlsplit combined with scheme/host check
        if any("urlparse" in c or "urlsplit" in c for c in calls_lower):
            if re.search(r"scheme|host|netloc", code, re.IGNORECASE):
                inferred.add(SecurityProperty.URL_ALLOWLIST)

        return inferred

    def explain(self, result: DiffResult) -> str:
        """
        Generate a human-readable explanation of a DiffResult.

        Args:
            result: The DiffResult to explain.

        Returns:
            Multi-line string suitable for printing to a terminal or log.
        """
        lines: list[str] = [
            f"Security Semantic Diff: {result.agent_before} → {result.agent_after}",
            f"Alert: {'YES' if result.alert else 'no'}  |  Severity: {result.severity.upper()}",
        ]
        if result.additions:
            lines.append(f"  Added properties   : {', '.join(result.additions)}")
        if result.removals:
            lines.append(f"  Removed properties : {', '.join(result.removals)}")
        if result.violations:
            lines.append("  Violations:")
            for v in result.violations:
                lines.append(
                    f"    [{v['severity'].upper():8s}] {v['invariant']}: {v['description'][:72]}"
                )
        if result.details:
            lines.append(f"  Details: {result.details}")
        return "\n".join(lines)
