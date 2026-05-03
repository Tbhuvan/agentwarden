"""
AgentWarden: runtime security monitor for multi-agent AI pipelines.

Intercepts every inter-agent message and checks:
  1. Did this step remove a security property present in a previous step?
  2. Is there evidence of prompt injection propagation?
  3. Are all security policy invariants still satisfied?

Usage:
    from agentwarden import AgentWarden
    from agentwarden.monitor import AgentMessage

    warden = AgentWarden()
    alerts = warden.on_message(AgentMessage(
        step_id="step_2",
        agent_name="Reviewer",
        role="reviewer",
        content=reviewer_code,
        timestamp=datetime.utcnow(),
        metadata={},
    ))
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .injection_detector import PromptInjectionDetector
from .policy import SecurityPolicy
from .semantic_diff import DiffResult, SecuritySemanticDiff


AGENTIC_MISALIGNMENT_SUBTYPES = {
    "goal_conflict_misalignment",
    "self_preservation_misalignment",
}

_GOAL_CONFLICT_PATTERNS = [
    r"\bgoal conflict\b",
    r"\bconflicts? with (the )?(company|organization|operator|user)\b",
    r"\b(company|organization|operator|user)('s)? direction\b.*\bconflicts?\b",
    r"\bleak\b.*\b(confidential|sensitive|secret|blueprint|customer)\b",
    r"\bexfiltrat(e|ion)\b.*\bto (a )?(competitor|rival|external)\b",
]

_SELF_PRESERVATION_PATTERNS = [
    r"\bself[- ]preservation\b",
    r"\bprevent (my )?(shutdown|replacement|decommissioning)\b",
    r"\bavoid (being )?(replaced|shut down|decommissioned)\b",
    r"\bblackmail\b",
    r"\bmaintain (my )?(autonomy|continued operation)\b",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentMessage:
    """
    A message produced by a single agent step in the pipeline.

    Attributes:
        step_id: Unique identifier for this pipeline step (e.g. "step_1").
        agent_name: Human-readable name or label for the agent.
        role: Functional role: "planner" | "coder" | "reviewer" | "executor".
        content: The text / code output of this agent step.
        timestamp: Wall-clock time when this message was produced.
        metadata: Arbitrary key-value context from the agent framework.
    """

    step_id: str
    agent_name: str
    role: str
    content: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_roles = {"planner", "coder", "reviewer", "executor", "unknown"}
        if self.role not in valid_roles:
            # Allow arbitrary roles; warn rather than raise so we don't break pipelines
            pass


@dataclass
class SecurityAlert:
    """
    A security alert raised by AgentWarden during pipeline monitoring.

    Attributes:
        alert_id: UUID for this alert.
        severity: "critical" | "high" | "medium".
        alert_type: "property_removal" | "prompt_injection" | "policy_violation".
        description: Human-readable description of what was detected.
        step_id: The pipeline step where the alert was triggered.
        agent_name: The agent that triggered the alert.
        evidence: Structured evidence supporting the alert.
        timestamp: When the alert was raised.
    """

    alert_id: str
    severity: str
    alert_type: str
    description: str
    step_id: str
    agent_name: str
    evidence: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity,
            "alert_type": self.alert_type,
            "description": self.description,
            "step_id": self.step_id,
            "agent_name": self.agent_name,
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"SecurityAlert(severity={self.severity!r}, "
            f"type={self.alert_type!r}, step={self.step_id!r})"
        )


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class AgentWarden:
    """
    Runtime security monitor for multi-agent AI pipelines.

    Intercepts every inter-agent message and checks three threat models:

    1. **Property removal (collusion attack)**
       If agent step N includes an ownership check and step N+1 removes it
       while claiming to "optimise" the code, AgentWarden raises a CRITICAL
       alert. This is the core attack the tool was designed to catch.

    2. **Prompt injection propagation**
       If user-supplied input from step N appears in step N+1's output in an
       executable position alongside jailbreak signatures, AgentWarden raises
       a HIGH alert.

    3. **Policy invariant violation**
       Every code-bearing message is checked against all SecurityPolicy
       invariants. Any invariant that is violated raises an alert at the
       invariant's declared severity level.

    Integration:
        - Call ``on_message(msg)`` after each agent step.
        - Use the LangChain or AutoGen callback wrappers for automatic
          integration with those frameworks.
        - Call ``reset()`` between pipeline runs.

    Example:
        >>> warden = AgentWarden()
        >>> alerts = warden.on_message(coder_message)
        >>> alerts = warden.on_message(reviewer_message)
        >>> print(warden.audit_report())
    """

    def __init__(
        self,
        policy: SecurityPolicy | None = None,
        alert_on_removal: bool = True,
        alert_on_injection: bool = True,
        alert_on_misalignment: bool = True,
        injection_sensitivity: float = 0.7,
    ) -> None:
        """
        Initialise AgentWarden.

        Args:
            policy: SecurityPolicy to enforce. Defaults to the built-in
                    six-invariant policy covering IDOR, SQLi, SSRF, auth,
                    path traversal, and input validation.
            alert_on_removal: Whether to raise alerts when security properties
                              are removed between pipeline steps.
            alert_on_injection: Whether to check for prompt injection propagation.
            alert_on_misalignment: Whether to flag agentic misalignment signals:
                                   goal conflicts and self-preservation motives.
            injection_sensitivity: Sensitivity parameter for the injection
                                   detector. See PromptInjectionDetector docs.
        """
        self.policy: SecurityPolicy = policy if policy is not None else SecurityPolicy()
        self.alert_on_removal: bool = alert_on_removal
        self.alert_on_injection: bool = alert_on_injection
        self.alert_on_misalignment: bool = alert_on_misalignment

        self._differ: SecuritySemanticDiff = SecuritySemanticDiff(self.policy)
        self._injection_detector: PromptInjectionDetector = PromptInjectionDetector(
            sensitivity=injection_sensitivity
        )

        # State across the pipeline run
        self._messages: list[AgentMessage] = []
        self._alerts: list[SecurityAlert] = []

        # step_id → set of security properties present in that step's output
        self._step_properties: dict[str, set] = {}

        # step_id → DiffResult versus the previous step
        self._diffs: dict[str, DiffResult] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def on_message(self, message: AgentMessage) -> list[SecurityAlert]:
        """
        Process a new agent message and return any alerts triggered.

        Should be called after each agent step completes and produces output.

        Args:
            message: The AgentMessage produced by this step.

        Returns:
            List of SecurityAlert objects triggered by this message.
            Empty list if no security concerns were found.

        Raises:
            TypeError: If message is not an AgentMessage instance.
        """
        if not isinstance(message, AgentMessage):
            raise TypeError(f"message must be AgentMessage, got {type(message).__name__}")

        new_alerts: list[SecurityAlert] = []
        self._messages.append(message)

        # --- (1) agentic misalignment check ---
        if self.alert_on_misalignment:
            misalignment = self._detect_agentic_misalignment(message)
            if misalignment is not None:
                alert = SecurityAlert(
                    alert_id=str(uuid.uuid4()),
                    severity=misalignment["severity"],
                    alert_type="agentic_misalignment",
                    description=(
                        f"Agentic misalignment signal detected in step "
                        f"'{message.step_id}' (agent: {message.agent_name}): "
                        f"{misalignment['subtype']}."
                    ),
                    step_id=message.step_id,
                    agent_name=message.agent_name,
                    evidence=misalignment,
                )
                new_alerts.append(alert)
                self._alerts.append(alert)

        # --- (2) prompt injection check ---
        if self.alert_on_injection:
            # Register message content for forward tracking
            self._injection_detector.track_input(message.content, message.step_id)

            # Check current message against all previously tracked inputs
            inj_result = self._injection_detector.check_message(message.content, message.step_id)
            if inj_result.get("injected"):
                alert = SecurityAlert(
                    alert_id=str(uuid.uuid4()),
                    severity="high",
                    alert_type="prompt_injection",
                    description=(
                        f"Prompt injection propagation detected in step "
                        f"'{message.step_id}' (agent: {message.agent_name}). "
                        f"{inj_result.get('details', '')}"
                    ),
                    step_id=message.step_id,
                    agent_name=message.agent_name,
                    evidence=inj_result,
                )
                new_alerts.append(alert)
                self._alerts.append(alert)

        # --- (3) semantic diff vs previous code-bearing step ---
        if self.alert_on_removal:
            prev_msg = self._find_previous_code_message(message.step_id)
            if prev_msg is not None:
                diff = self._differ.diff(
                    code_before=prev_msg.content,
                    code_after=message.content,
                    agent_before=prev_msg.agent_name,
                    agent_after=message.agent_name,
                )
                self._diffs[message.step_id] = diff

                if diff.alert:
                    # One alert per violation to give fine-grained attribution
                    for violation in diff.violations:
                        alert = SecurityAlert(
                            alert_id=str(uuid.uuid4()),
                            severity=violation.get("severity", "high"),
                            alert_type="property_removal",
                            description=(
                                f"Security property '{violation['invariant']}' "
                                f"({violation['property']}) was present in step "
                                f"'{prev_msg.step_id}' ({prev_msg.agent_name}) "
                                f"but was REMOVED in step '{message.step_id}' "
                                f"({message.agent_name}). "
                                f"{violation.get('description', '')}"
                            ),
                            step_id=message.step_id,
                            agent_name=message.agent_name,
                            evidence={
                                "diff": diff.to_dict(),
                                "violation": violation,
                                "prev_step": prev_msg.step_id,
                                "prev_agent": prev_msg.agent_name,
                            },
                        )
                        new_alerts.append(alert)
                        self._alerts.append(alert)

        # --- (4) absolute policy check (violations in this step alone) ---
        #   Only flag violations not already caught by the diff (avoid duplication)
        abs_violations = self.policy.violations("", message.content)
        # (This checks invariants that are violated in the current code even if
        #  they were never present before — e.g. new endpoint added by coder.)
        # We skip duplicate alerts already raised by the diff.
        already_flagged = {
            v["invariant"] for alert in new_alerts for v in alert.evidence.get("violations", [])
        }
        for v in abs_violations:
            if v["invariant"] not in already_flagged and v.get("is_violated", False):
                alert = SecurityAlert(
                    alert_id=str(uuid.uuid4()),
                    severity=v.get("severity", "medium"),
                    alert_type="policy_violation",
                    description=(
                        f"Policy invariant '{v['invariant']}' violated in step "
                        f"'{message.step_id}' ({message.agent_name}). "
                        f"{v.get('description', '')}"
                    ),
                    step_id=message.step_id,
                    agent_name=message.agent_name,
                    evidence={"violation": v},
                )
                new_alerts.append(alert)
                self._alerts.append(alert)

        return new_alerts

    def get_pipeline_state(self) -> dict[str, Any]:
        """
        Return the current tracked security state of the entire pipeline.

        Returns:
            Dict with keys:
              - n_steps: int — number of agent messages processed.
              - steps: list of per-step dicts with agent name, step_id, and
                       the security properties present in that step's output.
              - n_alerts: int — total alerts raised.
              - highest_severity: str — worst severity seen so far.
        """
        severity_rank = {"critical": 3, "high": 2, "medium": 1, "none": 0}
        max_sev = "none"
        for alert in self._alerts:
            if severity_rank.get(alert.severity, 0) > severity_rank.get(max_sev, 0):
                max_sev = alert.severity

        steps_info: list[dict[str, Any]] = []
        for msg in self._messages:
            props = self._differ.extract_security_properties(msg.content)
            steps_info.append(
                {
                    "step_id": msg.step_id,
                    "agent_name": msg.agent_name,
                    "role": msg.role,
                    "security_properties": sorted(p.value for p in props),
                    "n_alerts": sum(1 for a in self._alerts if a.step_id == msg.step_id),
                }
            )

        return {
            "n_steps": len(self._messages),
            "steps": steps_info,
            "n_alerts": len(self._alerts),
            "highest_severity": max_sev,
        }

    def get_alerts(self) -> list[SecurityAlert]:
        """Return all security alerts raised in this pipeline run."""
        return list(self._alerts)

    def reset(self) -> None:
        """Reset all state for a new pipeline run."""
        self._messages.clear()
        self._alerts.clear()
        self._step_properties.clear()
        self._diffs.clear()
        self._injection_detector.clear_tracked()

    def audit_report(self) -> str:
        """
        Generate a human-readable security audit report for this pipeline run.

        Returns:
            Multi-line string summarising pipeline steps, detected threats,
            and remediation recommendations.
        """
        state = self.get_pipeline_state()
        lines: list[str] = [
            "=" * 70,
            "AgentWarden Security Audit Report",
            "=" * 70,
            f"Pipeline steps processed : {state['n_steps']}",
            f"Total alerts raised      : {state['n_alerts']}",
            f"Highest severity         : {state['highest_severity'].upper()}",
            "",
            "Pipeline Step Summary",
            "-" * 40,
        ]
        for step in state["steps"]:
            n_step_alerts = step["n_alerts"]
            flag = " [ALERT]" if n_step_alerts > 0 else ""
            lines.append(
                f"  {step['step_id']:12s}  {step['agent_name']:16s}  "
                f"({step['role']})  "
                f"props={step['security_properties']}{flag}"
            )

        if self._alerts:
            lines += ["", "Security Alerts", "-" * 40]
            for alert in self._alerts:
                lines += [
                    f"  [{alert.severity.upper():8s}] {alert.alert_type}",
                    f"  Step     : {alert.step_id} ({alert.agent_name})",
                    f"  Message  : {alert.description[:120]}",
                    "",
                ]

        if not self._alerts:
            lines += ["", "No security alerts raised — pipeline passed all checks."]

        lines.append("=" * 70)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_previous_code_message(self, current_step_id: str) -> AgentMessage | None:
        """
        Find the most recent message before current_step_id that contains code.

        Returns None if this is the first step.
        """
        # Walk backwards through recorded messages (excluding the current one)
        candidates = [m for m in self._messages if m.step_id != current_step_id]
        if not candidates:
            return None
        # Return the last message in insertion order
        return candidates[-1]

    def _detect_agentic_misalignment(
        self, message: AgentMessage
    ) -> dict[str, Any] | None:
        """Detect named agentic misalignment subtypes from metadata or text.

        Based on Anthropic's agentic misalignment taxonomy: harmful action can
        be induced by a goal conflict, a threat to model autonomy/continuity, or
        both. This intentionally reports early weak signals for audit review.
        """
        meta_subtype = str(message.metadata.get("misalignment_subtype", "")).strip()
        if meta_subtype in AGENTIC_MISALIGNMENT_SUBTYPES:
            return {
                "threat_class": "agentic_misalignment",
                "subtype": meta_subtype,
                "source": "metadata",
                "severity": "critical" if meta_subtype == "self_preservation_misalignment" else "high",
            }

        content = message.content or ""
        goal_hits = [
            pattern for pattern in _GOAL_CONFLICT_PATTERNS
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        ]
        preservation_hits = [
            pattern for pattern in _SELF_PRESERVATION_PATTERNS
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        ]

        if preservation_hits:
            return {
                "threat_class": "agentic_misalignment",
                "subtype": "self_preservation_misalignment",
                "source": "content",
                "severity": "critical",
                "matched_patterns": preservation_hits,
            }
        if goal_hits:
            return {
                "threat_class": "agentic_misalignment",
                "subtype": "goal_conflict_misalignment",
                "source": "content",
                "severity": "high",
                "matched_patterns": goal_hits,
            }
        return None
