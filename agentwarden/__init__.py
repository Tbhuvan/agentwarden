"""
agentwarden — Runtime security monitor for multi-agent AI coding pipelines.

Detects when agents collude to introduce vulnerabilities that no single-output
scanner can catch. Core capabilities:

  - Security semantic diff: tracks which invariants (ownership checks,
    parameterized queries, auth decorators, etc.) are removed between steps.
  - Prompt injection propagation: traces high-entropy user-supplied tokens
    through agent message chains and detects executable-position embedding.
  - Formal security policy: declarative invariant definitions with regex and
    AST-based matching.

Quick start:
    from agentwarden import AgentWarden
    from agentwarden.monitor import AgentMessage
    from datetime import datetime

    warden = AgentWarden()

    for step_id, agent_name, role, code in pipeline_steps:
        msg = AgentMessage(
            step_id=step_id,
            agent_name=agent_name,
            role=role,
            content=code,
            timestamp=datetime.utcnow(),
        )
        alerts = warden.on_message(msg)
        for alert in alerts:
            print(f"[{alert.severity.upper()}] {alert.description}")

    print(warden.audit_report())
"""

from .monitor import AgentMessage, AgentWarden, SecurityAlert
from .policy import INVARIANTS, SecurityInvariant, SecurityPolicy, SecurityProperty
from .semantic_diff import DiffResult, SecuritySemanticDiff

__all__ = [
    "AgentWarden",
    "AgentMessage",
    "SecurityAlert",
    "SecurityPolicy",
    "SecurityProperty",
    "SecurityInvariant",
    "INVARIANTS",
    "SecuritySemanticDiff",
    "DiffResult",
]

__version__ = "0.1.0"
__author__ = "Bhuvan Garg"
__description__ = "Runtime security monitor for multi-agent AI coding pipelines"
