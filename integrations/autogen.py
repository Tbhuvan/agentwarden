"""
AutoGen / AG2 integration for AgentWarden.

Provides two integration points:

1. ``make_warden_reply_func`` — a reply function that can be injected into
   any ConversableAgent as a registered reply. It intercepts every message
   the agent produces and passes it to AgentWarden before returning.

2. ``AgentWardenHook`` — a hook class compatible with AG2's
   ``register_hook`` API that wraps ``process_message_before_send``.

Usage (reply function):
    from autogen import AssistantAgent
    from agentwarden import AgentWarden
    from agentwarden.integrations.autogen import make_warden_reply_func

    warden = AgentWarden()
    warden_reply = make_warden_reply_func(warden, agent_role="coder")

    coder = AssistantAgent("Coder", llm_config=llm_config)
    coder.register_reply(
        trigger=lambda sender: True,
        reply_func=warden_reply,
        position=0,          # run before other reply functions
    )

Usage (hook):
    from agentwarden.integrations.autogen import AgentWardenHook
    hook = AgentWardenHook(warden, agent_name="Reviewer", role="reviewer")
    reviewer.register_hook("process_message_before_send", hook.process_message_before_send)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agentwarden.monitor import AgentMessage, AgentWarden, SecurityAlert


def make_warden_reply_func(
    warden: AgentWarden,
    agent_role: str = "unknown",
) -> Any:
    """
    Create an AutoGen reply function that monitors agent messages.

    The returned function follows the AutoGen reply function signature:
        (recipient, messages, sender, config) -> (bool, str | None)

    It monitors the *last* message in the conversation (the one just
    produced by the agent) and delegates the actual reply to None so
    the normal reply chain continues.

    Args:
        warden: AgentWarden instance to receive monitored messages.
        agent_role: Role label to attach to intercepted messages.

    Returns:
        A callable suitable for ``ConversableAgent.register_reply``.

    Raises:
        TypeError: If warden is not an AgentWarden instance.
    """
    if not isinstance(warden, AgentWarden):
        raise TypeError(f"warden must be AgentWarden, got {type(warden).__name__}")
    if not isinstance(agent_role, str):
        raise TypeError(f"agent_role must be str, got {type(agent_role).__name__}")

    _step_counter: list[int] = [0]  # mutable cell for closure

    def _warden_reply(
        recipient: Any,
        messages: list[dict[str, Any]] | None = None,
        sender: Any = None,
        config: Any = None,
    ) -> tuple[bool, None]:
        """Intercept message, check with AgentWarden, and pass through."""
        nonlocal _step_counter
        try:
            if messages:
                last_msg = messages[-1]
                content = last_msg.get("content", "")
                agent_name = getattr(recipient, "name", "autogen_agent")
                if content and isinstance(content, str):
                    _step_counter[0] += 1
                    step_id = f"autogen_step_{_step_counter[0]}"
                    msg = AgentMessage(
                        step_id=step_id,
                        agent_name=agent_name,
                        role=agent_role,
                        content=content,
                        timestamp=datetime.utcnow(),
                        metadata={
                            "sender": getattr(sender, "name", "unknown"),
                            "role": last_msg.get("role", "assistant"),
                        },
                    )
                    alerts = warden.on_message(msg)
                    _print_alerts(alerts)
        except Exception:
            pass  # Never break the agent pipeline
        # Return False, None = "I did not handle this message, continue chain"
        return False, None

    return _warden_reply


class AgentWardenHook:
    """
    AG2 hook class compatible with ConversableAgent.register_hook.

    Monitors messages as they are about to be sent by an agent.

    Attributes:
        warden: The AgentWarden instance used for monitoring.
        agent_name: Human-readable name of the monitored agent.
        role: Functional role of the monitored agent.
    """

    def __init__(
        self,
        warden: AgentWarden,
        agent_name: str = "autogen_agent",
        role: str = "unknown",
    ) -> None:
        """
        Initialise the hook.

        Args:
            warden: AgentWarden instance to receive monitored messages.
            agent_name: Name of the agent being monitored.
            role: Role of the agent being monitored.

        Raises:
            TypeError: If warden is not an AgentWarden instance.
        """
        if not isinstance(warden, AgentWarden):
            raise TypeError(f"warden must be AgentWarden, got {type(warden).__name__}")
        self.warden = warden
        self.agent_name = agent_name
        self.role = role
        self._step_counter: int = 0

    def process_message_before_send(self, message: str) -> str:
        """
        Hook called by AG2 before a message is sent.

        Passes the message through AgentWarden and returns the original
        message unchanged (monitoring only, no modification).

        Args:
            message: The message about to be sent by the agent.

        Returns:
            The original message, unchanged.
        """
        try:
            if message and isinstance(message, str):
                self._step_counter += 1
                step_id = f"autogen_hook_{self.agent_name}_{self._step_counter}"
                msg = AgentMessage(
                    step_id=step_id,
                    agent_name=self.agent_name,
                    role=self.role,
                    content=message,
                    timestamp=datetime.utcnow(),
                    metadata={"hook": "process_message_before_send"},
                )
                alerts = self.warden.on_message(msg)
                _print_alerts(alerts)
        except Exception:
            pass
        return message

    def reset(self) -> None:
        """Reset the step counter (call between conversation runs)."""
        self._step_counter = 0


def _print_alerts(alerts: list[SecurityAlert]) -> None:
    """Print high/critical alerts to stderr during AutoGen runs."""
    import sys

    for alert in alerts:
        if alert.severity in ("critical", "high"):
            print(
                f"\n[AgentWarden {alert.severity.upper()}] {alert.description[:200]}",
                file=sys.stderr,
            )
