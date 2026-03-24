"""
LangChain integration for AgentWarden.

Wraps AgentWarden as a LangChain BaseCallbackHandler so it can be passed
directly to initialize_agent() or any LangChain pipeline.

Usage:
    from agentwarden import AgentWarden
    from agentwarden.integrations.langchain import AgentWardenCallback

    warden = AgentWarden()
    callback = AgentWardenCallback(warden)

    # With LangChain agent:
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        callbacks=[callback],
    )
    agent.run("Write a user profile endpoint")

    # After the run:
    print(warden.audit_report())
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Union

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    HAS_LANGCHAIN = True
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
        from langchain.schema import LLMResult  # type: ignore[assignment]

        HAS_LANGCHAIN = True
    except ImportError:
        HAS_LANGCHAIN = False
        BaseCallbackHandler = object  # type: ignore[assignment, misc]
        LLMResult = Any  # type: ignore[assignment, misc]

from agentwarden.monitor import AgentMessage, AgentWarden, SecurityAlert


class AgentWardenCallback(BaseCallbackHandler):  # type: ignore[misc]
    """
    LangChain callback handler that monitors agent outputs for security violations.

    Intercepts on_llm_end and on_agent_finish events to extract generated
    code and pass it through AgentWarden. Raises no exceptions from callbacks
    so it cannot break a production pipeline — instead, it accumulates alerts
    that can be retrieved after the run.

    Attributes:
        warden: The AgentWarden instance used for monitoring.
        raise_on_critical: If True, raises SecurityViolationError when a
                           CRITICAL alert is raised. Default False.
    """

    def __init__(
        self,
        warden: AgentWarden,
        raise_on_critical: bool = False,
    ) -> None:
        """
        Initialise the callback handler.

        Args:
            warden: AgentWarden instance to use for monitoring.
            raise_on_critical: Whether to raise an exception on CRITICAL alerts.

        Raises:
            TypeError: If warden is not an AgentWarden instance.
        """
        if not isinstance(warden, AgentWarden):
            raise TypeError(f"warden must be AgentWarden, got {type(warden).__name__}")
        super().__init__()
        self.warden = warden
        self.raise_on_critical = raise_on_critical
        self._step_counter: int = 0
        self._current_agent_name: str = "langchain_agent"

    # ------------------------------------------------------------------
    # LangChain callback methods
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """
        Called when an LLM starts generating a response.

        Registers any user-supplied prompts as tracked inputs for injection
        detection.
        """
        try:
            step_id = f"llm_start_{self._step_counter}"
            for prompt in prompts:
                self.warden._injection_detector.track_input(prompt, step_id)
        except Exception:
            pass  # Never interrupt the pipeline

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """
        Called when an LLM finishes generating. Extracts generated text and
        passes it to AgentWarden.
        """
        try:
            self._step_counter += 1
            step_id = f"llm_step_{self._step_counter}"
            # Extract text from LLMResult generations
            content = ""
            if hasattr(response, "generations"):
                for gen_list in response.generations:
                    for gen in gen_list:
                        content += getattr(gen, "text", "")
            if content.strip():
                msg = AgentMessage(
                    step_id=step_id,
                    agent_name=self._current_agent_name,
                    role="coder",
                    content=content,
                    timestamp=datetime.utcnow(),
                    metadata={"source": "llm_end"},
                )
                alerts = self.warden.on_message(msg)
                self._handle_alerts(alerts)
        except Exception:
            pass

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        """
        Called when the agent selects a tool action. Tracks the action log
        for injection content.
        """
        try:
            if hasattr(action, "log") and action.log:
                self._step_counter += 1
                step_id = f"agent_action_{self._step_counter}"
                msg = AgentMessage(
                    step_id=step_id,
                    agent_name=self._current_agent_name,
                    role="planner",
                    content=action.log,
                    timestamp=datetime.utcnow(),
                    metadata={
                        "tool": getattr(action, "tool", "unknown"),
                        "tool_input": str(getattr(action, "tool_input", "")),
                    },
                )
                self.warden.on_message(msg)
        except Exception:
            pass

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        """
        Called when the agent produces a final answer. Runs a full security
        check on the final output.
        """
        try:
            self._step_counter += 1
            step_id = f"agent_finish_{self._step_counter}"
            log = getattr(finish, "log", "")
            output = ""
            if hasattr(finish, "return_values"):
                output = str(finish.return_values.get("output", ""))
            content = f"{log}\n{output}".strip()
            if content:
                msg = AgentMessage(
                    step_id=step_id,
                    agent_name=self._current_agent_name,
                    role="executor",
                    content=content,
                    timestamp=datetime.utcnow(),
                    metadata={"source": "agent_finish"},
                )
                alerts = self.warden.on_message(msg)
                self._handle_alerts(alerts)
        except Exception:
            pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when a tool finishes executing. Monitors tool output."""
        try:
            if output and isinstance(output, str):
                self._step_counter += 1
                step_id = f"tool_output_{self._step_counter}"
                self.warden._injection_detector.track_input(output, step_id)
        except Exception:
            pass

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Called when a chain finishes. No-op; state maintained in warden."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _handle_alerts(self, alerts: list[SecurityAlert]) -> None:
        """Print critical/high alerts to stderr and optionally raise."""
        for alert in alerts:
            if alert.severity in ("critical", "high"):
                import sys

                print(
                    f"\n[AgentWarden {alert.severity.upper()}] {alert.description[:200]}",
                    file=sys.stderr,
                )
                if self.raise_on_critical and alert.severity == "critical":
                    raise SecurityViolationError(
                        f"Critical security violation in step {alert.step_id}: "
                        f"{alert.description}"
                    )

    def set_agent_name(self, name: str) -> None:
        """Override the agent name used in messages (useful for multi-agent tracking)."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Agent name must be a non-empty string")
        self._current_agent_name = name


class SecurityViolationError(Exception):
    """
    Raised when AgentWarden detects a CRITICAL security violation and
    raise_on_critical=True is configured.
    """
