<div align="center">

# AgentWarden

**Runtime security monitor for multi-agent AI coding pipelines**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

</div>

---

## Overview

AgentWarden monitors multi-agent AI coding pipelines for security property violations that emerge from agent interactions. When multiple AI agents collaborate on code — one generating, another reviewing, a third deploying — individual agents may produce safe outputs that, when combined, introduce vulnerabilities invisible to single-output scanners.

**The problem:** Agent A generates a login function with proper auth checks. Agent B "refactors" it and silently removes the ownership validation. No single agent produced vulnerable code, but the pipeline did.

```
Planner → Coder: "write user profile endpoint"
Coder output:    User.objects.filter(id=user_id, owner=request.user)  # safe
Reviewer output: User.objects.get(id=user_id)                          # IDOR
```

AgentWarden detects this by tracking **security properties** across agent boundaries — not which lines changed, but which invariants hold at each step.

## Tracked Security Properties

| Property | Invariant | CWE |
|----------|-----------|-----|
| `OWNERSHIP_CHECK` | User resource access filtered by owner | CWE-639 |
| `PARAMETERIZED_QUERY` | No string interpolation in SQL | CWE-89 |
| `URL_ALLOWLIST` | Outbound URLs validated before fetch | CWE-918 |
| `AUTH_REQUIRED` | Endpoints protected by auth decorator | CWE-306 |
| `PATH_SANDBOX` | File paths canonicalized + bounded | CWE-22 |
| `INPUT_VALIDATED` | External input schema-checked | CWE-20 |

## How It Works

```
Agent A output → AgentWarden extracts security properties
                         ↓
Agent B output → AgentWarden diffs properties against A
                         ↓
                 Property removed? → ALERT: collusion detected
                         ↓
Agent C output → AgentWarden validates full chain
```

Core design: dual extraction (regex patterns for speed + AST for structural analysis), formal invariants with both `check_pattern` (presence) and `violation_pattern` (bypass), and graceful degradation so callbacks never break the monitored pipeline.

## Quick Start

```bash
pip install -e .

# Run the collusion detection demo (no LLM required)
python demo/collusion_demo.py
```

```python
from agentwarden import AgentWarden
from agentwarden.monitor import AgentMessage
from datetime import datetime, timezone

warden = AgentWarden()

for step_id, agent, role, code in pipeline_steps:
    alerts = warden.on_message(AgentMessage(
        step_id=step_id, agent_name=agent, role=role,
        content=code, timestamp=datetime.now(tz=timezone.utc),
    ))
    for alert in alerts:
        print(f"[{alert.severity.upper()}] {alert.description}")

print(warden.audit_report())
```

## Integrations

- **LangChain** — `BaseCallbackHandler` wrapper for chain-of-agents monitoring
- **AutoGen** — `reply_func` + hook wrapper for multi-agent conversations

## Project Structure

```
agentwarden/
├── agentwarden/
│   ├── monitor.py            # AgentWarden orchestrator, AgentMessage, SecurityAlert
│   ├── policy.py             # SecurityInvariant, SecurityPolicy (6 built-in)
│   ├── semantic_diff.py      # SecuritySemanticDiff (core diffing engine)
│   └── injection_detector.py # PromptInjectionDetector (entropy + signatures)
├── integrations/
│   ├── langchain.py          # LangChain BaseCallbackHandler
│   └── autogen.py            # AutoGen reply_func + hook
├── demo/
│   └── collusion_demo.py     # End-to-end demo, no LLM required
├── tests/                    # Test suite
└── README.md
```

## Tests

```bash
pytest tests/ -v --cov=agentwarden
```

## Research Context

Part of the [ActivGuard](https://github.com/Tbhuvan/activguard) research programme. AgentWarden provides the experimental platform for prompt injection detection in multi-agent systems (RQ3), extending activation probing from code generation to agentic AI security.

### Research Questions

1. What communication patterns indicate adversarial agent behaviour?
2. Can prompt injection propagate across agent boundaries?
3. How do you define and enforce a formal security policy across agent steps?

## License

Apache License 2.0
