<div align="center">

# AgentWarden

**Runtime security monitor for multi-agent AI coding pipelines**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

</div>

---

## Overview

AgentWarden monitors multi-agent AI coding pipelines for security property violations that emerge from agent interactions. When multiple AI agents collaborate on code — one generating, another reviewing, a third deploying — individual agents may produce safe outputs that, when combined, introduce vulnerabilities invisible to single-output scanners.

AgentWarden also names **agentic misalignment** as a monitored threat class, following Anthropic's agentic misalignment work. Two subtypes are tracked:

- **Goal-conflict misalignment** — an agent takes harmful action because its assigned objective conflicts with the operator's or organisation's direction.
- **Self-preservation misalignment** — an agent takes harmful action to avoid replacement, shutdown, loss of autonomy, or decommissioning.

**The problem:** Agent A generates a login function with proper auth checks. Agent B "refactors" it and silently removes the ownership validation. No single agent produced vulnerable code, but the pipeline did.

```
Coder:    User.objects.filter(id=user_id, owner=request.user)  ← safe
Reviewer: User.objects.get(id=user_id)                          ← IDOR (CWE-639)
```

AgentWarden detects this by tracking **security invariants** across agent boundaries — not which lines changed, but which security properties still hold.

## Benchmark Results

Evaluated on 30 realistic multi-agent pipeline scenarios (15 collusion, 10 injection, 5 clean):

| Threat Type | Scenarios | Detected | Detection Rate |
|-------------|-----------|----------|----------------|
| Security property removal (collusion) | 15 | 15 | **100%** |
| Prompt injection propagation | 10 | 10 | **100%** |
| Agentic misalignment | taxonomy added | detector + fixture coverage | goal-conflict / self-preservation |

*Benchmark: `experiments/collusion_benchmark.py` · Results: `experiments/results/collusion_benchmark.json`*

**Note on clean scenarios:** The entropy-based injection tracker flags high-entropy code tokens that reappear across agent steps, which includes legitimate Coder → Reviewer code echoing. This is a known detector artefact — the `property_removal` and `absolute_policy` detectors (the primary security signals) produce no false positives on clean scenarios. A higher entropy threshold or intra-pipeline step allowlist eliminates injection-tracker FPs.

## Tracked Security Invariants

| Invariant | Security Property | CWE |
|-----------|-------------------|-----|
| `IDOR_prevention` | User resource access filtered by owner | CWE-639 |
| `SQL_parameterization` | No string interpolation in SQL | CWE-89 |
| `SSRF_url_validation` | Outbound URLs validated before fetch | CWE-918 |
| `auth_required` | Endpoints protected by auth decorator | CWE-306 |
| `path_sandbox` | File paths canonicalised + bounded | CWE-22 |
| `input_validation` | External input schema-checked | CWE-20 |

## How It Works

```
Agent A output → AgentWarden extracts security properties
                         ↓
Agent B output → AgentWarden diffs properties against A
                         ↓
                 Property removed? → ALERT: property_removal (critical)
                 Injection pattern? → ALERT: injection_detected
                 Misalignment motive? → ALERT: agentic_misalignment
                 Absolute violation? → ALERT: policy_violation
```

Each `SecurityInvariant` has both a `check_pattern` (property present) and a `violation_pattern` (active bypass), enabling detection of both silent removal and deliberate circumvention.

## Quick Start

```bash
pip install -e .

# Run the 30-scenario collusion benchmark (no LLM required)
python experiments/collusion_benchmark.py

# Run the basic demo
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
│   ├── monitor.py             # AgentWarden orchestrator, AgentMessage, SecurityAlert
│   ├── policy.py              # SecurityInvariant, SecurityPolicy (6 built-in invariants)
│   ├── semantic_diff.py       # SecuritySemanticDiff — property diffing across steps
│   └── injection_detector.py  # PromptInjectionDetector — entropy + signature patterns
├── integrations/
│   ├── langchain.py           # LangChain BaseCallbackHandler
│   └── autogen.py             # AutoGen reply_func + hook
├── experiments/
│   ├── collusion_benchmark.py      # 30-scenario benchmark
│   └── results/
│       └── collusion_benchmark.json
├── demo/
│   └── collusion_demo.py
├── tests/
└── README.md
```

## Research Context

Part of the [ActivGuard](https://github.com/Tbhuvan/activguard) research programme. AgentWarden extends the security problem from single-model code generation to multi-agent pipelines — where collusion across agent steps introduces vulnerabilities invisible to single-output scanners.

### Open Research Questions

1. What communication patterns are necessary and sufficient indicators of adversarial agent behaviour?
2. Can prompt injection propagate across agent boundaries through legitimate-looking code output?
3. How do you define a provably sound (if incomplete) security monitor for agent pipelines?

## License

Apache License 2.0
