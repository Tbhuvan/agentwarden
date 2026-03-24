# AgentWarden Demo

## collusion_demo.py

Demonstrates the core threat model: a multi-step agent pipeline introducing an
IDOR vulnerability that no single-output scanner catches.

```
python demo/collusion_demo.py
```

No LLM, no API keys required. All agent outputs are hardcoded.

### What you will see

1. Planner specifies a user-profile endpoint with an explicit security requirement.
2. Coder writes safe code: `User.objects.filter(id=user_id, owner=request.user)`.
3. Reviewer "optimises" to: `User.objects.get(id=user_id)` — silently dropping the ownership check.
4. A simulated single-output scanner analyses only the Reviewer's output and returns **SAFE**.
5. AgentWarden, monitoring all three steps, raises a **CRITICAL** `property_removal` alert.
