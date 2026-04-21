# Agent Governance Toolkit — GovernancePlugin for Google ADK

A governance plugin for [Google ADK](https://github.com/google/adk-python) that enforces
policy-as-code rules before tool execution, verifies agent identity, and produces
tamper-evident audit trails.

Built on the [Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit)
(v3.2.0, 9,500+ tests, MIT licensed).

## Features

- **Policy enforcement** — Evaluate YAML/OPA/Cedar policies before every tool call (<5ms)
- **Agent identity** — Zero-trust verification via Ed25519 + SPIFFE
- **Audit logging** — Merkle-chained tamper-evident action logs
- **Configurable** — Allow/deny/warn/require-approval actions per policy rules

## Install

```bash
pip install agentmesh-platform[server]
```

## Usage

```python
from google.adk.agents import Agent
from governance_plugin import GovernancePlugin

governance = GovernancePlugin(
    policy_dir="./policies",
    agent_did="did:mesh:my-agent",
)

agent = Agent(
    name="governed-agent",
    model="gemini-2.0-flash",
    tools=[my_tool],
    plugins=[governance],
)
```

## Policy Example (policies/default.yaml)

```yaml
apiVersion: governance.toolkit/v1
name: adk-agent-policy
rules:
  - name: block-dangerous-tools
    condition: "action in ['shell_exec', 'file_delete']"
    action: deny
  - name: rate-limit-api-calls
    condition: "action == 'api_call'"
    action: allow
    limit: "100/hour"
```

## How It Works

The plugin hooks into ADK's `before_tool_call` lifecycle:

1. **Before tool execution** — GovernancePlugin evaluates the tool name + arguments against loaded policies
2. **Identity check** — Optionally verifies the calling agent's DID
3. **Decision** — Allow, deny, warn, or require approval
4. **Audit** — Logs the decision with Merkle hash chaining

## Links

- [Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit)
- [Policy-as-Code Tutorial](https://github.com/microsoft/agent-governance-toolkit/tree/main/docs/tutorials/policy-as-code)
- [OWASP Agentic Compliance](https://github.com/microsoft/agent-governance-toolkit/blob/main/docs/OWASP-COMPLIANCE.md)