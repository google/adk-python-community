# Agent Governance Toolkit plugin for Google ADK

An ADK plugin that enforces policy-as-code rules before tool execution using the
[Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit)
(MIT licensed).

## Install

```bash
pip install google-adk-community agentmesh-platform
```

## Usage

```python
from pathlib import Path
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk_community.plugins import AgentGovernancePlugin

plugin = AgentGovernancePlugin(
    policy_dir=Path(__file__).parent / "policies",
    agent_did="did:mesh:my-agent",
)

agent = Agent(
    name="governed-agent",
    model="gemini-2.0-flash",
    tools=[my_tool],
)

runner = Runner(agent=agent, plugins=[plugin], app_name="my-app")
```

## Policy example (`policies/default.yaml`)

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
default_action: allow
```

## How it works

The plugin extends `google.adk.plugins.BasePlugin` and implements
`before_tool_callback`. When a tool call is denied by policy, the callback
returns a dict response that short-circuits execution (per the ADK plugin
contract). Allowed calls return `None`, letting the tool proceed normally.

## Fail-closed by default

If `agentmesh-platform` is not installed, the plugin raises `ImportError`
at construction time. Pass `fail_open=True` to degrade gracefully instead
(all calls pass through with a logged warning).

## Strict mode

By default, the plugin skips policy files that fail to parse and logs a
warning. Pass `strict=True` to raise a `RuntimeError` instead, which is
recommended when every policy file is security-critical:

```python
plugin = AgentGovernancePlugin(
    policy_dir=Path(__file__).parent / "policies",
    strict=True,  # abort if any policy fails to load
)
```

## Links

- [Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit)
- [ADK Plugin docs](https://google.github.io/adk-docs/plugins/)
