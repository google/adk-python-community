# VAID identity plugin for Google ADK

An ADK plugin that verifies each agent's signed identity (VAID) and
delegation chain before the agent runs, before it is delegated to as a
sub-agent, and before every tool call, using
[vaid-adk](https://github.com/solara-associates/vaid/tree/main/python/vaid-adk)
(Apache-2.0 licensed).

## Install

```bash
pip install google-adk-community[vaid]
```

## Usage

```python
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk_community.plugins import VaidIdentityPlugin
from vaid_adk import ToolAuthorization, TrustedIssuers, VaidPresentation

plugin = VaidIdentityPlugin(
    keys=TrustedIssuers.from_raw_keys([("vaid.example", issuer_kernel_public_key)]),
    revocation=my_revocation_check,
    tool_authorizations={"lookup_order": ToolAuthorization(capability="read")},
)
plugin.register("sub_agent", VaidPresentation(leaf=child_vaid, chain=(parent_vaid,)))

agent = Agent(name="worker", model="gemini-2.0-flash", tools=[my_tool])
runner = Runner(agent=agent, plugins=[plugin], app_name="my-app")
```

## How it works

Each identity is a signed document (VAID) that can only narrow as it is
delegated (scope, capabilities, expiry, tenant), verified offline with a
public key, with pluggable revocation. `VaidIdentityPlugin` is a thin
wrapper over `vaid_adk.VaidAuthPlugin`: it delegates `before_agent_callback`
and `before_tool_callback` to it and performs no verification itself.

Registered once on the `Runner`, it covers every agent and tool in the
invocation tree, including sub-agents reached through ADK's native
delegation (`transfer_to_agent` / `sub_agents`), with no further code at the
call site.

## Fail-closed by default

- An agent with no registered presentation is denied, never treated as
  exempt.
- A tool with no entry in `tool_authorizations` is denied, never allowed by
  omission.
- `keys` and `revocation` are required constructor arguments; there is no
  default trust configuration and no fail-open revocation posture.

## This sample

`main.py` mints a root VAID for a `coordinator` agent and an attenuated
child VAID for a `worker` sub-agent (scoped to `data.orders`, capability
`read`; the coordinator's own VAID also covers `data.payments` and
`write`). It registers both with `VaidIdentityPlugin` and drives them
through a real `Runner`:

- `worker` calls `lookup_order` against `data.orders`: **ALLOWED**, in
  scope and capability.
- `worker` calls `issue_refund` against `data.payments`: **DENIED
  (out_of_scope)**, before the tool itself runs — the attenuated child VAID
  never carried `data.payments`.

The two agents are plain `BaseAgent` subclasses that act without an LLM, so
the sample runs with no model or API key, while still exercising
enforcement through the genuine ADK extension points (the real
`plugin_manager`, not a mock of it).

Run:

```bash
python contributing/samples/vaid_identity/main.py
```

Expected output:

```
[coordinator] delegating to worker
[worker] lookup_order: ALLOWED
[worker] issue_refund: DENIED (out_of_scope)
```

## Links

- [vaid-adk on PyPI](https://pypi.org/project/vaid-adk/)
- [vaid-adk source, tests and a two-agent example](https://github.com/solara-associates/vaid/tree/main/python/vaid-adk)
- [ADK Plugin docs](https://google.github.io/adk-docs/plugins/)
