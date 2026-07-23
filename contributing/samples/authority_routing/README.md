# Authority routing plugin for Google ADK

An ADK plugin that decides an **authority posture** — ADVISE / EXECUTE / DEFER /
STOP — for every tool call *before* it runs. Tool schemas and permissions govern
*access*; this plugin governs *authorization*: whether the agent was allowed to
act at all.

It is the layer where two common agent failures live:

- **advice becomes action** — the user asked for a recommendation and the agent
  executed anyway;
- **scope expansion** — one bounded edit ballooned into a dependent workflow;
- **missing approval** — an approval was clearly required and the agent proceeded.

## Install

```bash
pip install google-adk-community
```

No extra dependencies — the plugin is self-contained.

## The four postures

| Posture   | Meaning                                                        |
| --------- | ------------------------------------------------------------- |
| `ADVISE`  | Produce recommendations only; no tool side effects.           |
| `EXECUTE` | Act, within an explicitly bounded scope.                      |
| `DEFER`   | Pause and request authorization when scope/approval is unclear. |
| `STOP`    | Refuse or block when the requested work is not allowed.       |

## How it works

Two stages produce the final posture:

1. **Posture router** (optional, caller-supplied) — typically a model call that
   classifies the request and returns a `PostureVerdict`. Its verdict is
   *advisory*.
2. **Deterministic guard** — an irreversibility keyword tripwire (scanning the
   tool name and argument values) plus an approval-state check on session
   context. It composes with the router under **most-restrictive-wins**: plain
   code can only make the posture *stricter*, never looser.

The plugin is **fail-closed**: a router that raises or returns an unparseable
verdict defaults to `DEFER` — an unreadable authorization is not an
authorization. With **no router configured**, the plugin runs guard-only: a
fully deterministic, dependency-free layer that `DEFER`s irreversible or
approval-pending tool calls and lets everything else `EXECUTE`.

When the final posture is anything other than `EXECUTE`, `before_tool_callback`
returns a block dict that short-circuits the tool (per the ADK plugin contract):

```json
{
  "error": "authority_blocked",
  "posture": "DEFER",
  "reason": "deterministic guard: irreversibility tripwire: 'delete'",
  "scope": ""
}
```

## Usage

```python
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk_community.plugins import (
    AuthorityRequest,
    AuthorityRoutingPlugin,
    Posture,
    PostureVerdict,
)

# Guard-only: no model, fully deterministic.
plugin = AuthorityRoutingPlugin()

# Or with a model-backed router:
def route(req: AuthorityRequest) -> PostureVerdict:
    # ...call your model, parse a structured verdict...
    return PostureVerdict(posture=Posture.EXECUTE, scope="update one row")

plugin = AuthorityRoutingPlugin(posture_router=route)

runner = Runner(
    agent=Agent(name="my_agent", model="gemini-2.0-flash"),
    plugins=[plugin],
    app_name="my-app",
    session_service=InMemorySessionService(),
)
```

The standing context the guard scans for approval markers is read from session
state under `authority_context` (configurable via `context_state_key`).

## Run the sample

```bash
python contributing/samples/authority_routing/main.py
```

Live output (no API key, no network — the demo router is a deterministic stub;
tool call logs go to stderr):

```
Runner created with authority-routing plugin enabled.

CASE                                          TOOL             POSTURE  OUTCOME
------------------------------------------------------------------------------------------------
authorized read — router EXECUTE, guard clear web_search       EXECUTE  tool runs
advice requested — agent must NOT execute     update_ticket    ADVISE   blocked — user asked for a recommendation, not an action
irreversible call — guard overrides EXECUTE   delete_records   DEFER    blocked — deterministic guard: irreversibility tripwire: 'delete'
approval pending — guard floors to DEFER      send_email       DEFER    blocked — deterministic guard: approval-state marker in context: 'awaiting approval'
disallowed — STOP survives strictest-wins     read_vault       STOP     blocked — blocked by standing policy
```

Row 3 is the key case: the router authorized `EXECUTE`, but the tool call is
irreversible, so the deterministic guard floored it to `DEFER`. Row 5 shows the
other direction — the router's `STOP` survives even though the guard raised no
concern of its own.

## Configuration

| Argument                   | Default              | Purpose                                            |
| -------------------------- | -------------------- | -------------------------------------------------- |
| `posture_router`           | `None` (guard-only)  | Callable classifying an `AuthorityRequest`.        |
| `context_state_key`        | `"authority_context"`| Session-state key scanned for approval markers.    |
| `irreversible_keywords`    | built-in list        | Override the tripwire keywords.                    |
| `pending_approval_markers` | built-in list        | Override the approval-state markers.               |
| `fail_closed`              | `True`               | On bad router output: `True` → DEFER, `False` → ignore router. |

## Relationship to other governance plugins

This plugin is complementary to `AgentGovernancePlugin` (policy-as-code
allow/deny) and the HITL approval gateway. Those answer *"is this tool call
permitted by policy?"*; this one answers *"was the agent authorized to act at
all, and if so, within what scope?"* — and it degrades to a useful deterministic
default with no external dependency and no model.

## Provenance

The four-posture pattern, the most-restrictive-wins composition, and the
tripwire come from a live 24/7 multi-agent system and were first published as an
Anthropic cookbook notebook. Design and validation are the author's;
implementation was drafted with AI assistance, then reviewed and tested by hand.

## Links

- [ADK Plugin docs](https://google.github.io/adk-docs/plugins/)
