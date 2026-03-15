# Governance Plugin Example

Budget enforcement, circuit breaking, and model degradation for ADK agents.

## Quickstart

```python
from google.adk_community.governance import GovernanceConfig, VeronicaGovernancePlugin

plugin = VeronicaGovernancePlugin(GovernanceConfig(max_cost_usd=1.0))

runner = Runner(agent=agent, session_service=session_service, plugins=[plugin])
```

That's it. The plugin intercepts model and tool callbacks automatically.

## Setup

```bash
pip install google-adk-community
export GOOGLE_API_KEY="your-key"
```

## Run the full example

```bash
python main.py
```

## What it does

The example creates three agents (orchestrator, researcher, summarizer) and
registers a `VeronicaGovernancePlugin` on the Runner. The plugin:

- Enforces a $0.50 org budget and $0.25 per-agent budget
- Blocks the `shell_exec` tool
- Degrades to `gemini-2.0-flash-lite` when budget hits 70%
- Disables `web_search` during degradation
- Trips the circuit breaker after 3 consecutive failures

After the run completes, the plugin logs a summary:

```
[GOVERNANCE] Run complete in 2.3s. Model calls: 4, Tool calls: 0.
[GOVERNANCE] Budget: $0.0023 / $0.5000 (0.5% used).
[GOVERNANCE]   Agent 'researcher': $0.0012 / $0.2500.
[GOVERNANCE]   Agent 'summarizer': $0.0008 / $0.2500.
```

If degradation triggers, the summary includes:

```
[GOVERNANCE] Degradation events (1):
[GOVERNANCE]   Agent 'researcher' at 72.0% -- degraded gemini-2.5-flash -> gemini-2.0-flash-lite.
```
