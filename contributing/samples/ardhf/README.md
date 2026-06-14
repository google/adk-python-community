# ARDHF — Agent Finder (ARD) Toolset for ADK

## Overview

ARDHF wraps [HuggingFace's Agent Finder](https://github.com/huggingface/hf-agentfinder)
(ARD — Agentic Resource Discovery) as an ADK `BaseToolset`.  It gives any ADK
agent the ability to **search for and discover** agents, skills, MCP servers,
and other agentic resources at runtime.

The toolset provides two tools:

| Tool | Description |
|---|---|
| `search_agents` | Search ARD registries by natural-language query |
| `get_agent_card` | Fetch a specific artifact (agent card, skill, MCP descriptor) by URL |

## Sample Inputs

- `Find MCP servers for image processing`

- `Search for code review agents`

  *Returns skill and agent entries related to code review.*

- `What tools are available for background removal?`

- `Get the agent card at https://huggingface.co/api/agentfinder/skills/huggingface/rembg/SKILL.md`

  *Fetches the full skill markdown for the rembg Space.*

## How To

### Install

```bash
pip install google-adk-community
# Optional, for local (in-process) mode:
pip install hf-agentfinder
```

### Basic usage

```python
from google.adk import Agent
from google.adk_community.tools.ardhf import AgentFinderToolset

agent = Agent(
    name="my_agent",
    instruction="Search for tools when you need a capability.",
    tools=[AgentFinderToolset()],
)
```

### Remote vs local mode

By default, the toolset sends HTTP requests to the hosted HuggingFace Agent
Finder registry.  For in-process search (no network calls), install
`hf-agentfinder` and set `local=True`:

```python
toolset = AgentFinderToolset(local=True)
```

Or set environment variables:

```bash
export ARDHF_LOCAL=1
export HF_TOKEN=hf_...  # optional, for authenticated access
```

### Custom registry URL

Point to any ARD-compatible registry:

```python
toolset = AgentFinderToolset(
    registry_url="http://localhost:8090",  # e.g. challenge server
)
```

### Running the sample

```bash
# With adk web
adk web contributing/samples/ardhf

# Or directly
python -m contributing.samples.ardhf.agent
```

### Using the challenge server for deterministic testing

The `hf-agentfinder` package includes a deterministic challenge server:

```bash
# Terminal 1: start the challenge server
pip install hf-agentfinder
hf-agentfinder challenge serve --port 8090

# Terminal 2: run the agent against it
ARDHF_REGISTRY_URL=http://127.0.0.1:8090 \
    python -m contributing.samples.ardhf.agent
```

## Architecture

```
AgentFinderToolset (BaseToolset)
├── search_agents(query, artifact_type?, limit?)
│   ├── remote: HTTP POST to registry /search
│   └── local: agentfinder.server.search_agent_finder()
└── get_agent_card(url)
    └── HTTP GET to artifact URL
```

## Related

- [HuggingFace Agent Finder](https://github.com/huggingface/hf-agentfinder) — ARD reference implementation
- [ADK BaseToolset](https://google.github.io/adk-docs/) — ADK toolset documentation
