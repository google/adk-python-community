# ARDHF — Agentic Resource Discovery (ARD) Toolset for ADK

## Overview

ARDHF wraps [HuggingFace's Agent Finder](https://github.com/huggingface/hf-agentfinder)
(ARD — Agentic Resource Discovery) as an ADK `BaseToolset`.  It gives any ADK
agent the ability to **discover, inspect, and connect to** agents, skills,
MCP servers, HuggingFace Spaces, and other agentic resources at runtime.

The core workflow is **discover → inspect → connect**:

1. **Discover** — search ARD registries for resources matching a natural-language query.
2. **Inspect** — fetch the full artifact (agent card, skill markdown, MCP descriptor) by URL.
3. **Connect** — send a message to a remote A2A agent and get a response (which may be the start of a long-running A2A conversation).

## Quick Start

### 1. Create the agent (`agent.py`)

```python
from google.adk import Agent
from google.adk_community.tools.ardhf import AgentFinderToolset

root_agent = Agent(
    name="discovery_agent",
    description="An agent that discovers and connects to agentic resources.",
    instruction="Search for agents, skills, and tools when you need a capability.",
    tools=[AgentFinderToolset()],
)
```

### 2. Run the app

```bash
# Interactive web UI
adk web .

# Or run programmatically
adk run .
```

The agent is an ADK app — serve it with `adk web` for the interactive UI,
or `adk run` for CLI mode.  The toolset provides all discovery and connection
tools automatically.

## Available Tools

| Tool | Description |
|---|---|
| `search_ards` | Search ARD registries across all artifact types (agents, skills, MCP servers, Spaces) |
| `search_agents` | Search filtered to A2A agents (`application/a2a-agent-card+json`) |
| `search_skills` | Search filtered to skills (`application/ai-skill`) |
| `search_tools` | Search filtered to MCP servers (`application/mcp-server+json`) |
| `search_spaces` | Search filtered to HuggingFace Spaces (`application/vnd.huggingface.space+json`) |
| `get_agent_card` | Fetch a specific artifact (agent card, skill markdown, MCP descriptor) by URL |
| `connect_agent` | Send a message to a remote A2A agent — may return an immediate response or start a long-running task with its own lifecycle |

The `search_agents`, `search_skills`, `search_tools`, and `search_spaces`
tools are convenience aliases — each calls the same core search logic with
the artifact type pre-set.  Use `search_ards` when you want to search across
all types at once.

## Realistic Scenarios

### Finding and using a Skill

> "Find a code review skill and apply it to my PR"

1. The agent calls `search_skills('code review')` to find skills related to code review.
2. It picks the best match and calls `get_agent_card(url)` to fetch the full skill markdown.
3. It reads the skill instructions and applies them to the user's code.

```
User: Find a skill for reviewing Python code
Agent: [calls search_skills('Python code review')]
Agent: Found "python-review-skill" — fetching details...
Agent: [calls get_agent_card('https://huggingface.co/.../SKILL.md')]
Agent: Here's what the skill covers: ...
```

### Finding and using an MCP Tool

> "Find a database query tool"

1. The agent calls `search_tools('database query')` to find MCP servers.
2. It calls `get_agent_card(url)` to fetch the MCP server descriptor.
3. The descriptor contains tool definitions that can be connected via `McpToolset`.

```
User: Find tools for querying SQL databases
Agent: [calls search_tools('SQL database query')]
Agent: Found "sql-executor" MCP server with tools: execute_query, list_tables
Agent: [calls get_agent_card('https://..../mcp-descriptor.json')]
Agent: The server exposes these tools: ...
```

### Finding and using a Skill + Tool together

> "Find both a triage skill and a labeling tool for my issues"

1. The agent calls `search_skills('issue triage')` to find a triage skill.
2. It calls `search_tools('issue labeling')` to find a labeling MCP server.
3. It combines the skill's instructions with the tool's capabilities.

### Connecting to a Remote A2A Agent

> "Find an image generation agent and ask it to make a logo"

1. The agent calls `search_agents('image generation')` to find A2A agents.
2. It inspects the best match with `get_agent_card(url)`.
3. It delegates the task with `connect_agent(url, 'Create a minimalist logo for a coffee shop')`.
4. The remote agent processes the request and returns the result.

```
User: Find an agent that can generate images and make me a logo
Agent: [calls search_agents('image generation')]
Agent: Found "image-gen-agent" — connecting...
Agent: [calls connect_agent('https://.../agent.json', 'Create a minimalist logo for a coffee shop')]
Agent: The image generation agent responded with: ...
```

**Note on A2A conversations:** `connect_agent` sends a single message and
collects the response, but the remote agent may return a **long-running task**
with its own lifecycle (submitted → working → completed).  The response you
get back may be the final result or an intermediate status.  This initial
exchange is the **beginning of an A2A conversation** — for multi-turn
interactions with a discovered agent, consider using ADK's `RemoteA2aAgent`
directly with the agent card URL returned by `get_agent_card`:

```python
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

# After discovering an agent via search_agents + get_agent_card:
remote = RemoteA2aAgent(
    name="discovered_agent",
    agent_card="https://example.com/.well-known/agent.json",
)

# Use as a sub-agent for ongoing A2A conversation
orchestrator = Agent(
    name="orchestrator",
    sub_agents=[remote],
)
```

### Discovering HuggingFace Spaces

> "Find a text-to-speech Space"

1. The agent calls `search_spaces('text to speech')` to find HF Spaces.
2. It calls `get_agent_card(url)` to fetch the Space metadata.
3. It presents the Space info (URL, description, capabilities) to the user.

```
User: Find a Space for text to speech
Agent: [calls search_spaces('text to speech')]
Agent: Found "bark-tts" Space — here are the details: ...
```

## Configuration

### Registry URL

By default, the toolset queries the hosted HuggingFace Agent Finder registry.
Point to any ARD-compatible registry:

```python
toolset = AgentFinderToolset(
    registry_url="http://localhost:8090",
)
```

Or set the environment variable:

```bash
export ARDHF_REGISTRY_URL=http://localhost:8090
```

### Authentication

Pass a HuggingFace token for authenticated registry access:

```python
toolset = AgentFinderToolset(token="hf_...")
```

Or set the environment variable:

```bash
export HF_TOKEN=hf_...
```

### Local mode

For in-process, offline-capable search (no HTTP requests), install the
`hf-agentfinder` package and enable local mode:

```python
toolset = AgentFinderToolset(local=True)
```

Or set the environment variable:

```bash
export ARDHF_LOCAL=1
```

### Environment variables summary

| Variable | Description |
|---|---|
| `ARDHF_REGISTRY_URL` | Override the default registry URL |
| `HF_TOKEN` | Bearer token for authenticated registry access |
| `ARDHF_LOCAL` | Set to `1` / `true` / `yes` to enable local mode |

## Customizations

### Filtering exposed tools

Use `tool_filter` to expose only specific tools to the agent:

```python
# Only expose search and inspect tools (no connect)
toolset = AgentFinderToolset(
    tool_filter=["search_ards", "search_agents", "get_agent_card"],
)
```

### Tool name prefix

Add a prefix to avoid name collisions with other toolsets:

```python
toolset = AgentFinderToolset(tool_name_prefix="ard")
# Tools become: ard_search_ards, ard_search_agents, etc.
```

### Multiple registries

Use multiple toolset instances to search different registries:

```python
hf_toolset = AgentFinderToolset(
    registry_url="https://huggingface.co/api/agentfinder",
    tool_name_prefix="hf",
)
internal_toolset = AgentFinderToolset(
    registry_url="https://internal.example.com/ard",
    tool_name_prefix="internal",
)

agent = Agent(
    name="multi_registry_agent",
    instruction="Search multiple registries for the best tool.",
    tools=[hf_toolset, internal_toolset],
)
```

### Combining with other ADK toolsets

ARDHF works alongside any other ADK toolset:

```python
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

agent = Agent(
    name="combined_agent",
    instruction="Use discovery and local tools together.",
    tools=[AgentFinderToolset(), McpToolset(...)],
)
```

## Testing

### Using the HF challenge server

The `hf-agentfinder` package includes a deterministic challenge server with
fixed fixtures — no API keys or network access needed:

```bash
# Terminal 1: start the challenge server
pip install hf-agentfinder
hf-agentfinder challenge serve --port 8090

# Terminal 2: run the sample app against it
cd contributing/samples/ardhf
ARDHF_REGISTRY_URL=http://127.0.0.1:8090 adk web .
```

### Running the unit tests

```bash
# Unit tests (no server needed)
pytest tests/unittests/tools/ardhf/ -v

# Integration tests (start challenge server first)
hf-agentfinder challenge serve --port 8090 &
pytest tests/unittests/tools/ardhf/ -v
```

## References

- [ARD Specification](https://github.com/nichochar/ard-spec) — Agentic Resource Discovery specification
- [HuggingFace Agent Finder](https://github.com/huggingface/hf-agentfinder) — ARD reference implementation
- [ai-catalog](https://github.com/nichochar/ai-catalog) — Curated agentic resource catalog
- [ADK Documentation](https://google.github.io/adk-docs/) — Google Agent Development Kit
- [A2A Protocol](https://github.com/a2aproject/a2a-spec) — Agent-to-Agent protocol specification
