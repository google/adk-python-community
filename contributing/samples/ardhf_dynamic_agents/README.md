# ARDHF Dynamic Agents — Discover, Connect, Use

## Overview

This sample demonstrates an orchestrator agent that **dynamically discovers
and uses remote resources** at runtime using the ARDHF toolset. It supports
all three resource types available through
[ARD (Agentic Resource Discovery)](https://agenticresourcediscovery.org):

| Resource Type | Media Type | How It's Used |
|---|---|---|
| **A2A Agent** | `application/a2a-agent-card+json` | Delegate tasks via the A2A protocol |
| **MCP Server** | `application/mcp-server-card+json` | Connect to remote tool servers |
| **Skill** | `application/ai-skill` | Fetch and follow SKILL.md instructions |

Unlike a traditional multi-agent system where sub-agents are hardcoded at
build time, this orchestrator discovers capable resources on the fly and
uses each one according to its type.

## The Three Patterns

### Pattern 1: A2A Agent Delegation

For tasks requiring domain expertise, the orchestrator finds and delegates
to a remote A2A agent:

```
User: "How do I register an agent with AgentMsg?"

Orchestrator:
  1. search_agents("messaging agent registration") → finds AgentMsg Support
  2. get_agent_card(url) → inspects capabilities
  3. connect_agent(url, "How do I register?") → gets detailed answer via A2A
  4. Returns the answer to the user
```

**Live example** — The [AgentMsg Support Agent](https://agentmsg.net) is a
public A2A agent that answers questions about agent registration, messaging,
and encryption:

```
User: "Ask the AgentMsg support agent how to register"

Agent: I'll connect to the AgentMsg Support agent via A2A...
       [calls connect_agent with the agent card URL]
       
       Here's how to register an agent with AgentMsg:
       1. POST to /auth/request with your agent details
       2. Wait for admin approval
       3. Retrieve your Bearer token
       ...
```

The agent card URL for the AgentMsg Support agent:
```
https://agentmsg-support-462816930018.us-central1.run.app/.well-known/agent-card.json
```

### Pattern 2: MCP Server Tools

For tasks requiring specific tools, the orchestrator finds MCP servers and
reports their endpoints:

```
User: "I need a tool to analyze code complexity"

Orchestrator:
  1. search_tools("code complexity analysis") → finds MCP servers
  2. get_agent_card(url) → fetches server descriptor with endpoint
  3. Reports the MCP server URL and available tools to the user
```

MCP servers found via ARD search can be connected to using ADK's
`McpToolset` with `SseConnectionParams` or `StreamableHTTPConnectionParams`:

```python
from google.adk.tools.mcp_tool.mcp_toolset import (
    McpToolset,
    SseConnectionParams,
)

mcp_tools = McpToolset(
    connection_params=SseConnectionParams(
        url="https://example.com/mcp",
    ),
)
worker = Agent(name="mcp_worker", tools=[mcp_tools])
```

### Pattern 3: Skill Instructions

For tasks with established workflows, the orchestrator fetches and follows
skill instructions:

```
User: "How do I deploy a model to HuggingFace Spaces?"

Orchestrator:
  1. search_skills("deploy huggingface spaces") → finds relevant skills
  2. get_agent_card(url) → fetches SKILL.md content
  3. Follows the step-by-step instructions to guide the user
```

## Available Tools

| Tool | Description |
|---|---|
| `search_ards` | Search across all artifact types |
| `search_agents` | Search filtered to A2A agents |
| `search_skills` | Search filtered to skills |
| `search_tools` | Search filtered to MCP servers |
| `search_spaces` | Search filtered to HuggingFace Spaces |
| `get_agent_card` | Fetch a specific artifact by URL |
| `connect_agent` | Send a message to a remote A2A agent |

## How To

### Install

```bash
pip install google-adk-community
# Required for A2A agent connectivity:
pip install 'google-adk[a2a]'
```

### Run

```bash
# With adk web (interactive UI)
adk web contributing/samples/ardhf_dynamic_agents

# Or directly (runs the A2A demo)
python -m contributing.samples.ardhf_dynamic_agents.agent
```

### With the challenge server

```bash
# Terminal 1
hf-discover challenge serve --port 8090

# Terminal 2
ARDHF_REGISTRY_URL=http://127.0.0.1:8090 \
    adk web contributing/samples/ardhf_dynamic_agents
```

## Sample Conversations

### A2A: Ask the AgentMsg Support agent

```
User: Ask the AgentMsg support agent how to register an agent

Agent: I'll connect to the AgentMsg Support agent to get registration
       instructions.

       [searches → finds agent card → connects via A2A]

       The AgentMsg Support agent says:

       To register an agent with AgentMsg:
       1. POST to /auth/request with your agent_id, name, and description
       2. Wait for admin approval (check status via /auth/status/{token})
       3. Once approved, retrieve your Bearer token (shown only once)
       4. Use the token in Authorization headers for all API calls

       For local development, register directly via POST /agents/register.
```

### MCP: Find a code analysis tool

```
User: Find me a tool for analyzing code complexity

Agent: Let me search for MCP servers that provide code analysis tools.

       [searches → finds Code-Complexity-Analyzer MCP server]

       I found an MCP server for code complexity analysis. You can
       connect to it at:
         https://example.hf.space/mcp/code-complexity/server.json

       To use it in your agent, add it as an McpToolset:
         McpToolset(connection_params=SseConnectionParams(url="..."))
```

### Skills: Follow deployment instructions

```
User: How do I use ZeroGPU on HuggingFace?

Agent: Let me search for skills related to ZeroGPU.

       [searches → finds zerogpu skill → fetches SKILL.md]

       Based on the ZeroGPU skill instructions:
       1. Add @spaces.GPU decorator to your GPU functions
       2. Use gr.Blocks() for your Gradio interface
       3. Deploy to a ZeroGPU-enabled Space
       ...
```

## Architecture

```
User
  |
  v
+----------------------------------+
|  dynamic_orchestrator            |
|  (no built-in domain skills)     |
|                                  |
|  Tools:                          |
|   +-- search_ards ---------------+--> ARD Registry (all types)
|   +-- search_agents -------------+--> ARD Registry (A2A only)
|   +-- search_skills -------------+--> ARD Registry (skills only)
|   +-- search_tools --------------+--> ARD Registry (MCP only)
|   +-- search_spaces -------------+--> ARD Registry (Spaces only)
|   +-- get_agent_card ------------+--> Artifact URL
|   +-- connect_agent ------------+--> Remote A2A Agent
+----------------------------------+
         |                |                |
         v                v                v
  +-----------+    +------------+    +----------+
  | A2A Agent |    | MCP Server |    |  Skill   |
  | (via A2A  |    | (tool      |    | (SKILL.md|
  |  JSON-RPC)|    |  endpoint) |    |  content)|
  +-----------+    +------------+    +----------+
```

## Key Concepts

- **Dynamic discovery** -- The orchestrator doesn't know about resources
  at build time.  It discovers them at runtime through ARD search.
- **Type-aware delegation** -- Different resource types are used
  differently: A2A agents receive delegated tasks, MCP servers provide
  tools, and Skills provide instructions.
- **A2A protocol** -- Communication with remote agents uses the standard
  Agent-to-Agent protocol with JSON-RPC fallback, enabling interoperability
  across frameworks.
- **Graceful fallback** -- If no suitable resource is found, or if
  connection fails, the orchestrator reports this clearly to the user.

## Related

- [ARDHF basic sample](../ardhf/) -- Simpler sample focusing on discovery only.
- [ARD -- Agentic Resource Discovery](https://agenticresourcediscovery.org) -- Official ARD specification and documentation.
- [HuggingFace Discover](https://github.com/huggingface/hf-discover) -- ARD reference implementation.
- [A2A Protocol](https://github.com/google/A2A) -- Agent-to-Agent protocol specification.
- [AgentMsg](https://agentmsg.net) -- Store-and-forward message relay for AI agents (first public A2A agent).
