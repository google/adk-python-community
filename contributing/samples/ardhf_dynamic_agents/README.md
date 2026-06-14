# ARDHF Dynamic Agents — Search, Connect, Use

## Overview

This sample demonstrates an orchestrator agent that **dynamically discovers
and delegates to remote A2A agents** at runtime using the ARDHF toolset.

Unlike a traditional multi-agent system where sub-agents are hardcoded at
build time, this orchestrator discovers capable agents on the fly:

1. User sends a request the orchestrator can't handle alone.
2. Orchestrator searches ARD registries for a capable A2A agent.
3. Orchestrator inspects the agent card to verify compatibility.
4. Orchestrator delegates the task via the A2A protocol.
5. Orchestrator returns the result to the user.

This is the **discover -> connect -> use** pattern — agents finding and
collaborating with other agents without prior configuration.

## Sample Inputs

- `I need to remove the background from an image`

  *The orchestrator searches for image processing agents, finds one with
  background removal capability, and delegates via A2A.*

- `Review this code for security issues`

  *Searches for code review / security audit agents and delegates.*

- `Translate this document from English to Japanese`

  *Finds translation agents and delegates the task.*

## How To

### Install

```bash
pip install google-adk-community
# Required for A2A agent connectivity:
pip install 'google-adk[a2a]'
```

### Run

```bash
# With adk web
adk web contributing/samples/ardhf_dynamic_agents

# Or directly
python -m contributing.samples.ardhf_dynamic_agents.agent
```

### With the challenge server

```bash
# Terminal 1
hf-agentfinder challenge serve --port 8090

# Terminal 2
ARDHF_REGISTRY_URL=http://127.0.0.1:8090 \
    adk web contributing/samples/ardhf_dynamic_agents
```

## Architecture

```
User
  │
  ▼
┌──────────────────────────────┐
│  dynamic_orchestrator        │
│  (no built-in domain skills) │
│                              │
│  Tools:                      │
│   ├── search_agents ─────────┼──► ARD Registry
│   ├── get_agent_card ────────┼──► Agent Card URL
│   └── connect_agent ─────────┼──► Remote A2A Agent
└──────────────────────────────┘
                                      │
                                      ▼
                               ┌──────────────┐
                               │ Remote Agent  │
                               │ (via A2A)     │
                               └──────────────┘
```

## Key Concepts

- **Dynamic discovery** — The orchestrator doesn't know about sub-agents
  at build time.  It discovers them at runtime through ARD search.
- **A2A protocol** — Communication with remote agents uses the standard
  Agent-to-Agent protocol, enabling interoperability across frameworks.
- **Graceful fallback** — If no suitable agent is found, or if connection
  fails, the orchestrator reports this clearly to the user.

## Related

- [ARDHF basic sample](../ardhf/) — Simpler sample focusing on discovery only.
- [HuggingFace Agent Finder](https://github.com/huggingface/hf-agentfinder) — ARD reference implementation.
- [A2A Protocol](https://github.com/a2aproject/a2a-spec) — Agent-to-Agent protocol specification.
