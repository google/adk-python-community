# ADK HITL Approval Dashboard

A drop-in **production-ready Human-in-the-Loop (HITL) approval middleware** for Google Agent Development Kit (ADK) agents — complete with an API backend and a demo Streamlit dashboard UI.

## The Problem Solved

ADK 1.x ships with an experimental `require_confirmation=True` feature that handles pausing the LLM loop for human verification. However, it is fundamentally built for local debugging and introduces major blockers to an enterprise environment:

1. **Incompatible with Persistent Sessions:** Native confirmations intentionally do not serialize well and will completely fail to resume your agent if you use `DatabaseSessionService`, `SpannerSessionService`, or `VertexAiSessionService` (the mandatory session backends for production deployments).
2. **Single-Agent Limitations:** They silently break across `AgentTool` nested bounds and true multi-agent (A2A) topologies, causing missing events or infinitely looping models.
3. **No Resilient Audit Log:** The native confirmation tool leaves no easily queryable paper trail linking the human supervisor to a precise LLM request.

*This project is the production implementation of the HITL pattern covered in the [ADK Multi-Agent Patterns Guide (Advent of Agents Day 13)](https://medium.com/@garythomasgeorge/why-google-adks-human-in-the-loop-story-has-a-production-gap-and-one-way-it-could-be-fixed-66aabef33a32).*

## What This Library Provides

This project solves the production gaps by explicitly decoupling the human approval payload from ADK's internal session memory. It introduces a session-agnostic REST API layer using an Adapter pattern.

### The 3-Layer Architecture

```
┌─────────────────────────────────────────┐
│     Dashboard UI (Streamlit)            │  Layer 3: Demo/reference UI
│     Approval inbox, audit log viewer    │  (Easily replaced by Zendesk/etc.)
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│     ApprovalRequest Model (Pydantic)    │  Layer 2: Normalised Contract API
│     FastAPI backend + SQLite store      │  Session-agnostic persistence
└────────────────┬────────────────────────┘
                 │
      ┌──────────┴───────────┐
┌─────▼──────┐    ┌──────────▼──────┐
│  ADK 1.x   │    │   ADK 2.0       │  Layer 1: Adapters
│  Adapter   │    │   Adapter       │  Only this changes between versions
└────────────┘    └─────────────────┘
```

By retaining HITL state inside an independent FastAPI engine and SQLite database, an active agent can pause safely. When a human supervisor hits "Approve" inside a centralized web portal hours later, the middleware simply posts the decision back into the agent's `/run_sse` stream seamlessly.

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `ADK_HITL_API_URL` | `http://localhost:8000` | URL of the HITL approval FastAPI backend. Override for Cloud Run or any remote deployment. |
| `ADK_HITL_POLL_INTERVAL_S` | `2.0` | Base polling interval in seconds. Up to 1s of random jitter is added automatically to reduce backend traffic under concurrent load. |

Set these before starting the gateway:

```bash
export ADK_HITL_API_URL="https://your-hitl-service.run.app"
export ADK_HITL_POLL_INTERVAL_S="3.0"
```

## Quick Start (Local Sandbox)

We have provided a demo customer service agent (`credit_agent`) alongside a launch script to test the interaction end-to-end.

1. Create your Python virtual environment and sync dependencies using `uv` (requires Python 3.11+):

```bash
uv venv --python "python3.11" ".venv"
source .venv/bin/activate
uv sync --all-extras
```

2. Start the FastAPI backend, Streamlit dashboard, and ADK Live Chat agent all at once:

```bash
./start_servers.sh
```

3. Open `http://localhost:8080` to chat with the agent and ask for a $75 account credit.
4. When the agent pauses and asks for a supervisor, open `http://localhost:8501` to approve or reject the request.

## How to Use in Your Own ADK Application

Wrapping an ADK agent with a formal enterprise HITL checkpoint takes under 5 lines of code:

1. Import the `hitl_tool` gateway wrapper.
2. Decorate your function tool.
3. Attach it to your ADK Agent initialization using a standard `FunctionTool`.

```python
from google.adk.tools import FunctionTool
from google.adk_community.tools.hitl.gateway import hitl_tool

# 1. Wrap your function with the decorator
@hitl_tool(agent_name="my_billing_agent")
async def issue_refund(user_id: str, amount: float):
    # This block won't execute until explicitly approved in the dashboard
    return {"status": "success", "amount_refunded": amount}

# 2. Attach to ADK Agent
root_agent = Agent(
    name="my_billing_agent",
    tools=[FunctionTool(issue_refund)]
)
```

## Production Integration Strategies

This repository acts as the production baseline for a contact center or enterprise orchestration grid. Once deployed to staging, consider swapping out:

- **Storage Layer:** Replace the local `SQLite` engine in `app/api/store.py` with `PostgreSQL` or `Cloud Spanner`.
- **Proactive Notification:** Hook the FastAPI `POST /approvals/` route into Slack, PagerDuty, or Microsoft Teams to actively ping channels when a high-risk request pops up.
- **Remove Streamlit:** Bypass the Streamlit frontend completely and point your existing support portal interface (like Salesforce Service Cloud) directly to `GET /approvals/pending` and `POST /approvals/{id}/decide`.

## ADK 2.0 Compatibility

This project currently uses ADK 1.x conventions and event triggers. Because it strictly implements an `adapters` layer, all the Pydantic API schemas and Streamlit logic are completely forward-compatible with ADK 2.0 `RequestInput` workflow yielding. You'll simply need to switch the adapter layer translation once ADK 2.0 exits Alpha. The `ADK_HITL_API_URL` and `ADK_HITL_POLL_INTERVAL_S` environment variables remain valid across both adapter versions.