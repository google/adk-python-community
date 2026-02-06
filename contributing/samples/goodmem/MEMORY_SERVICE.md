# Goodmem Memory Service for ADK

`GoodmemMemoryService` extends ADK's `BaseMemoryService` interface, giving
any ADK agent persistent, per-user memory backed by Goodmem.


## Basics about memory services

A memory service (base `BaseMemoryService`) is an abstraction with two methods:
1. `add_session_to_memory(session)` for writing
2. `search_memory(app_name, user_id, query)` for reading

A memory service is used by a Runner. But you cannot simply pass it to the Runner and expect it to work. 
Instead, the two methods above need to be manually configured as callbacks or paired with tools. And the way to do it is asymmetric for writing and reading.
* To write, a developer must pass the memory service's `add_session_to_memory` method to an `Agent`'s `after_agent_callback` callback. This callback is triggered after every agent turn. It passes the entire session object to the memory service, which will decide what to write to memory. Yes, in this sense, a memory service is like a plugin.
* Reading from memory is done via two ADK-provided tools, both of which call the memory service's `search_memory` (via `tool_context.search_memory`). **preload_memory** is invoked by ADK before each LLM request (via its `process_llm_request` hook) -- in this sense, it is not really a tool which is meant for LLM agent to decide when to call. **load_memory** is called by the LLM/agent when it chooses to search memory.

## What Goodmem's memory service does

It uses a Goodmem space named `adk_memory_{app_name}_{user_id}` to store conversation turns.
If the space does not exist, it is created using the first available embedder, or the embedder specified in `GOODMEM_EMBEDDER_ID`.

1. **Memory writing** It saves new conversation turns to Goodmem after each agent response.
   By default each turn is stored as **one** text memory (user and LLM in one chunk):
   ```
   User: <query>
   LLM: <response>
   ```
   It can be split into two memories per turn (separate `User: ...` and `LLM: ...`) by passing `split_turn=True` to `GoodmemMemoryService` (see Usage example below).
   Binary attachments (PDFs, images) from user events are always stored as
   separate memories via multipart upload.

2. **Semantic search and prompt formatting** (expands `BaseMemoryService.search_memory` and adds formatting)
   Retrieved memories are formatted into a single string for prompt injection like this:
   ```
   BEGIN MEMORY
   ...usage rules...
   RETRIEVED MEMORIES:
   - id: mem-abc123
     time: 2025-02-05 14:30
     content: |
       User: My favorite color is blue.
       LLM: I'll remember that your favorite color is blue.
   ...more memories...
   END MEMORY
   ```

## Prerequisites

1. `pip install google-adk google-adk-community`
2. Install and configure Goodmem locally or serverlessly:
   [Goodmem quick start](https://goodmem.ai/quick-start)
3. Create at least one embedder in Goodmem.
4. Set these environment variables:
   - `GOODMEM_API_KEY` (required)
   - `GOODMEM_BASE_URL` (optional, defaults to `https://api.goodmem.ai`)
   - `GOODMEM_EMBEDDER_ID` (optional; first available embedder is used if omitted)
5. Set a model API key for ADK:
   - `GOOGLE_API_KEY` or `GEMINI_API_KEY`

## Usage

Using a memory service requires three pieces: an `after_agent_callback` to
write, memory tools to read, and the service on the Runner.

```python
# @file agent.py
import os
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import load_memory, preload_memory
from google.adk_community.memory.goodmem import GoodmemMemoryService

memory_service = GoodmemMemoryService(
    base_url=os.getenv("GOODMEM_BASE_URL"),
    api_key=os.getenv("GOODMEM_API_KEY"),
    embedder_id=os.getenv("GOODMEM_EMBEDDER_ID"),
    top_k=10,   # Number of memories to retrieve per search, range: 1-100, default 5
    timeout=60.0,      # seconds, default 30.0
    split_turn=False,  # False: one memory per turn (User+LLM); True: two (User, LLM)
)

async def save_to_memory(callback_context: CallbackContext) -> None:
    await callback_context.add_session_to_memory()

agent = Agent(
    model="gemini-2.5-flash",
    name="my_agent",
    instruction="You are a helpful assistant with persistent memory.",
    after_agent_callback=save_to_memory,
    tools=[preload_memory, load_memory],
)

runner = Runner(
    app_name="my_app",
    agent=agent,
    memory_service=memory_service,
)
```

## Run the demo

This repo includes a ready-to-run demo in `goodmem_memory_service_demo/`.
The demo uses `adk web` to give you the ADK Dev UI.

**Important:** Run `adk web` from the **parent** directory
`contributing/samples/goodmem/`, not from inside `goodmem_memory_service_demo/`.
ADK discovers agents as subdirectories and loads `services.py` from that parent
to register the Goodmem memory service.

```bash
cd contributing/samples/goodmem
adk web --memory_service_uri="goodmem://env" .
```

Or from anywhere, passing the agents directory explicitly:

```bash
adk web --memory_service_uri="goodmem://env" contributing/samples/goodmem
```

This opens the ADK Dev UI at `http://localhost:8000`. Select **goodmem_memory_service_demo**
from the left panel. Chat with the agent in a session, then leave the session and
start a new session. The agent will remember information from earlier conversations.

> **Note:** `adk run` does not support memory services.  Use `adk web`.

The demo uses:
- `goodmem_memory_service_demo/agent.py` — agent definition with the `after_agent_callback` and memory tools (no Runner or memory_service; adk web creates those).
- `goodmem/services.py` — **required** for `adk web`: registers the Goodmem factory. Edit the `GoodmemMemoryService(...)` call there to set top_k, timeout, split_turn, or debug. ADK loads `services.py` only from the **agents root** (the directory you pass to `adk web`).


## Installation for local development

If you want to use this service with local changes, install from this repository in editable mode:

```bash
cd adk-python-community
pip install -e .
```

This makes `from google.adk_community.memory.goodmem import GoodmemMemoryService`
available immediately, and local changes are picked up without reinstalling.

## File structure

```text
adk-python-community/
├─ src/google/adk_community/
│  ├─ plugins/goodmem/
│  │  └─ client.py                     (shared HTTP client)
│  └─ memory/goodmem/
│     ├─ __init__.py
│     └─ goodmem_memory_service.py     (BaseMemoryService implementation)
├─ tests/unittests/memory/
│  └─ test_goodmem_memory_service.py
└─ contributing/samples/goodmem/
   ├─ MEMORY_SERVICE.md
   ├─ services.py                      (adk web: register goodmem factory at agents root)
   └─ goodmem_memory_service_demo/
      └─ agent.py
```

## Limitations and caveats

1. **`add_session_to_memory` receives a read-only `Session`**
   ADK's `BaseMemoryService.add_session_to_memory` receives a `Session` object,
   not a writable context.  The service cannot persist state (e.g., the space ID
   cache) in session state — it relies on in-memory caches instead.

2. **No rate-limit handling**
   HTTP 429 responses are not retried.

3. **Ingestion status is not polled**
   Binary uploads may still be processing when `add_session_to_memory` returns.

4. **Dedup is in-memory only**
   The processed-events index is per-process.  If the service is restarted,
   events from previous runs may be re-processed.

5. **Timeout is managed by the shared client**
   The `timeout` field in `GoodmemMemoryServiceConfig` is retained for
   configuration compatibility but is not currently passed to the shared client.
   The shared client uses its own per-method timeouts (30 s for most calls,
   120 s for binary uploads).


## Why should or shouldn't you use a memory service?

Functionally, a memory service is similar to a plugin + tool combination.

The benefit of a memory service is that it allows you to **swap backends without changing agent code**. You configure the memory service on the Runner, and `LoadMemoryTool` / `PreloadMemoryTool` just work against
whatever implementation is plugged in. Switch from `InMemoryMemoryService` (dev) to `VertexAiMemoryBankService` (prod) by changing one line:

  ```python
  # dev
  Runner(memory_service=InMemoryMemoryService(), ...)
  # prod
  Runner(memory_service=VertexAiMemoryBankService(agent_engine_id="..."), ...)
  ```

The disadvantage of a memory service is that its interface is deliberately minimal. Plugins and tools can offer finer-grained control: per-message storage (instead of paired turns), deletion, metadata filtering, and direct control over when each piece of content is stored.