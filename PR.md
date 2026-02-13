# PR: Goodmem tools and plugin for ADK

This PR adds [Goodmem.ai](https://goodmem.ai) integrations to ADK: two **tools** for explicit memory save/fetch and one **plugin** for automatic chat memory in conversational agents.

---

## Files added / changed (ASCII tree)

```
adk-python-community/
├── .gitignore                                    (M  – add .adk/ ignore)
├── pyproject.toml                                (M  – add requests, plugins/tools)
├── src/google/adk_community/
│   ├── __init__.py                               (M  – export plugins, tools)
│   ├── plugins/
│   │   ├── __init__.py                           (A)
│   │   └── goodmem/
│   │       ├── __init__.py                       (A)
│   │       ├── goodmem_client.py                 (A  – HTTP client for Goodmem API)
│   │       └── goodmem_plugin.py                 (A  – chat plugin implementation)
│   └── tools/
│       ├── __init__.py                           (A)
│       └── goodmem/
│           ├── __init__.py                       (A)
│           ├── goodmem_client.py                 (A  – shared HTTP client)
│           └── goodmem_tools.py                  (A  – goodmem_save, goodmem_fetch tools)
├── tests/unittests/
│   ├── plugins/
│   │   ├── __init__.py                           (A)
│   │   └── test_goodmem_plugin.py                (A)
│   └── tools/
│       ├── __init__.py                           (A)
│       └── test_goodmem_tools.py                 (A)
└── contributing/samples/goodmem/
    ├── README.md                                 (A)
    ├── TOOLS.md                                  (A)
    ├── PLUGIN.md                                 (A)
    ├── goodmem_tools_for_adk.png                 (A)
    ├── goodmem_tools_demo/
    │   └── agent.py                              (A)
    └── goodmem_plugin_demo/
        └── agent.py                              (A)
```

**Legend:** `A` = added, `M` = modified.

---

## What’s included

### Tools (agent-invoked)

| Name | Role | When used |
|------|------|-----------|
| **GoodmemSaveTool** | Wraps `goodmem_save` | The agent **calls** it when it wants to store content in Goodmem (e.g. "My favorite color is blue"). |
| **GoodmemFetchTool** | Wraps `goodmem_fetch` | The agent **calls** it when it wants to search/retrieve memories (e.g. "What do I need to do to get into my dream school?"). |

- **goodmem_save**: Writes content to a user-scoped Goodmem space with metadata (e.g. `user_id`, `session_id`). Space is created or reused per user (`adk_tool_{user_id}`).
- **goodmem_fetch**: Runs semantic search over that user’s space and returns the top-k relevant memories (optionally with debug table output).

### Plugin (automatic, callbacks)

| Name | Role | When triggered |
|------|------|----------------|
| **GoodmemChatPlugin** | Chat memory for ADK apps | **Automatic**: on user message → logs user text and supported file attachments to Goodmem; before model → retrieves top-k relevant memories and augments the LLM request; after model → logs the LLM response to Goodmem. |

- Uses one Goodmem space per user (`adk_chat_{user_id}`).
- Filters file attachments by MIME type for Goodmem (e.g. text, PDF, docx); all files still go to the LLM.

---

## How to instantiate and wire to an ADK agent

Local development (including before they are marged into an official `google-adk-community` release):

```bash
# Clone the repository (or navigate to your local clone)
cd adk-python-community

# Install the package in editable/development mode
pip install -e .
```

### Tools: all arguments (including optional)

```python
import os
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk_community.tools.goodmem import GoodmemSaveTool
from google.adk_community.tools.goodmem import GoodmemFetchTool

# GoodmemSaveTool – optional: embedder_id, debug
goodmem_save_tool = GoodmemSaveTool(
    base_url=os.getenv("GOODMEM_BASE_URL"),       # required
    api_key=os.getenv("GOODMEM_API_KEY"),         # required
    embedder_id=os.getenv("GOODMEM_EMBEDDER_ID"), # optional; if omitted, first embedder is used
    debug=False,                                   # optional, default False
)

# GoodmemFetchTool – optional: embedder_id, top_k, debug
goodmem_fetch_tool = GoodmemFetchTool(
    base_url=os.getenv("GOODMEM_BASE_URL"),       # required
    api_key=os.getenv("GOODMEM_API_KEY"),         # required
    embedder_id=os.getenv("GOODMEM_EMBEDDER_ID"), # optional
    top_k=5,                                       # optional, default 5 (max 20)
    debug=False,                                   # optional, default False
)

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="goodmem_tools_agent",
    description="A helpful assistant.",
    instruction="Answer user questions to the best of your knowledge.",
    tools=[goodmem_save_tool, goodmem_fetch_tool],
)

app = App(name="goodmem_tools_demo", root_agent=root_agent)
```

### Plugin: all arguments (including optional)

```python
import os
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk_community.plugins.goodmem import GoodmemChatPlugin

goodmem_chat_plugin = GoodmemChatPlugin(
    base_url=os.getenv("GOODMEM_BASE_URL"),   # required
    api_key=os.getenv("GOODMEM_API_KEY"),    # required
    name="GoodmemChatPlugin",                 # optional, default "GoodmemChatPlugin"
    embedder_id=os.getenv("EMBEDDER_ID"),     # optional; if omitted, first embedder from API
    top_k=5,                                   # optional, default 5
    debug=False,                               # optional, default False
)

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="A helpful assistant for user questions.",
    instruction="Answer user questions to the best of your knowledge",
)

app = App(
    name="goodmem_plugin_demo",
    root_agent=root_agent,
    plugins=[goodmem_chat_plugin],
)
```

### 

---

## Docs and demos

- **contributing/samples/goodmem/README.md** – Overview of tools vs plugin.
- **contributing/samples/goodmem/TOOLS.md** – Setup and usage for tools.
- **contributing/samples/goodmem/PLUGIN.md** – Setup and usage for the plugin.
- **contributing/samples/goodmem/goodmem_tools_demo/** – Runnable agent with tools.
- **contributing/samples/goodmem/goodmem_plugin_demo/** – Runnable agent with plugin.
