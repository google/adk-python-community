# Goodmem integrations with ADK

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

## Usage

* For tools, see [TOOLS.md](TOOLS.md) and the demo in `goodmem_tools_demo/`.
* For plugin, see [PLUGIN.md](PLUGIN.md) and the demo in `goodmem_plugin_demo/`.

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
