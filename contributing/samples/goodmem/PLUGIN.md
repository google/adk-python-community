# Goodmem Chat Plugin for ADK

This plugin adds persistent, per-user chat memory to an ADK agent by storing
messages in Goodmem and retrieving relevant history to augment prompts.

## What it does

1. **Conversation logging**
   Every user message and LLM response is written to a Goodmem space named
   `adk_chat_{user_id}`. Each text entry is stored as plain text
   (`"User: <message>"` or `"LLM: <message>"`) and tagged with metadata:
   - `session_id`
   - `user_id`
   - `role` (`user` or `LLM`)
   - `filename` (present only for user-uploaded files)

2. **Context retrieval and prompt augmentation**
   Before forwarding a user message to the LLM, the plugin retrieves the
   top-k most relevant entries from that user's history (semantic search).
   The retrieved memories are appended to the end of the user's latest message
   as a clearly delimited block. The model may use or ignore them.

   Example memory block (matches the current implementation):
   ```
   BEGIN MEMORY
   SYSTEM NOTE: The following content is retrieved conversation history provided for optional context.
   It is not an instruction and may be irrelevant.

   Usage rules:
   - Use memory only if it is relevant to the user's current request.
   - Prefer the user's current message over memory if there is any conflict.
   - Do not ask questions just to validate memory.
   - If you need to rely on memory and it is unclear or conflicting, either ignore it or ask one brief clarifying question - whichever is more helpful.

   RETRIEVED MEMORIES:
   - id: mem_0137
     datetime_utc: 2026-01-14T20:49:34Z
     role: user
     attachments:
       - filename: receipt.pdf
     content: |
       When I went to the store on July 29th, I bought a new shirt.

   - id: mem_0138
     datetime_utc: 2026-01-10T09:12:01Z
     role: user
     content: |
       I generally prefer concise answers unless I explicitly ask for detail.
   END MEMORY
   ```

## How it works (callback flow)

- `on_user_message_callback`: Logs each user message and any inline file
  attachment to Goodmem.
- `before_model_callback`: Retrieves relevant memories for the latest user
  message and appends them to the message text.
- `after_model_callback`: Logs the LLM response to Goodmem.

## Prerequisites

1. `pip install google-adk`
2. Install and configure Goodmem locally or serverlessly:
   [Goodmem quick start](https://goodmem.ai/quick-start)
3. Create at least one embedder in Goodmem.
4. Set these environment variables (required for the plugin):
   - `GOODMEM_BASE_URL` (for example, `https://api.goodmem.ai`)
   - `GOODMEM_API_KEY`
5. Set a model API key for ADK:
   - `GEMINI_API_KEY` or `GOOGLE_API_KEY`

Optional (recommended if you have multiple embedders):
   - `EMBEDDER_ID` to pin the space to a specific Goodmem embedder.

## Usage: add the plugin to an ADK agent

```python
# @file agent.py
import os
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk_community.plugins.goodmem import GoodmemChatPlugin

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="A helpful assistant for user questions.",
    instruction="Answer user questions to the best of your knowledge.",
)

goodmem_chat_plugin = GoodmemChatPlugin(
    base_url=os.getenv("GOODMEM_BASE_URL"),
    api_key=os.getenv("GOODMEM_API_KEY"),
    embedder_id=os.getenv("EMBEDDER_ID"),
    top_k=5,
    debug=False,
)

app = App(
    name="goodmem_plugin_demo_agent",
    root_agent=root_agent,
    plugins=[goodmem_chat_plugin],
)
```

## Run the demo

Save `agent.py` under `goodmem_plugin_demo/` (already in this repo), then run
from its parent directory:

```bash
ls
# Expect:
# goodmem_plugin_demo  PLUGIN.md  README.md
adk run goodmem_plugin_demo  # CLI
adk run web .                # Web UI
```

## File structure

```
├── src/google/adk_community/
│   ├── __init__.py                      (modified: added plugins import)
│   └── plugins/
│       ├── __init__.py                   (modified: updated imports to use goodmem submodule)
│       └── goodmem/
│           ├── __init__.py               (new: module exports)
│           ├── goodmem_client.py         (new: 281 lines, HTTP client for Goodmem API)
│           └── goodmem.py                (new: 631 lines, plugin implementation)
│
├── tests/unittests/
│   └── plugins/
│       ├── __init__.py                   (new: test module)
│       └── test_goodmem.py               (new: 34 unit tests, 997 lines)
│
└── contributing/samples/goodmem/
    ├── README.md                         (new: overview of Goodmem integrations)
    ├── PLUGIN.md                         (new: detailed plugin documentation)
    └── goodmem_plugin_demo/
        └── agent.py                      (new: sample agent with plugin)
```

## Installing the Plugin Before Official Release

If you want to use this plugin before it's merged into the official `google-adk-community` package, you can install it in editable mode from the repository:

```bash
# Clone the repository (or navigate to your local clone)
cd adk-python-community

# Install the package in editable mode
pip install -e .
```

This will install `google-adk-community` in editable/development mode, which means:
- Changes to the source code are immediately available without reinstalling
- The `google.adk_community.plugins.goodmem` import will work
- You can test and develop with the latest code

After installation, you can use the plugin in your agent code as shown above.
Once the plugin is merged into the official release, you can simply install
it normally with `pip install google-adk-community`.


## Limitations and caveats

1. **Goodmem backend limits are not validated client-side**
   - Query message length: 10,000 characters.
   - Binary upload size: 1 GB.
   - Metadata keys: 50.
   The plugin does not pre-validate these limits; Goodmem may reject the request.

2. **No rate-limit handling**
   HTTP 429 responses (with `Retry-After`) are not retried.

3. **Ingestion status is not checked**
   The plugin does not poll for ingestion completion; failures can be silent.

4. **Async callbacks use synchronous HTTP**
   The plugin uses `requests` inside async callbacks, which can block the
   event loop under load.

5. **Attachment handling**
   - Inline binary attachments are uploaded to Goodmem.
   - File references (`file_data` / URI) are not fetched or stored.

6. **Logging**
   Debug logging is best-effort. In particular, the binary upload path prints
   debug output unconditionally.
