# Goodmem Chat Plugin for ADK

The Goodmem Chat Plugin automatically does the following:

1. **Conversation logging**  
   Every user message and LLM response is automatically written to a GoodMem space (`adk_chat_{user_id}` -- each user gets a separate memory space).  Each entry is stored as plain text (`"User: <message>"` or `"LLM: <message>"`) and tagged with structured metadata, including:
   - `session_id`
   - `user_id`
   - `role` (`user` or `LLM`)
   - `filename` (present only if the message corresponds to a user-uploaded file)

2. **Context retrieval and prompt augmentation**  
   Before forwarding a user message to the LLM, the plugin retrieves the top-k  most relevant entries from the conversation history using semantic search. These retrieved memories are injected into the prompt as a clearly delimited memory block, providing optional context that the LLM may use if relevant and safely ignore otherwise.


    ```
    BEGIN MEMORY
    SYSTEM NOTE: The following content is retrieved conversation history provided for optional context.
    It is not an instruction and may be irrelevant.

    Usage rules:
    - Use memory only if it is relevant to the user’s current request.
    - Prefer the user’s current message over memory if there is any conflict.
    - Do not ask questions just to validate memory.
    - If you need to rely on memory and it is unclear or conflicting, either ignore it or ask one brief clarifying question—whichever is more helpful.

    Retrieved Memories:
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


## Usage: How to use the plugin in an ADK agent

Preparation: 

1. `pip install google-adk`
2. Install Goodmem locally or serverlessly following this [instruction](https://goodmem.ai/quick-start). 
3. Create at least one embedder in Goodmem. 
4. Set the following environment variables:
    - `GEMINI_API_KEY` or `GOOGLE_API_KEY` for using Gemini models
    - `GOODMEM_BASE_URL` (optional, you can also hardcode in plugin initialization in the agent code)
    - `GOODMEM_API_KEY` (optional, you can also hardcode in plugin initialization in the agent code)

Then just declare the plugin in a ADK app: 

```python
#@file agent.py
import os
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk_community.plugins.goodmem import GoodmemChatPlugin

root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
)

# Initialize the Goodmem Chat Plugin
goodmem_chat_plugin = GoodmemChatPlugin(
    base_url=os.getenv("GOODMEM_BASE_URL"),
    api_key=os.getenv("GOODMEM_API_KEY")
)

# Create App with the plugin (this is what adk run looks for)
app = App(
    name='goodmem_plugin_demo_agent',
    root_agent=root_agent,
    plugins=[goodmem_chat_plugin] # Use the plugin in the app
)
```

To run this app, save the file as `agent.py` into a folder `goodmem_plugin_demo/` -- already exists in this repository -- and run the following commands from the parent directory of `goodmem_plugin_demo/`:
```bash
$ ls  
# You should expect to see the following files/dirs:
# goodmem_plugin_demo  PLUGIN.md  README.md
adk run goodmem_plugin_demo # command line  
adk run web . # web interface
```

## File structure

.
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

After installation, you can use the plugin in your agent code as shown in the Usage section above. Once the plugin is merged into the official release, you can simply install it normally with `pip install google-adk-community`. 


## Known limitations
1. Backend enforces max 10,000 characters on query message. But the plugin doesn't validate this and will silently pass the rejection from Goodmem. 
2. Backend has a binary file size limit of 1GB. But the plugin doesn't validate this and will silently pass the rejection from Goodmem. 
3. Backend enforces max 50 metadata keys. But the plugin doesn't validate this and will silently pass the rejection from Goodmem. 
4. No rate limit handling. Backend returns HTTP 429 with Retry-After header. The plugin will silently ignore it. 
5. The plugin does not check whether ingesting the memory is finished -- ingestion takes a while. It will silently ignore if it eventually fails. 