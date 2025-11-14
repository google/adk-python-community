# OpenMemory Sample

This sample demonstrates how to use OpenMemory as a self-hosted memory backend
for ADK agents using the community package.

## Prerequisites

- Python 3.9+ (Python 3.11+ recommended)
- Docker (for running OpenMemory)
- ADK and ADK Community installed

## Setup

### 1. Install Dependencies

```bash
pip install google-adk google-adk-community
```

### 2. Set Up OpenMemory Server

Follow the [OpenMemory Quick Start Guide](https://openmemory.cavira.app/docs/quick-start) to install and configure your OpenMemory server.

Once OpenMemory is running, you'll need:
- The OpenMemory server URL (default: `http://localhost:8080`)
- An API key for authentication by setting OM_API_KEY=<your-secret-api-key> (configured in your OpenMemory server)

### 3. Configure Environment Variables for adk

Create a `.env` file in this directory :

```bash
# Required: Google API key for the agent
GOOGLE_API_KEY=your-google-api-key

# Required: OpenMemory API key for authentication
OPENMEMORY_API_KEY=your-openmemory-api-key

# Optional: OpenMemory base URL (defaults to http://localhost:8080)
OPENMEMORY_BASE_URL=http://localhost:8080
```

**Note:** `OPENMEMORY_API_KEY` is required for OpenMemory authentication.

## Usage

### Option 1: Using `get_fast_api_app` with URI (Recommended - Easiest)

The simplest way to use this sample is to run the included `main.py`:

```bash
python main.py
```

The `main.py` file demonstrates how to:
- Register the OpenMemory service factory for the `openmemory://` URI scheme
- Use the `openmemory://` URI scheme with `get_fast_api_app`
- Build the URI from environment variables

**Note:** The `adk web` CLI command won't work directly with this sample because it doesn't automatically register the OpenMemory service factory. You must use `main.py` which handles the registration.

### Option 2: Using `Runner` Directly

For programmatic usage, you can use `Runner` with OpenMemory service directly:

```python
from google.adk_community.memory import OpenMemoryService, OpenMemoryServiceConfig
from google.adk.runners import Runner

# Create OpenMemory service with API key (required)
memory_service = OpenMemoryService(
    base_url="http://localhost:8080",  # Adjust to match your OpenMemory server URL
    api_key="your-api-key"  # Required - get this from your OpenMemory server configuration
)

# Use with runner
runner = Runner(
    app_name="my_app",
    agent=root_agent,
    memory_service=memory_service
)
```

### Advanced Configuration

```python
from google.adk_community.memory import OpenMemoryService, OpenMemoryServiceConfig

# Custom configuration
config = OpenMemoryServiceConfig(
    search_top_k=20,              # Retrieve more memories per query
    timeout=10.0,                 # Faster timeout for production
    user_content_salience=0.9,    # Higher importance for user messages
    model_content_salience=0.75,   # Medium importance for model responses
    enable_metadata_tags=True     # Add tags (session, app, author) for filtering
                                  # When enabled, memories are tagged with session ID, app name,
                                  # and author, allowing search queries to filter by app name
)

memory_service = OpenMemoryService(
    base_url="http://localhost:8080",  # Adjust to match your OpenMemory server URL
    api_key="your-api-key",  # Required - get this from your OpenMemory server configuration
    config=config
)
```

## Sample Structure

```
open_memory/
├── main.py                    # FastAPI server using get_fast_api_app (Option 1)
├── open_memory_agent/
│   ├── __init__.py            # Agent package initialization
│   └── agent.py               # Agent definition with memory tools
└── README.md                  # This file
```

## Sample Agent

The sample agent (`open_memory_agent/agent.py`) includes:
- Memory tools (`load_memory`, `preload_memory`) for retrieving past conversations
- Auto-save callback that saves sessions to memory after each agent turn
- Time context for the agent to use current time in responses

## Sample Query
- hello my name is Amy and i love gaming
- I really love fps games like GameA and GameB, but my favourite is GameC. my favourite snack is doritos.

Then in a new session
- tell me everything you know about me
(Agent recalled prior details ( my name, love for gaming and mentioned games ,snacks etc))

## Configuration Options

### OpenMemory URI Format

When using `get_fast_api_app` (Option 1), you can use the `openmemory://` URI scheme:

- `openmemory://localhost:8080` - API key must be set via `OPENMEMORY_API_KEY` environment variable

### OpenMemoryServiceConfig

- `search_top_k` (int, default: 10): Maximum memories to retrieve per search
- `timeout` (float, default: 30.0): HTTP request timeout in seconds
- `user_content_salience` (float, default: 0.8): Importance for user messages
- `model_content_salience` (float, default: 0.7): Importance for model responses
- `default_salience` (float, default: 0.6): Fallback importance value
- `enable_metadata_tags` (bool, default: True): Include tags for filtering. When enabled,
  memories are tagged with `session:{session_id}`, `app:{app_name}`, and `author:{author}`.
  These tags are used to filter search results by app name, improving isolation between
  different applications using the same OpenMemory instance.

## Features

OpenMemory provides:

- **Multi-sector embeddings**: Factual, emotional, temporal, relational memory
- **Graceful decay curves**: Automatic reinforcement keeps relevant context sharp
- **Self-hosted**: Full data ownership, no vendor lock-in
- **High performance**: 2-3× faster than hosted alternatives
- **Cost-effective**: 6-10× cheaper than SaaS memory APIs

## Learn More

- [OpenMemory Documentation](https://openmemory.cavira.app/)
- [OpenMemory API Reference](https://openmemory.cavira.app/docs/api/add-memory)
- [ADK Memory Documentation](https://google.github.io/adk-docs)

