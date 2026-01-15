# Redis Agent Memory Sample

This sample demonstrates how to use the Redis Agent Memory Server as a long-term
memory backend for ADK agents using the community package.

## Prerequisites

- Python 3.10+ (Python 3.11+ recommended)
- Docker (for running Redis Stack)
- Redis Agent Memory Server running
- ADK and ADK Community installed

## Setup

### 1. Install Dependencies

```bash
pip install "google-adk-community[redis-agent-memory]"
```

### 2. Set Up Redis Stack

```bash
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```

### 3. Set Up Redis Agent Memory Server

Clone and run the Agent Memory Server:

```bash
git clone https://github.com/redis-developer/agent-memory-server.git
cd agent-memory-server
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (required for embeddings)
pip install -e .
uvicorn agent_memory_server.main:app --port 8000
```

### 4. Configure Environment Variables

Create a `.env` file in this directory:

```bash
# Required: Google API key for the agent
GOOGLE_API_KEY=your-google-api-key

# Optional: Redis Agent Memory Server URL (defaults to http://localhost:8000)
REDIS_AGENT_MEMORY_URL=http://localhost:8000

# Optional: Namespace for memory isolation (defaults to adk_sample)
REDIS_AGENT_MEMORY_NAMESPACE=adk_sample

# Optional: Extraction strategy - discrete, summary, or preferences (defaults to discrete)
REDIS_AGENT_MEMORY_EXTRACTION_STRATEGY=discrete

# Optional: Enable recency-boosted search (defaults to true)
REDIS_AGENT_MEMORY_RECENCY_BOOST=true

# Optional: Semantic similarity weight (defaults to 0.8)
REDIS_AGENT_MEMORY_SEMANTIC_WEIGHT=0.8

# Optional: Recency weight (defaults to 0.2)
REDIS_AGENT_MEMORY_RECENCY_WEIGHT=0.2
```

## Usage

### Option 1: Using `main.py` with FastAPI (Recommended)

```bash
python main.py
```

This starts the ADK web interface at `http://localhost:8080`.

### Option 2: Using `Runner` Directly

```python
from google.adk.runners import Runner
from google.adk.agents import LlmAgent
from google.adk_community.memory import (
    RedisAgentMemoryService,
    RedisAgentMemoryServiceConfig,
)

# Configure the memory service
config = RedisAgentMemoryServiceConfig(
    api_base_url="http://localhost:8000",
    default_namespace="my_app",
    extraction_strategy="discrete",
    enable_recency_boost=True,
)

# Create the memory service
memory_service = RedisAgentMemoryService(config=config)

# Use with ADK Runner
agent = LlmAgent(name="assistant", model="gemini-2.5-flash")
runner = Runner(
    app_name="my_app",
    agent=agent,
    memory_service=memory_service,
)
```

## Sample Structure

```
redis_agent_memory/
├── main.py                         # FastAPI server using get_fast_api_app
├── redis_agent_memory_agent/
│   ├── __init__.py                 # Agent package initialization
│   └── agent.py                    # Agent definition with memory tools
└── README.md                       # This file
```

## Sample Queries

Try these conversations to test long-term memory:

**Session 1:**
- "Hello, my name is Alex and I'm a software engineer"
- "I love hiking and photography. My favorite mountain is Mt. Rainier"

**Session 2 (new session):**
- "What do you remember about me?"
- "What are my hobbies?"

The agent should recall information from Session 1.

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `api_base_url` | `http://localhost:8000` | Agent Memory Server URL |
| `default_namespace` | `None` | Namespace for memory isolation |
| `extraction_strategy` | `discrete` | Memory extraction: `discrete`, `summary`, `preferences` |
| `recency_boost` | `True` | Enable recency-boosted semantic search |
| `semantic_weight` | `0.8` | Weight for semantic similarity (0-1) |
| `recency_weight` | `0.2` | Weight for recency (0-1) |

## Features

Redis Agent Memory Server provides:

- **Two-tier memory**: Working memory (session) + Long-term memory (persistent)
- **Intelligent extraction**: Automatically extracts facts, preferences, and episodic memories
- **Recency-boosted search**: Balances semantic relevance with temporal freshness
- **Vector search**: High-performance semantic search powered by Redis Stack
- **Namespace isolation**: Separate memory spaces for different apps/users

## Learn More

- [Redis Agent Memory Server](https://github.com/redis-developer/agent-memory-server)
- [ADK Memory Documentation](https://google.github.io/adk-docs)

