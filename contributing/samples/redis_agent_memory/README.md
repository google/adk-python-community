# Redis Agent Memory Sample

This sample demonstrates the **complete two-tier memory architecture** using Redis Agent Memory Server with ADK:

1. **RedisWorkingMemorySessionService** - Session management with auto-summarization
2. **RedisLongTermMemoryService** - Persistent long-term memory with semantic search

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                          ADK Agent                             │
├──────────────────────────────┬─────────────────────────────────┤
│     TIER 1: Working Memory   │    TIER 2: Long-Term Memory     │
├──────────────────────────────┼─────────────────────────────────┤
│ • Current session messages   │ • Extracted facts & preferences │
│ • Auto-summarization         │ • Semantic vector search        │
│ • Context window management  │ • Cross-session persistence     │
│ • TTL support                │ • Recency-boosted retrieval     │
├──────────────────────────────┴─────────────────────────────────┤
│                    Agent Memory Server API                     │
├────────────────────────────────────────────────────────────────┤
│                         Redis Stack                            │
└────────────────────────────────────────────────────────────────┘
```

## Example Flow

```
User Message
     │
     ▼
┌─────────────┐    store     ┌──────────────────────┐
│  ADK Agent  │─────────────▶│   Working Memory     │
└─────────────┘              │  (current session)   │
     │                       └──────────┬───────────┘
     │                                  │ extract
     │ search                           ▼
     │                       ┌──────────────────────┐
     └──────────────────────▶│   Long-Term Memory   │
                             │  (all sessions)      │
                             └──────────────────────┘
```

## Prerequisites

- Python 3.10+
- Docker (for Redis Stack and Agent Memory Server)

## Setup

### 1. Install Dependencies

```bash
pip install "google-adk-community[redis-agent-memory]"
```

> **Important**: The server is NOT installed via pip - it's a separate service that must be running. The pip package only installs the client to communicate with it.

### 2. Start Redis Stack

```bash
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```

### 3. Start Agent Memory Server

```bash
docker run -d --name agent-memory-server -p 8000:8000 \
  -e REDIS_URL=redis://host.docker.internal:6379 \
  -e OPENAI_API_KEY=your-openai-key \
  redislabs/agent-memory-server:latest \
  agent-memory api --host 0.0.0.0 --port 8000 --task-backend=asyncio
```

> **Note**: The memory server requires an OpenAI API key for embeddings by default. See the [Agent Memory Server docs](https://redis.github.io/agent-memory-server/) for alternative embedding providers.

### 4. Verify Setup

```bash
curl http://localhost:8000/health
```

### 5. Configure Environment

Create `.env` in this directory:

```bash
GOOGLE_API_KEY=your-google-api-key
REDIS_MEMORY_SERVER_URL=http://localhost:8000
REDIS_MEMORY_NAMESPACE=adk_agent_memory
REDIS_MEMORY_EXTRACTION_STRATEGY=discrete
REDIS_MEMORY_CONTEXT_WINDOW=8000
REDIS_MEMORY_RECENCY_BOOST=true
```

## Usage

```bash
python main.py
```

Open http://localhost:8080 in your browser.

## Test Conversation

**Session 1** - Share information:
```
User: Hi, I'm Nitin. I'm a Machine Learning Engineer working on ML projects.
User: I love coffee, especially Berliner Früstuck Coffee from Berliner Kaffeerösterei. 
User: My favorite programming language is Python.
```

**Session 2** - Test memory recall:
```
User: What do you remember about me?
User: What's my favorite coffee?
```

## Features

| Feature | Working Memory (Tier 1) | Long-Term Memory (Tier 2) |
|---------|------------------------|---------------------------|
| Scope | Current session | All sessions |
| Auto-summarization | ✅ Yes | No |
| Semantic search | No | ✅ Yes |
| Fact extraction | Background | ✅ Persistent |
| TTL support | ✅ Yes | No |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_MEMORY_SERVER_URL` | `http://localhost:8000` | Memory server URL |
| `REDIS_MEMORY_NAMESPACE` | `adk_agent_memory` | Namespace for isolation |
| `REDIS_MEMORY_EXTRACTION_STRATEGY` | `discrete` | `discrete`, `summary`, `preferences` |
| `REDIS_MEMORY_CONTEXT_WINDOW` | `8000` | Max tokens before summarization |
| `REDIS_MEMORY_RECENCY_BOOST` | `true` | Boost recent memories in search |

## Memory Server Configuration

The Redis Agent Memory Server has important settings that affect memory extraction:

| Setting | Default | Description |
|---------|---------|-------------|
| `EXTRACTION_DEBOUNCE_SECONDS` | `300` (5 min) | Time between extraction runs per session |
| `LONG_TERM_MEMORY` | `true` | Enable long-term memory storage |
| `ENABLE_DISCRETE_MEMORY_EXTRACTION` | `true` | Enable fact extraction from messages |

**Note on Debouncing**: The memory server debounces extraction to avoid constantly re-extracting
from the same conversation. For testing, you can reduce `EXTRACTION_DEBOUNCE_SECONDS` to `5` in
the memory server's `.env` file.

