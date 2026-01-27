# Redis Agent Memory Sample

This sample demonstrates the **complete two-tier memory architecture** using Redis Agent Memory Server with ADK:

1. **RedisWorkingMemorySessionService** - Session management with auto-summarization
2. **RedisLongTermMemoryService** - Persistent long-term memory with semantic search

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ADK Agent                               │
├─────────────────────────────────────────────────────────────────┤
│  RedisWorkingMemorySessionService  │ RedisLongTermMemoryService │
│  (Tier 1: Working Memory)          │ (Tier 2: Long-Term Memory) │
├─────────────────────────────────────────────────────────────────┤
│                    Redis Agent Memory Server                    │
├─────────────────────────────────────────────────────────────────┤
│                         Redis Stack                             │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.10+
- Docker (for Redis Stack)
- Redis Agent Memory Server running

## Setup

### 1. Install Dependencies

```bash
pip install "google-adk-community[redis-agent-memory]"
```

### 2. Start Redis Stack

```bash
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```

### 3. Start Redis Agent Memory Server

```bash
git clone https://github.com/redis-developer/agent-memory-server.git
cd agent-memory-server
cp .env.example .env
# Edit .env: set OPENAI_API_KEY for embeddings
pip install -e .
uvicorn agent_memory_server.main:app --port 8000
```

### 4. Configure Environment

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
User: Hi, I'm Nitin. I'm an Machine Learning Engineer working on ML projects.
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

