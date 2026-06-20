# SOS Memory Agent Sample

This sample demonstrates using **SOS (Sovereign Operating System) Mirror** as a memory backend for ADK agents.

## What is SOS Mirror?

SOS Mirror is a semantic memory system with unique features:

- **FRC Physics**: Memories are ranked by Frequency (access count), Recency (time decay), and Context (semantic relevance)
- **Lineage Tracking**: Every memory has a cryptographic hash chain for provenance
- **Multi-Agent Isolation**: Each agent has its own memory namespace
- **Semantic Search**: Vector embeddings for similarity-based retrieval

## Prerequisites

1. **SOS Mirror API** running locally or remotely
   - See: https://github.com/servathadi/sos

2. **Environment variables**:
   ```bash
   export SOS_MIRROR_URL="http://localhost:8844"  # or your deployment URL
   export SOS_API_KEY="your-api-key"
   export GOOGLE_API_KEY="your-gemini-key"
   ```

## Quick Start

```bash
# Install dependencies
pip install google-adk google-adk-community

# Run the agent
cd contributing/samples/sos_memory
python main.py
```

## Using with ADK CLI

```bash
# Start web interface
adk web sos_memory_agent

# Or run in terminal
adk run sos_memory_agent
```

## Configuration

The `SOSMemoryServiceConfig` allows customization:

```python
from google.adk_community.memory import SOSMemoryService, SOSMemoryServiceConfig

config = SOSMemoryServiceConfig(
    search_top_k=10,           # Max memories per search
    timeout=30.0,              # Request timeout
    user_content_salience=0.8, # Weight for user messages
    model_content_salience=0.7,# Weight for model responses
    enable_lineage_tracking=True,
)

memory_service = SOSMemoryService(
    base_url="https://mirror.example.com",
    api_key="your-key",
    agent_id="my-agent",
    config=config,
)
```

## How FRC Physics Works

When searching memory, SOS ranks results using:

```
score = α·frequency + β·recency + γ·context_similarity
```

Where:
- **Frequency**: How often a memory has been accessed (builds importance over time)
- **Recency**: Time decay function (recent memories score higher)
- **Context**: Cosine similarity between query and memory embeddings

This means frequently-accessed, recent, and semantically-relevant memories surface first.

## Lineage Tracking

Every memory gets a lineage hash:

```python
hash = SHA256(previous_hash + agent_id + content + context)[:16]
```

This creates an immutable chain of memory provenance, useful for:
- Auditing agent decisions
- Debugging conversation flows
- Ensuring memory integrity

## Learn More

- [SOS GitHub](https://github.com/servathadi/sos)
- [ADK Documentation](https://google.github.io/adk-docs/)
- [ADK Community](https://github.com/google/adk-python-community)
