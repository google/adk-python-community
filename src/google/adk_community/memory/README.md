# Memory Services

Community-contributed memory service implementations for the
[Google ADK](https://google.github.io/adk-docs/) framework.

## Available Services

### ValkeyMemoryService

A memory service backed by [Valkey](https://valkey.io/) using the
[Valkey Search module](https://valkey.io/topics/search/) for vector
similarity search. Uses the [valkey-glide](https://github.com/valkey-io/valkey-glide)
client library. This provides functionality analogous to
`VertexAiRagMemoryService` for users with Valkey infrastructure.

**Features:**
- Vector similarity search (HNSW) powered by the Valkey Search module
- Configurable embedding function (bring your own: OpenAI, Gemini, sentence-transformers, etc.)
- KNN search with pre-filtering by `app_name` and `user_id` TAG fields
- Configurable `vector_distance_threshold` for filtering low-quality matches
- Configurable distance metric (COSINE, L2, IP)
- Optional TTL for automatic memory expiration
- Batch embedding generation for efficient ingestion
- Supports `add_session_to_memory` and `add_events_to_memory`

**Requirements:**
- Valkey server with the Search module loaded (e.g.,
  [valkey-bundle](https://hub.docker.com/r/valkey/valkey-bundle) image)
- `valkey-glide >= 2.4.0`
- An embedding function (async callable)

**Installation:**

```bash
pip install google-adk-community[valkey]
```

**Usage:**

```python
from glide import GlideClient, GlideClientConfiguration, NodeAddress
from google.adk_community.memory import ValkeyMemoryService, ValkeyMemoryServiceConfig

# 1. Create a valkey-glide client
config = GlideClientConfiguration(
    addresses=[NodeAddress(host="localhost", port=6379)],
    client_name="my_adk_app",
)
client = await GlideClient.create(config)

# 2. Define your embedding function (bring your own model)
# Example with Google Gemini:
from google import genai
genai_client = genai.Client()

async def embed_texts(texts: list[str]) -> list[list[float]]:
    response = await genai_client.models.embed_content_async(
        model="text-embedding-004",
        contents=texts,
    )
    return [e.values for e in response.embeddings]

# 3. Create the memory service
memory_config = ValkeyMemoryServiceConfig(
    similarity_top_k=10,           # Max results per search (KNN)
    vector_distance_threshold=0.6, # Filter distant results (optional)
    embedding_dimensions=768,      # Must match your embedding model
    key_prefix="adk:memory",       # Valkey key prefix
    index_name="adk_memory_idx",   # Search index name
    distance_metric="COSINE",      # COSINE, L2, or IP
    ttl_seconds=None,              # Optional TTL (None = no expiry)
)
memory_service = ValkeyMemoryService(
    client=client,
    embedding_function=embed_texts,
    config=memory_config,
)

# The index is created automatically on first use, or explicitly:
await memory_service.create_index()

# 4. Use with ADK Runner
from google.adk.runners import Runner

runner = Runner(
    agent=my_agent,
    memory_service=memory_service,
    ...
)
```

**How it works:**

1. When `add_session_to_memory` is called, text is extracted from session
   events, embeddings are generated in batch using your embedding function,
   and each event is stored as a Valkey Hash with the embedding vector.

2. When `search_memory` is called, an embedding is generated for the query,
   then `FT.SEARCH` performs a KNN search with pre-filtering by `app_name`
   and `user_id` TAG fields. Results are returned ranked by vector similarity.

**Running Valkey with Search module:**

```bash
# Using podman
podman run -d --name valkey -p 6379:6379 valkey/valkey-bundle:9.1

# Using docker
docker run -d --name valkey -p 6379:6379 valkey/valkey-bundle:9.1
```

**Note on Redis Session Service:** The existing `RedisSessionService` in this
repo is wire-protocol compatible with Valkey. You can point it at a Valkey
instance directly for session storage without needing a separate Valkey
session service.

---

### OpenMemoryService

A memory service backed by [OpenMemory](https://openmemory.cavira.app/).
Uses HTTP API calls for memory storage and retrieval with LLM-powered
memory extraction.

**Installation:**

```bash
pip install google-adk-community
```

See the `OpenMemoryService` class documentation for usage details.
