# Memory Services

Community-contributed memory service implementations for the
[Google ADK](https://google.github.io/adk-docs/) framework.

## Available Services

### ValkeyMemoryService

A memory service backed by [Valkey](https://valkey.io/) using the
[Valkey Search module](https://valkey.io/topics/search/) for full-text
search. Uses the [valkey-glide](https://github.com/valkey-io/valkey-glide)
client library.

**Features:**
- Full-text search powered by the Valkey Search module (FT.CREATE / FT.SEARCH)
- Memories stored as Valkey Hash keys with automatic indexing
- TAG-based filtering by `app_name` and `user_id` for scoped queries
- Configurable TTL for automatic memory expiration
- Case-insensitive search out of the box

**Requirements:**
- Valkey server with the Search module loaded (e.g.,
  [valkey-bundle](https://hub.docker.com/r/valkey/valkey-bundle) image)
- `valkey-glide >= 2.4.0`

**Installation:**

```bash
pip install google-adk-community[valkey]
```

**Usage:**

```python
from glide import GlideClient, GlideClientConfiguration, NodeAddress
from google.adk_community.memory import ValkeyMemoryService, ValkeyMemoryServiceConfig

# Create a valkey-glide client
config = GlideClientConfiguration(
    addresses=[NodeAddress(host="localhost", port=6379)],
    client_name="my_adk_app",
)
client = await GlideClient.create(config)

# Create the memory service
memory_config = ValkeyMemoryServiceConfig(
    search_top_k=10,       # Max results per search
    key_prefix="adk:memory",  # Valkey key prefix
    index_name="adk_memory_idx",  # Search index name
    ttl_seconds=None,      # Optional TTL (None = no expiry)
)
memory_service = ValkeyMemoryService(client=client, config=memory_config)

# The index is created automatically on first use, or explicitly:
await memory_service.create_index()

# Use with ADK runner
from google.adk.runners import Runner

runner = Runner(
    agent=my_agent,
    memory_service=memory_service,
    ...
)
```

**Running Valkey with Search module:**

```bash
# Using podman
podman run -d --name valkey -p 6379:6379 valkey/valkey-bundle:9.1

# Using docker
docker run -d --name valkey -p 6379:6379 valkey/valkey-bundle:9.1
```

---

### OpenMemoryService

A memory service backed by [OpenMemory](https://openmemory.cavira.app/).
Uses HTTP API calls for memory storage and retrieval.

**Installation:**

```bash
pip install google-adk-community
```

See the `OpenMemoryService` class documentation for usage details.
