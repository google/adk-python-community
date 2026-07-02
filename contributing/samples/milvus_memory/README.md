# Milvus Memory Service sample

This sample shows how to use Milvus as an ADK `BaseMemoryService` backend for
cross-session memory.

`MilvusMemoryService` supports the same configuration shape for:

- Milvus Lite: local development with a local database path
- Milvus server: self-hosted Milvus, such as `http://localhost:19530`
- Zilliz Cloud: managed Milvus with a cloud endpoint and token

## Installation

```bash
pip install "google-adk-community[milvus]"
```

The `milvus` extra installs current `pymilvus` and Milvus Lite packages.
Milvus Lite 3.x local storage is not compatible with older 2.x local database
files or directories, so create a new local database path for new projects.

## Configuration

Use `MILVUS_URI` and `MILVUS_TOKEN` for all deployment modes:

```bash
# Milvus Lite
export MILVUS_URI="./adk_milvus_memory.db"

# Milvus server
export MILVUS_URI="http://localhost:19530"

# Zilliz Cloud
export MILVUS_URI="https://your-endpoint.api.gcp-us-west1.zillizcloud.com"
export MILVUS_TOKEN="your-token"
```

`MILVUS_TOKEN` is only needed for authenticated deployments such as Zilliz
Cloud. If you use a non-default Milvus database, set `MILVUS_DB_NAME`.

## Use with Runner

`MilvusMemoryService` accepts any embedding function that returns one vector per
input text. This example uses Gemini embeddings:

```python
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk_community.memory import MilvusMemoryService
from google.genai import Client


genai_client = Client()


def embedding_function(texts):
    response = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=list(texts),
    )
    return [list(embedding.values) for embedding in response.embeddings]


memory_service = MilvusMemoryService(
    embedding_function=embedding_function,
    dimension=3072,
    collection_name="adk_memory",
)

agent = Agent(
    name="memory_agent",
    model="gemini-flash-latest",
    instruction="Use memory to personalize responses when relevant.",
)

runner = Runner(
    app_name="milvus_memory_app",
    agent=agent,
    session_service=InMemorySessionService(),
    memory_service=memory_service,
)
```

You can also use another hosted embedding provider as long as `dimension`
matches the returned vectors. For example, OpenAI `text-embedding-3-small`
returns 1536-dimensional vectors:

```python
import os

import httpx


def embedding_function(texts):
    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": "text-embedding-3-small", "input": list(texts)},
        timeout=30,
    )
    response.raise_for_status()
    data = sorted(response.json()["data"], key=lambda item: item["index"])
    return [item["embedding"] for item in data]


memory_service = MilvusMemoryService(
    embedding_function=embedding_function,
    dimension=1536,
)
```

After a session has useful conversation history, persist it:

```python
session = await runner.session_service.get_session(
    app_name="milvus_memory_app",
    user_id="user-1",
    session_id="session-1",
)
await memory_service.add_session_to_memory(session)
```

Later, retrieve relevant memories:

```python
result = await memory_service.search_memory(
    app_name="milvus_memory_app",
    user_id="user-1",
    query="what did the user say about database preferences?",
)
for memory in result.memories:
    print(memory.content.parts[0].text)
```

## Notes

- `dimension` must match the embedding model output dimension.
- Re-ingesting the same ADK event uses a stable ID and updates the existing
  Milvus record instead of creating duplicates.
- Search is scoped by `app_name` and `user_id`, so users cannot retrieve each
  other's memories through the memory service.
