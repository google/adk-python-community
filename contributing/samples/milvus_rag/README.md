# Milvus RAG Toolset sample

This sample shows how to use Milvus as a vector store for ADK retrieval tools.
`MilvusToolset` exposes a `milvus_similarity_search` tool that agents can call
to retrieve relevant context from indexed text.

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
export MILVUS_URI="./adk_milvus_rag.db"

# Milvus server
export MILVUS_URI="http://localhost:19530"

# Zilliz Cloud
export MILVUS_URI="https://your-endpoint.api.gcp-us-west1.zillizcloud.com"
export MILVUS_TOKEN="your-token"
```

`MILVUS_TOKEN` is only needed for authenticated deployments such as Zilliz
Cloud. If you use a non-default Milvus database, set `MILVUS_DB_NAME`.

## Build a Vector Store

`MilvusVectorStore` accepts any embedding function that returns one vector per
input text. This example uses OpenAI `text-embedding-3-small`, which returns
1536-dimensional vectors:

```python
import os

import httpx
from google.adk_community.tools.milvus import MilvusVectorStore
from google.adk_community.tools.milvus import MilvusVectorStoreSettings


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


vector_store = MilvusVectorStore(
    embedding_function=embedding_function,
    settings=MilvusVectorStoreSettings(
        collection_name="adk_rag",
        dimension=1536,
    ),
)

vector_store.add_texts(
    [
        "Milvus Lite is useful for local RAG development.",
        "Zilliz Cloud provides managed Milvus for production workloads.",
    ],
    metadatas=[
        {"source": "milvus-lite"},
        {"source": "zilliz-cloud"},
    ],
)
```

## Use with Agent

```python
from google.adk.agents import Agent
from google.adk_community.tools.milvus import MilvusToolset


milvus_toolset = MilvusToolset(vector_store=vector_store)
tools = await milvus_toolset.get_tools_with_prefix()

agent = Agent(
    name="rag_agent",
    model="gemini-flash-latest",
    instruction="Use retrieval context when answering questions.",
    tools=tools,
)
```

The exposed tool name is `milvus_similarity_search`. It accepts a single
`query` argument and returns:

```python
{
    "status": "SUCCESS",
    "rows": [
        {
            "id": "...",
            "content": "...",
            "source": "...",
            "metadata": {...},
            "distance": 0.12,
        }
    ],
}
```

## Notes

- `dimension` must match the embedding model output dimension.
- `MilvusVectorStore` creates the collection if it does not already exist and
  validates the existing schema before reuse.
- Tool names are prefixed through ADK's `BaseToolset` prefix mechanism, matching
  the pattern used by other ADK toolsets.
