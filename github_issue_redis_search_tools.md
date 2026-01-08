# [Proposal] Add RedisVL Search Tools for Knowledge Base Retrieval

## Is your feature request related to a problem? Please describe.

ADK has retrieval tools for Google Cloud (Vertex AI Search, Discovery Engine), but **no self-hosted option** for developers who need full control over their infrastructure. RAG is a fundamental pattern for building useful agents, and Redis is already running in most organizations, these tools let developers add vector search, keyword search, and hybrid retrieval to their existing Redis infrastructure. [RedisVL](https://github.com/redis/redis-vl-python) is the official Redis Vector Search library that's also well adopted. 

Moreover, the repo has `RedisSessionService` for session persistence, but **there are no tools for agents to search Redis-based knowledge bases** without building it themselves.

## Describe the solution you'd like

Add `redisvl` as an optional dependency with four search tools that wrap RedisVL's query capabilities as ADK `BaseTool` implementations.

**pyproject.toml change:**
```toml
[project.optional-dependencies]
redis-vl = [
    "redisvl>=0.13.2",
    "nltk>=3.8.0",
    "sentence-transformers>=2.2.0",
]
```

**Installation:**
```bash
# Existing functionality unchanged
pip install google-adk-community

# Opt-in to vector search capabilities
pip install google-adk-community[redis-vl]
```

This aligns with the community repository's stated philosophy:

> "This approach allows the **core ADK to remain stable and lightweight**, while giving the community the freedom to build and share powerful extensions."

### Tools Provided

| Tool | Search Type | Use Case |
|------|-------------|----------|
| `RedisVectorSearchTool` | KNN vector similarity | Semantic/conceptual queries |
| `RedisTextSearchTool` | BM25 full-text | Exact terms, acronyms, API names |
| `RedisHybridSearchTool` | Vector + BM25 combined | Best of both worlds |
| `RedisRangeSearchTool` | Distance threshold | Exhaustive retrieval, quality filtering |

### Developer Experience

```python
from google.adk import Agent
from google.adk_community.tools.redis import RedisVectorSearchTool
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer

index = SearchIndex.from_yaml("schema.yaml")
index.connect("redis://localhost:6379")
vectorizer = HFTextVectorizer(model="redis/langcache-embed-v2")

tool = RedisVectorSearchTool(
    index=index,
    vectorizer=vectorizer,
    num_results=5,
    return_fields=["title", "content", "url"],
)

agent = Agent(model="gemini-2.5-flash", tools=[tool])
```

### Common Features Across All Tools

- **Filtering**: Tag, numeric, and geo filters via `filter_expression`
- **Field selection**: Control returned fields via `return_fields`
- **Async support**: Works with both `SearchIndex` and `AsyncSearchIndex`
- **Score normalization**: Convert distances to 0-1 similarity via `normalize_vector_distance=True`
- **Full parameter exposure**: All RedisVL query parameters are configurable

## Describe alternatives you've considered

1. **Implement vector search with raw Redis commands** — Would duplicate existing, maintained code in RedisVL. Users would get a degraded experience compared to using RedisVL directly.

2. **Require users to install RedisVL separately** — Creates friction and doesn't provide ADK-native abstractions like `BaseTool` wrappers with proper function declarations for LLMs.

3. **Use a different vector database** — Redis is already widely deployed in many organizations. Adding vector search to existing Redis infrastructure is lower friction than adopting a new database.



## Why RedisVL?

[RedisVL](https://github.com/redis/redis-vl-python) is the official Redis vector library (~50MB footprint). It provides:
- Schema-driven index management
- Multiple query types (vector, text, hybrid, range)
- Built-in vectorizers (HuggingFace, OpenAI, Cohere, etc.)
- Both sync and async APIs

