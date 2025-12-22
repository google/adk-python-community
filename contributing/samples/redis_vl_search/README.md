# RedisVL Search Agent

This sample demonstrates using Redis search tools to give an ADK agent
access to a Redis-based knowledge base with multiple search capabilities.

## What This Sample Shows

- Setting up a Redis vector index with a schema
- Using 3 Redis search tools in one agent (4th requires Redis 8.4+):
  - **RedisVectorSearchTool**: Semantic similarity search (KNN) - finds conceptually similar content
  - **RedisTextSearchTool**: Full-text keyword search (BM25) - matches exact terms and phrases
  - **RedisRangeSearchTool**: Distance threshold search - returns ALL docs within a relevance radius
  - **RedisHybridSearchTool**: Combined vector + text search (requires Redis 8.4+ and redis-py 7.1+)
- Integrating RedisVL with an ADK agent

## Prerequisites

1. **Redis Stack** running locally (or Redis Cloud with Search capability)
   ```bash
   # Using Docker
   docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
   ```

2. **No API keys needed for embeddings** - uses Redis' open-source `redis/langcache-embed-v2` model (768 dimensions)

## Setup

1. Install dependencies:
   ```bash
   pip install "google-adk-community[redis-vl]"
   ```

2. Download NLTK stopwords (required for keyword search):
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

3. Set environment variables (or create a `.env` file):
   ```bash
   export REDIS_URL=redis://localhost:6379
   export GOOGLE_API_KEY=your-google-api-key  # For Gemini LLM
   ```

4. Load sample data into Redis:
   ```bash
   cd contributing/samples/redis_vl_search
   python load_data.py
   ```

5. Run the agent:
   ```bash
   cd contributing/samples/redis_vl_search
   adk web
   ```

## Files

| File | Description |
|------|-------------|
| `schema.yaml` | Redis index schema defining document structure |
| `load_data.py` | Script to populate Redis with sample documents |
| `redis_vl_search_agent/agent.py` | Agent definition with all Redis search tools |

## How It Works

1. **Schema Definition** (`schema.yaml`): Defines the index structure with fields
   for title, content, URL, category, and a vector embedding field.

2. **Data Loading** (`load_data.py`): Populates Redis with sample documents about
   Redis and ADK, embedding the content using Redis' langcache-embed-v2 model.

3. **Agent** (`redis_vl_search_agent/agent.py`): Creates an agent with access to
   multiple search tools for different use cases.

## Search Tools

### semantic_search (RedisVectorSearchTool)
**Best for:** Conceptual questions, natural language queries, finding similar content.

**How it works:** Converts query to vector embedding, finds K nearest neighbors by cosine similarity.

**Returns:** Top-K most similar documents (default: 5).

**Example queries:**
- "What is Redis?" → finds docs about Redis even if they don't say "What is Redis"
- "How do I build a chatbot?" → finds docs about "intelligent assistants", "conversational AI"
- "Fast database for caching" → finds Redis docs even without exact keyword match

### keyword_search (RedisTextSearchTool)
**Best for:** Exact terms, acronyms, technical jargon, API names, error messages.

**How it works:** BM25 text scoring algorithm - matches exact tokens, weighs by term frequency.

**Returns:** Top-K documents ranked by keyword relevance.

**Example queries:**
- "HNSW algorithm" → exact match on "HNSW" acronym
- "BM25 formula" → finds docs containing "BM25"
- "VectorQuery class" → API/class name lookup
- "RRF ranking" → technical term that needs exact match

### range_search (RedisRangeSearchTool)
**Best for:** Exhaustive retrieval, comprehensive coverage, finding ALL related documents.

**How it works:** Returns ALL documents within a distance threshold (not just top-K).

**Returns:** Variable number - every document above the relevance bar.

**Use when:**
- User wants "everything" about a topic
- Comprehensive research needed
- Quality filtering (only highly relevant docs)
- Clustering/grouping similar content

**Example queries:**
- "Tell me everything about RAG pipelines" → returns all RAG-related docs
- "All Redis data structures" → comprehensive list
- "Complete guide to embeddings" → exhaustive retrieval

### hybrid_search (RedisHybridSearchTool)
**Best for:** Queries that benefit from both semantic understanding AND exact keyword matching.

**How it works:** Combines vector similarity + BM25 text scores using RRF or linear weighting.

**Requires:** Redis 8.4+ and redis-py 7.1+

## Example Queries

Once running, try asking the agent:

| Query | Expected Tool | Why |
|-------|---------------|-----|
| "What is Redis?" | semantic_search | Conceptual question |
| "HNSW algorithm details" | keyword_search | Technical acronym |
| "Tell me everything about RAG" | range_search | Exhaustive retrieval |
| "How do I build a chatbot?" | semantic_search | Natural language |
| "BM25 formula" | keyword_search | Exact term lookup |
| "All vector search methods" | range_search | Comprehensive coverage |

## Customization

### Using a Different Vectorizer

```python
from redisvl.utils.vectorize import HuggingFaceTextVectorizer

vectorizer = HuggingFaceTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
```

Note: Update `dims` in `schema.yaml` to match your model's embedding dimensions.

### Adding Filters

You can add filter expressions to narrow search results:

```python
from redisvl.query.filter import Tag

redis_search = RedisVectorSearchTool(
    index=index,
    vectorizer=vectorizer,
    num_results=5,
    return_fields=["title", "content", "url", "category"],
    filter_expression=Tag("category") == "redis",  # Only search Redis docs
)
```

See [RedisVL Filter documentation](https://docs.redisvl.com/api/filter.html) for more filter options.

### Advanced Query Options

`RedisVectorSearchTool` exposes all VectorQuery parameters:

```python
redis_search = RedisVectorSearchTool(
    index=index,
    vectorizer=vectorizer,
    num_results=10,
    return_fields=["title", "content"],
    # Query tuning
    dtype="float32",                    # Vector dtype
    return_score=True,                  # Include similarity score
    normalize_vector_distance=True,     # Convert to 0-1 similarity
    # Hybrid filtering
    filter_expression=Tag("category") == "redis",
    hybrid_policy="BATCHES",            # or "ADHOC_BF"
    batch_size=100,                     # For BATCHES policy
    # HNSW tuning
    ef_runtime=150,                     # Higher = better recall, slower
    epsilon=0.01,                       # Range search approximation
    # SVS-VAMANA tuning
    search_window_size=20,              # Search window size
    use_search_history="AUTO",          # "OFF", "ON", or "AUTO"
    search_buffer_capacity=30,          # 2-level compression tuning
)
```

See [RedisVL Query documentation](https://docs.redisvl.com/api/query.html) for details.

### Connecting to Redis Cloud

```bash
export REDIS_URL=redis://default:password@your-redis-cloud-host:port
```

