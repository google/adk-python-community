# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load sample data into Redis for the redis_search_agent demo.

Data is designed to demonstrate when different search tools excel:
- Semantic search: conceptual queries, synonyms, paraphrasing
- Keyword search: technical terms, acronyms, exact phrases
- Range search: finding all highly relevant docs above threshold
- Hybrid search: combining both approaches
"""

import os
from pathlib import Path

from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer

# Documents designed to showcase different search tool strengths
SAMPLE_DOCS = [
    # === SEMANTIC SEARCH DEMOS ===
    # These use varied vocabulary - semantic search finds them even with different words
    {
        "title": "Introduction to Redis",
        "content": (
            "Redis is a lightning-fast in-memory data store. It excels at caching, "
            "session management, and real-time analytics. Think of it as a Swiss Army "
            "knife for data - versatile, quick, and reliable."
        ),
        "url": "https://redis.io/docs/about/",
        "category": "redis",
        "doc_type": "reference",
        "difficulty": "beginner",
    },
    {
        "title": "Understanding Vector Databases",
        "content": (
            "Vector databases store numerical representations of data called embeddings. "
            "These embeddings capture semantic meaning, enabling similarity search. "
            "Applications include recommendation engines, image search, and chatbots."
        ),
        "url": "https://redis.io/docs/vectors/",
        "category": "concepts",
        "doc_type": "reference",
        "difficulty": "intermediate",
    },
    {
        "title": "Building Intelligent Assistants",
        "content": (
            "Modern AI assistants combine language models with external knowledge. "
            "They can search databases, call APIs, and maintain conversation context. "
            "The key is giving them the right tools for each task."
        ),
        "url": "https://google.github.io/adk-docs/agents/",
        "category": "adk",
        "doc_type": "tutorial",
        "difficulty": "intermediate",
    },
    # === KEYWORD SEARCH DEMOS ===
    # These contain specific technical terms that keyword search finds precisely
    {
        "title": "HNSW Algorithm Deep Dive",
        "content": (
            "HNSW (Hierarchical Navigable Small World) is the algorithm Redis uses for "
            "approximate nearest neighbor search. It builds a multi-layer graph where "
            "each layer has exponentially fewer nodes. Search starts at the top layer "
            "and navigates down. Parameters: M (connections per node), EF (search width)."
        ),
        "url": "https://redis.io/docs/hnsw/",
        "category": "redis",
        "doc_type": "reference",
        "difficulty": "advanced",
    },
    {
        "title": "BM25 Scoring Explained",
        "content": (
            "BM25 (Best Matching 25) is a ranking function for full-text search. "
            "It improves on TF-IDF by adding document length normalization and term "
            "frequency saturation. Redis supports BM25STD and BM25 scorers. "
            "Formula: score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))"
        ),
        "url": "https://redis.io/docs/bm25/",
        "category": "redis",
        "doc_type": "reference",
        "difficulty": "advanced",
    },
    {
        "title": "RRF: Reciprocal Rank Fusion",
        "content": (
            "RRF (Reciprocal Rank Fusion) combines results from multiple search methods. "
            "Score = sum(1 / (k + rank)) across all methods. Default k=60. "
            "Used in hybrid search to merge vector and keyword results. "
            "Alternative: LINEAR combination with alpha weighting."
        ),
        "url": "https://redis.io/docs/hybrid/",
        "category": "redis",
        "doc_type": "reference",
        "difficulty": "advanced",
    },
    {
        "title": "RedisVL API Reference",
        "content": (
            "Key classes: SearchIndex, AsyncSearchIndex, VectorQuery, TextQuery, "
            "HybridQuery, VectorRangeQuery. Vectorizers: HFTextVectorizer (recommended, "
            "uses redis/langcache-embed-v2 model, 768 dimensions). Methods: index.query(), "
            "index.search(), index.load(), index.create(overwrite=True)."
        ),
        "url": "https://redis.io/docs/redisvl/api/",
        "category": "redis",
        "doc_type": "api",
        "difficulty": "intermediate",
    },
    # === RANGE SEARCH DEMOS ===
    # Highly related docs - range search finds all above threshold
    {
        "title": "RAG Architecture Overview",
        "content": (
            "Retrieval-Augmented Generation (RAG) enhances LLMs with external knowledge. "
            "Step 1: Embed the user query. Step 2: Search vector database for relevant "
            "documents. Step 3: Include retrieved context in LLM prompt. "
            "Step 4: Generate grounded response."
        ),
        "url": "https://redis.io/solutions/rag/",
        "category": "concepts",
        "doc_type": "tutorial",
        "difficulty": "intermediate",
    },
    {
        "title": "RAG Best Practices",
        "content": (
            "Tips for effective RAG: chunk documents appropriately (512-1024 tokens), "
            "use hybrid search for better recall, rerank results before prompting, "
            "include metadata for filtering, monitor retrieval quality metrics."
        ),
        "url": "https://redis.io/solutions/rag/best-practices/",
        "category": "concepts",
        "doc_type": "tutorial",
        "difficulty": "intermediate",
    },
    {
        "title": "RAG vs Fine-tuning",
        "content": (
            "RAG retrieves knowledge at query time; fine-tuning bakes it into weights. "
            "RAG pros: up-to-date info, traceable sources, no training cost. "
            "Fine-tuning pros: faster inference, specialized behavior. "
            "Often combined: fine-tune for style, RAG for facts."
        ),
        "url": "https://redis.io/solutions/rag/comparison/",
        "category": "concepts",
        "doc_type": "reference",
        "difficulty": "advanced",
    },
    # === ADK SPECIFIC ===
    {
        "title": "ADK Tool Development",
        "content": (
            "Create custom tools by subclassing BaseTool. Implement run_async() for "
            "the tool logic and _get_declaration() for the schema. Tools receive "
            "args dict and ToolContext. Return results as dict or string."
        ),
        "url": "https://google.github.io/adk-docs/tools/custom/",
        "category": "adk",
        "doc_type": "tutorial",
        "difficulty": "intermediate",
    },
    {
        "title": "ADK Agent Orchestration",
        "content": (
            "Agents can delegate to sub-agents using transfer_to_agent(). "
            "Define agent hierarchy with parent/child relationships. "
            "Use instruction prompts to guide when delegation occurs. "
            "Sub-agents inherit tools from parent unless overridden."
        ),
        "url": "https://google.github.io/adk-docs/agents/orchestration/",
        "category": "adk",
        "doc_type": "tutorial",
        "difficulty": "advanced",
    },
    # === FAQ STYLE (good for both search types) ===
    {
        "title": "FAQ: Redis Connection Issues",
        "content": (
            "Q: Connection refused error? A: Check Redis is running on the correct "
            "port (default 6379). Q: Authentication failed? A: Set REDIS_PASSWORD "
            "env var or pass password to connect(). Q: Timeout errors? A: Increase "
            "socket_timeout parameter or check network latency."
        ),
        "url": "https://redis.io/docs/faq/connection/",
        "category": "redis",
        "doc_type": "faq",
        "difficulty": "beginner",
    },
    {
        "title": "FAQ: Embedding Dimensions Mismatch",
        "content": (
            "Q: Dimension mismatch error? A: Ensure query embeddings match index "
            "dimensions. Common dimensions: OpenAI ada-002 (1536), langcache-embed-v2 "
            "(768), sentence-transformers (384-768). Check schema.yaml dims field."
        ),
        "url": "https://redis.io/docs/faq/vectors/",
        "category": "redis",
        "doc_type": "faq",
        "difficulty": "beginner",
    },
]


def load_data() -> None:
    """Load sample documents into Redis with embeddings."""
    schema_path = Path(__file__).parent / "schema.yaml"
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    print(f"Connecting to Redis at {redis_url}")
    index = SearchIndex.from_yaml(str(schema_path))
    index.connect(redis_url)

    print("Creating index (will overwrite if exists)...")
    index.create(overwrite=True)

    print("Generating embeddings...")
    vectorizer = HFTextVectorizer(model="redis/langcache-embed-v2")

    docs_with_embeddings = []
    for doc in SAMPLE_DOCS:
        embedding = vectorizer.embed(doc["content"], as_buffer=True)
        docs_with_embeddings.append({**doc, "embedding": embedding})
        print(f"  [{doc['doc_type']:9}] {doc['title']}")

    print("\nLoading documents into Redis...")
    index.load(docs_with_embeddings)

    print(f"""
Loaded {len(SAMPLE_DOCS)} documents.

Try these queries to see different tools in action:
  - "What is Redis?" → semantic_search (conceptual)
  - "HNSW algorithm" → keyword_search (exact term)
  - "Tell me everything about RAG" → range_search (exhaustive retrieval)
  - "BM25 formula" → keyword_search (technical)
  - "building chatbots" → semantic_search (synonym for assistants)

Run: adk web
""")


if __name__ == "__main__":
    load_data()

