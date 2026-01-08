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

"""Redis search agent demonstrating vector, text, range, and hybrid search tools."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk import Agent
from google.adk_community.tools.redis import (
    RedisRangeSearchTool,
    RedisTextSearchTool,
    RedisVectorSearchTool,
)
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer

SCHEMA_PATH = Path(__file__).parent.parent / "schema.yaml"
RETURN_FIELDS = ["title", "content", "url", "category", "doc_type", "difficulty"]

INSTRUCTION = """You are a helpful assistant with a technical knowledge base.

You have 3 search tools, each optimized for different query types:

## semantic_search (Vector KNN)
Best for: Conceptual questions, natural language queries, finding similar content.
How it works: Converts query to vector embedding, finds K nearest neighbors by cosine similarity.
Returns: Top-K most similar documents (default: 5).
Example queries:
  - "What is Redis?" → finds docs about Redis even if they don't say "What is Redis"
  - "How do I build a chatbot?" → finds docs about "intelligent assistants", "conversational AI"
  - "Fast database for caching" → finds Redis docs even without exact keyword match
  - "Machine learning embeddings" → finds semantically related vector/AI content

## keyword_search (BM25 Full-Text)
Best for: Exact terms, acronyms, technical jargon, API names, error messages.
How it works: BM25 text scoring algorithm - matches exact tokens, weighs by term frequency.
Returns: Top-K documents ranked by keyword relevance.
Example queries:
  - "HNSW algorithm" → exact match on "HNSW" acronym
  - "BM25 formula" → finds docs containing "BM25"
  - "VectorQuery class" → API/class name lookup
  - "RRF ranking" → technical term that needs exact match
  - "ConnectionError redis" → error message lookup

## range_search (Vector Range/Threshold)
Best for: Exhaustive retrieval, comprehensive coverage, finding ALL related documents.
How it works: Returns ALL documents within a distance threshold (not just top-K).
Returns: Variable number - every document above the relevance bar.
Use when:
  - User wants "everything" about a topic: "Tell me everything about RAG"
  - Comprehensive research: "All vector search algorithms"
  - Quality filtering: Only docs highly relevant to the query
  - Clustering/grouping: Find all docs in a semantic neighborhood
  - Deduplication: Find near-duplicate content
Example queries:
  - "Tell me everything about RAG pipelines" → returns all RAG-related docs
  - "All Redis data structures" → comprehensive list
  - "Complete guide to embeddings" → exhaustive retrieval
  - "Every caching strategy" → all related documents, not just top 5

## Strategy
1. Start with the most appropriate tool based on query type
2. If results are poor or incomplete, try another tool
3. For broad topics, consider range_search first, then refine with others
4. For technical terms/acronyms, prefer keyword_search
5. For natural questions, prefer semantic_search

Always cite sources with title and difficulty level."""


def get_index(schema_path: Path, redis_url: str) -> SearchIndex:
    """Create and connect to Redis search index."""
    index = SearchIndex.from_yaml(str(schema_path))
    index.connect(redis_url)
    return index


def get_search_tools(index: SearchIndex, vectorizer: HFTextVectorizer) -> list:
    """Create search tools for the agent."""
    return [
        RedisVectorSearchTool(
            name="semantic_search",
            description="Semantic similarity search for conceptual queries.",
            index=index,
            vectorizer=vectorizer,
            num_results=5,
            return_fields=RETURN_FIELDS,
        ),
        RedisTextSearchTool(
            name="keyword_search",
            description="Keyword search for exact terms and phrases.",
            index=index,
            text_field_name="content",
            num_results=5,
            return_fields=RETURN_FIELDS,
            text_scorer="BM25STD",
        ),
        RedisRangeSearchTool(
            name="range_search",
            description=(
                "Vector range search - returns ALL documents within a semantic distance "
                "threshold (not just top-K). Use for: exhaustive topic retrieval, "
                "comprehensive coverage, quality-filtered results, finding all related "
                "documents, clustering similar content, or when you need everything "
                "above a relevance bar rather than a fixed number of results."
            ),
            index=index,
            vectorizer=vectorizer,
            distance_threshold=0.5,
            return_fields=RETURN_FIELDS,
        ),
    ]


def create_agent() -> Agent:
    """Create the Redis search agent."""
    load_dotenv()

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    index = get_index(SCHEMA_PATH, redis_url)
    vectorizer = HFTextVectorizer(model="redis/langcache-embed-v2")
    tools = get_search_tools(index, vectorizer)

    return Agent(
        model="gemini-2.5-flash",
        name="root_agent",
        instruction=INSTRUCTION,
        tools=tools,
    )


# Required for `adk web` to discover the agent
root_agent = create_agent()


if __name__ == "__main__":
    # For local testing
    print(f"Agent '{root_agent.name}' created with {len(root_agent.tools)} tools")
    for tool in root_agent.tools:
        print(f"  - {tool.name}")

