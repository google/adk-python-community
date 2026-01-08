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

"""Redis tools for ADK Community using RedisVL.

This module provides tools for Redis-based search operations:

- `RedisVectorSearchTool`: KNN vector similarity search
- `RedisHybridSearchTool`: Combined vector + BM25 text search
- `RedisRangeSearchTool`: Distance threshold-based vector search
- `RedisTextSearchTool`: Full-text BM25 keyword search

Example:
    ```python
    from redisvl.index import SearchIndex
    from redisvl.utils.vectorize import HFTextVectorizer
    from google.adk_community.tools.redis import RedisVectorSearchTool

    index = SearchIndex.from_yaml("schema.yaml")
    vectorizer = HFTextVectorizer(model="redis/langcache-embed-v2")

    tool = RedisVectorSearchTool(
        index=index,
        vectorizer=vectorizer,
        num_results=5,
    )
    ```
"""

try:
  from .base_search_tool import BaseRedisSearchTool
  from .base_search_tool import VectorizedSearchTool
  from .hybrid_search_tool import RedisHybridSearchTool
  from .range_search_tool import RedisRangeSearchTool
  from .text_search_tool import RedisTextSearchTool
  from .vector_search_tool import RedisVectorSearchTool
except ImportError as e:
  raise ImportError(
      "Redis tools require redisvl. "
      "Install with: pip install google-adk-community[redis-vl]"
  ) from e

__all__ = [
    "BaseRedisSearchTool",
    "VectorizedSearchTool",
    "RedisVectorSearchTool",
    "RedisHybridSearchTool",
    "RedisRangeSearchTool",
    "RedisTextSearchTool",
]
