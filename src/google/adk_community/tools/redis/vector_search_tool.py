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

"""Redis vector similarity search tool using RedisVL."""

from __future__ import annotations

from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from google.genai import types
from redisvl.index import AsyncSearchIndex
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import BaseVectorizer

from .base_search_tool import BaseRedisSearchTool

# Type alias for sort specification
SortSpec = Optional[
    Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]]
]


class RedisVectorSearchTool(BaseRedisSearchTool):
  """Vector similarity search tool using RedisVL.

  This tool performs K-nearest neighbor (KNN) vector similarity search
  over a Redis index. It embeds the query text using the provided
  vectorizer and finds the most similar documents.

  Example:
      ```python
      from redisvl.index import SearchIndex
      from redisvl.utils.vectorize import HFTextVectorizer
      from redisvl.query.filter import Tag
      from google.adk_community.tools.redis import RedisVectorSearchTool

      index = SearchIndex.from_yaml("schema.yaml")
      vectorizer = HFTextVectorizer(model="redis/langcache-embed-v2")

      tool = RedisVectorSearchTool(
          index=index,
          vectorizer=vectorizer,
          num_results=5,
          return_fields=["title", "content", "url"],
          filter_expression=Tag("category") == "redis",  # Optional filter
      )

      # Use with an agent
      agent = Agent(model="gemini-2.5-flash", tools=[tool])
      ```
  """

  def __init__(
      self,
      *,
      index: Union[SearchIndex, AsyncSearchIndex],
      vectorizer: BaseVectorizer,
      vector_field_name: str = "embedding",
      num_results: int = 10,
      return_fields: Optional[List[str]] = None,
      filter_expression: Optional[Any] = None,
      dtype: str = "float32",
      return_score: bool = True,
      dialect: int = 2,
      sort_by: SortSpec = None,
      in_order: bool = False,
      hybrid_policy: Optional[str] = None,
      batch_size: Optional[int] = None,
      ef_runtime: Optional[int] = None,
      epsilon: Optional[float] = None,
      search_window_size: Optional[int] = None,
      use_search_history: Optional[str] = None,
      search_buffer_capacity: Optional[int] = None,
      normalize_vector_distance: bool = False,
      name: str = "redis_vector_search",
      description: str = "Search for semantically similar documents using vector similarity with Redis.",
  ):
    """Initialize the vector search tool.

    Args:
        index: The RedisVL SearchIndex to query.
        vectorizer: The vectorizer for embedding queries.
        vector_field_name: The name of the vector field in the index.
        num_results: Default number of results to return (default: 10).
        return_fields: Optional list of fields to return in results.
        filter_expression: Optional RedisVL FilterExpression to narrow results.
        dtype: The dtype of the vector (default: "float32").
        return_score: Whether to return the vector distance (default: True).
        dialect: The RediSearch query dialect (default: 2).
        sort_by: Field(s) to order results by. Can be str, tuple, or list.
        in_order: Require query terms in same order as document (default: False).
        hybrid_policy: Filter application policy - "BATCHES" or "ADHOC_BF".
        batch_size: Batch size when hybrid_policy is "BATCHES".
        ef_runtime: HNSW exploration factor at query time (higher = better recall).
        epsilon: Range search approximation factor for HNSW/SVS-VAMANA indexes.
        search_window_size: SVS-VAMANA search window size (higher = better recall).
        use_search_history: SVS-VAMANA history mode - "OFF", "ON", or "AUTO".
        search_buffer_capacity: SVS-VAMANA 2-level compression tuning parameter.
        normalize_vector_distance: Convert distance to similarity score 0-1 (default: False).
        name: The name of the tool (exposed to LLM).
        description: The description of the tool (exposed to LLM).
    """
    super().__init__(
        name=name,
        description=description,
        index=index,
        vectorizer=vectorizer,
        return_fields=return_fields,
    )
    self._vector_field_name = vector_field_name
    self._num_results = num_results
    self._filter_expression = filter_expression
    self._dtype = dtype
    self._return_score = return_score
    self._dialect = dialect
    self._sort_by = sort_by
    self._in_order = in_order
    self._hybrid_policy = hybrid_policy
    self._batch_size = batch_size
    self._ef_runtime = ef_runtime
    self._epsilon = epsilon
    self._search_window_size = search_window_size
    self._use_search_history = use_search_history
    self._search_buffer_capacity = search_buffer_capacity
    self._normalize_vector_distance = normalize_vector_distance

  def _get_declaration(self) -> types.FunctionDeclaration:
    """Get the function declaration for the LLM."""
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description="The search query text.",
                ),
                "num_results": types.Schema(
                    type=types.Type.INTEGER,
                    description=(
                        "Number of results to return (default:"
                        f" {self._num_results})."
                    ),
                ),
            },
            required=["query"],
        ),
    )

  def _build_query(
      self, query_text: str, embedding: List[float], **kwargs: Any
  ) -> VectorQuery:
    """Build a VectorQuery for KNN search.

    Args:
        query_text: The original query text (unused for vector search).
        embedding: The vector embedding of the query text.
        **kwargs: Additional parameters (e.g., num_results).

    Returns:
        A VectorQuery configured for KNN search.
    """
    num_results = kwargs.get("num_results", self._num_results)

    # Build query kwargs, only including optional params if set
    query_kwargs: dict[str, Any] = {
        "vector": embedding,
        "vector_field_name": self._vector_field_name,
        "num_results": num_results,
        "return_fields": self._return_fields,
        "filter_expression": self._filter_expression,
        "dtype": self._dtype,
        "return_score": self._return_score,
        "dialect": self._dialect,
        "sort_by": self._sort_by,
        "in_order": self._in_order,
        "normalize_vector_distance": self._normalize_vector_distance,
    }

    # Add optional parameters only if set (for version compatibility)
    if self._hybrid_policy is not None:
      query_kwargs["hybrid_policy"] = self._hybrid_policy
    if self._batch_size is not None:
      query_kwargs["batch_size"] = self._batch_size
    if self._ef_runtime is not None:
      query_kwargs["ef_runtime"] = self._ef_runtime
    if self._epsilon is not None:
      query_kwargs["epsilon"] = self._epsilon
    if self._search_window_size is not None:
      query_kwargs["search_window_size"] = self._search_window_size
    if self._use_search_history is not None:
      query_kwargs["use_search_history"] = self._use_search_history
    if self._search_buffer_capacity is not None:
      query_kwargs["search_buffer_capacity"] = self._search_buffer_capacity

    return VectorQuery(**query_kwargs)
