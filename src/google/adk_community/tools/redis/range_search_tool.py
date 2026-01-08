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

"""Redis vector range search tool using distance threshold."""

from __future__ import annotations

from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from google.genai import types
from redisvl.index import AsyncSearchIndex
from redisvl.index import SearchIndex
from redisvl.query import VectorRangeQuery
from redisvl.utils.vectorize import BaseVectorizer

from .base_search_tool import VectorizedSearchTool

# Type alias for sort specification
SortSpec = Optional[
    Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]]
]


class RedisRangeSearchTool(VectorizedSearchTool):
  """Vector range search tool using distance threshold.

  This tool finds all documents within a specified distance threshold
  from the query vector. Unlike KNN search which returns a fixed number
  of results, range search returns all documents that are "close enough"
  based on the threshold.

  Example:
      ```python
      from redisvl.index import SearchIndex
      from redisvl.utils.vectorize import HFTextVectorizer
      from google.adk_community.tools.redis import RedisRangeSearchTool

      index = SearchIndex.from_yaml("schema.yaml")
      vectorizer = HFTextVectorizer(model="redis/langcache-embed-v2")

      tool = RedisRangeSearchTool(
          index=index,
          vectorizer=vectorizer,
          distance_threshold=0.3,  # Only return docs within 0.3 distance
          return_fields=["title", "content"],
      )

      agent = Agent(model="gemini-2.5-flash", tools=[tool])
      ```
  """

  def __init__(
      self,
      *,
      index: Union[SearchIndex, AsyncSearchIndex],
      vectorizer: BaseVectorizer,
      vector_field_name: str = "embedding",
      distance_threshold: float = 0.2,
      num_results: int = 10,
      return_fields: Optional[List[str]] = None,
      filter_expression: Optional[Any] = None,
      dtype: str = "float32",
      return_score: bool = True,
      dialect: int = 2,
      sort_by: SortSpec = None,
      in_order: bool = False,
      epsilon: Optional[float] = None,
      normalize_vector_distance: bool = False,
      name: str = "redis_range_search",
      description: str = "Find all documents within a similarity threshold.",
  ):
    """Initialize the range search tool.

    Args:
        index: The RedisVL SearchIndex or AsyncSearchIndex to query.
        vectorizer: The vectorizer for embedding queries.
        vector_field_name: The name of the vector field in the index.
        distance_threshold: Maximum distance for results (default: 0.2).
        num_results: Maximum number of results to return (default: 10).
        return_fields: Optional list of fields to return in results.
        filter_expression: Optional filter expression to narrow results.
        dtype: The dtype of the vector (default: "float32").
        return_score: Whether to return the vector distance (default: True).
        dialect: The RediSearch query dialect (default: 2).
        sort_by: Field(s) to order results by.
        in_order: Require query terms in same order (default: False).
        epsilon: Range search approximation factor for HNSW/SVS-VAMANA.
        normalize_vector_distance: Convert distance to 0-1 similarity.
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
    self._distance_threshold = distance_threshold
    self._num_results = num_results
    self._filter_expression = filter_expression
    self._dtype = dtype
    self._return_score = return_score
    self._dialect = dialect
    self._sort_by = sort_by
    self._in_order = in_order
    self._epsilon = epsilon
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
                "distance_threshold": types.Schema(
                    type=types.Type.NUMBER,
                    description=(
                        "Max distance threshold (default:"
                        f" {self._distance_threshold})."
                    ),
                ),
            },
            required=["query"],
        ),
    )

  def _build_query(
      self, query_text: str, embedding: List[float], **kwargs: Any
  ) -> VectorRangeQuery:
    """Build a VectorRangeQuery for distance-based search.

    Args:
        query_text: The original query text (unused for range search).
        embedding: The vector embedding of the query text.
        **kwargs: Additional parameters (e.g., distance_threshold).

    Returns:
        A VectorRangeQuery configured for range search.
    """
    distance_threshold = kwargs.get(
        "distance_threshold", self._distance_threshold
    )

    query_kwargs: dict[str, Any] = {
        "vector": embedding,
        "vector_field_name": self._vector_field_name,
        "distance_threshold": distance_threshold,
        "num_results": self._num_results,
        "return_fields": self._return_fields,
        "filter_expression": self._filter_expression,
        "dtype": self._dtype,
        "return_score": self._return_score,
        "dialect": self._dialect,
        "sort_by": self._sort_by,
        "in_order": self._in_order,
        "normalize_vector_distance": self._normalize_vector_distance,
    }

    if self._epsilon is not None:
      query_kwargs["epsilon"] = self._epsilon

    return VectorRangeQuery(**query_kwargs)
