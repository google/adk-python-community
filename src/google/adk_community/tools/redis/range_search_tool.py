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
from typing import Union

from google.genai import types
from redisvl.index import AsyncSearchIndex
from redisvl.index import SearchIndex
from redisvl.query import VectorRangeQuery
from redisvl.utils.vectorize import BaseVectorizer

from .base_search_tool import VectorizedSearchTool
from .config import RedisRangeQueryConfig


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
      from google.adk_community.tools.redis import (
          RedisRangeSearchTool,
          RedisRangeQueryConfig,
      )

      index = SearchIndex.from_yaml("schema.yaml")
      vectorizer = HFTextVectorizer(model="redis/langcache-embed-v2")

      # Using config object (recommended)
      config = RedisRangeQueryConfig(
          distance_threshold=0.3,  # Only return docs within 0.3 distance
      )
      tool = RedisRangeSearchTool(
          index=index,
          vectorizer=vectorizer,
          config=config,
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
      config: Optional[RedisRangeQueryConfig] = None,
      return_fields: Optional[List[str]] = None,
      filter_expression: Optional[Any] = None,
      name: str = "redis_range_search",
      description: str = "Find all documents within a similarity threshold.",
  ):
    """Initialize the range search tool.

    Args:
        index: The RedisVL SearchIndex or AsyncSearchIndex to query.
        vectorizer: The vectorizer for embedding queries.
        config: Configuration for query parameters. If None, uses defaults.
            See RedisRangeQueryConfig for available options including
            distance_threshold, vector_field_name, and epsilon.
        return_fields: Optional list of fields to return in results.
        filter_expression: Optional filter expression to narrow results.
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
    self._config = config or RedisRangeQueryConfig()
    self._filter_expression = filter_expression

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
                        f" {self._config.distance_threshold})."
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
    # Allow runtime override of distance_threshold
    distance_threshold = kwargs.get(
        "distance_threshold", self._config.distance_threshold
    )

    # Get query kwargs from config
    query_kwargs = self._config.to_query_kwargs(
        vector=embedding,
        filter_expression=self._filter_expression,
    )
    query_kwargs["return_fields"] = self._return_fields
    query_kwargs["distance_threshold"] = distance_threshold

    return VectorRangeQuery(**query_kwargs)
