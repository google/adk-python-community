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
from typing import Union

from google.genai import types
from redisvl.index import AsyncSearchIndex
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import BaseVectorizer

from .base_search_tool import VectorizedSearchTool
from .config import RedisVectorQueryConfig


class RedisVectorSearchTool(VectorizedSearchTool):
  """Vector similarity search tool using RedisVL.

  This tool performs K-nearest neighbor (KNN) vector similarity search
  over a Redis index. It embeds the query text using the provided
  vectorizer and finds the most similar documents.

  Example:
      ```python
      from redisvl.index import SearchIndex
      from redisvl.utils.vectorize import HFTextVectorizer
      from redisvl.query.filter import Tag
      from google.adk_community.tools.redis import (
          RedisVectorSearchTool,
          RedisVectorQueryConfig,
      )

      index = SearchIndex.from_yaml("schema.yaml")
      vectorizer = HFTextVectorizer(model="redis/langcache-embed-v2")

      # Using config object (recommended)
      config = RedisVectorQueryConfig(
          num_results=5,
          ef_runtime=100,  # Higher = better recall
      )
      tool = RedisVectorSearchTool(
          index=index,
          vectorizer=vectorizer,
          config=config,
          return_fields=["title", "content", "url"],
          filter_expression=Tag("category") == "redis",
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
      config: Optional[RedisVectorQueryConfig] = None,
      return_fields: Optional[List[str]] = None,
      filter_expression: Optional[Any] = None,
      name: str = "redis_vector_search",
      description: str = "Search for semantically similar documents using vector similarity with Redis.",
  ):
    """Initialize the vector search tool.

    Args:
        index: The RedisVL SearchIndex to query.
        vectorizer: The vectorizer for embedding queries.
        config: Configuration for query parameters. If None, uses defaults.
            See RedisVectorQueryConfig for available options including
            num_results, vector_field_name, dtype, and version-dependent
            parameters like ef_runtime and hybrid_policy.
        return_fields: Optional list of fields to return in results.
        filter_expression: Optional RedisVL FilterExpression to narrow results.
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
    self._config = config or RedisVectorQueryConfig()
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
                "num_results": types.Schema(
                    type=types.Type.INTEGER,
                    description=(
                        "Number of results to return (default:"
                        f" {self._config.num_results})."
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
    # Allow runtime override of num_results
    num_results = kwargs.get("num_results", self._config.num_results)

    # Get query kwargs from config, with runtime overrides
    query_kwargs = self._config.to_query_kwargs(
        vector=embedding,
        filter_expression=self._filter_expression,
    )
    query_kwargs["return_fields"] = self._return_fields
    query_kwargs["num_results"] = num_results

    return VectorQuery(**query_kwargs)
