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

"""Redis hybrid search tool combining vector similarity and BM25 text search."""

from __future__ import annotations

from typing import Any
from typing import List
from typing import Optional
from typing import Union
import warnings

from google.genai import types
from packaging.version import parse
from redisvl.index import AsyncSearchIndex
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import BaseVectorizer

from .base_search_tool import VectorizedSearchTool
from .config import RedisAggregatedHybridQueryConfig
from .config import RedisHybridQueryConfig

# Minimum RedisVL version required for native FT.HYBRID support
_MIN_NATIVE_HYBRID_VERSION = "0.13.0"


def _get_redisvl_version() -> str:
  """Get the installed RedisVL version string.

  Returns:
      Version string (e.g., "0.13.0") or "0.0.0" if not available.
  """
  try:
    import redisvl

    return getattr(redisvl, "__version__", "0.0.0")
  except ImportError:
    return "0.0.0"


def _supports_native_hybrid() -> bool:
  """Check if the installed RedisVL version supports native HybridQuery.

  Returns:
      True if RedisVL >= 0.13.0, False otherwise.
  """
  try:
    return parse(_get_redisvl_version()) >= parse(_MIN_NATIVE_HYBRID_VERSION)
  except Exception:
    return False


class RedisHybridSearchTool(VectorizedSearchTool):
  """Hybrid search tool combining vector similarity and BM25 text search.

  This tool performs a hybrid search that combines semantic vector similarity
  with keyword-based BM25 text matching. It automatically detects the installed
  RedisVL version and uses the appropriate implementation:

  - **RedisVL >= 0.13.0**: Uses native FT.HYBRID command (server-side fusion)
    with RedisHybridQueryConfig. Requires Redis >= 8.4.0.
  - **RedisVL < 0.13.0**: Uses AggregateHybridQuery (client-side fusion)
    with RedisAggregatedHybridQueryConfig. Works with any Redis version.

  Example (native mode - RedisVL >= 0.13.0):
      ```python
      from google.adk_community.tools.redis import (
          RedisHybridSearchTool,
          RedisHybridQueryConfig,
      )

      config = RedisHybridQueryConfig(
          text_field_name="content",
          combination_method="LINEAR",
          linear_alpha=0.7,  # 70% text, 30% vector
      )
      tool = RedisHybridSearchTool(
          index=index,
          vectorizer=vectorizer,
          config=config,
      )
      ```

  Example (aggregate mode - older versions):
      ```python
      from google.adk_community.tools.redis import (
          RedisHybridSearchTool,
          RedisAggregatedHybridQueryConfig,
      )

      config = RedisAggregatedHybridQueryConfig(
          text_field_name="content",
          alpha=0.7,  # 70% text, 30% vector
      )
      tool = RedisHybridSearchTool(
          index=index,
          vectorizer=vectorizer,
          config=config,
      )
      ```
  """

  def __init__(
      self,
      *,
      index: Union[SearchIndex, AsyncSearchIndex],
      vectorizer: BaseVectorizer,
      config: Optional[
          Union[RedisHybridQueryConfig, RedisAggregatedHybridQueryConfig]
      ] = None,
      return_fields: Optional[List[str]] = None,
      filter_expression: Optional[Any] = None,
      name: str = "redis_hybrid_search",
      description: str = "Search using both semantic similarity and keyword matching.",
  ):
    """Initialize the hybrid search tool.

    Args:
        index: The RedisVL SearchIndex or AsyncSearchIndex to query.
        vectorizer: The vectorizer for embedding queries.
        config: Configuration for query parameters. Can be either:
            - RedisHybridQueryConfig: For native FT.HYBRID (RedisVL >= 0.13.0)
            - RedisAggregatedHybridQueryConfig: For client-side hybrid (older)
            If None, auto-detects based on installed RedisVL version.
        return_fields: Optional list of fields to return in results.
        filter_expression: Optional filter expression to narrow results.
        name: The name of the tool (exposed to LLM).
        description: The description of the tool (exposed to LLM).

    Raises:
        ValueError: If RedisHybridQueryConfig is used with RedisVL < 0.13.0.

    Warns:
        DeprecationWarning: If RedisAggregatedHybridQueryConfig is used when
            native hybrid is available (RedisVL >= 0.13.0).
    """
    super().__init__(
        name=name,
        description=description,
        index=index,
        vectorizer=vectorizer,
        return_fields=return_fields,
    )

    self._supports_native = _supports_native_hybrid()

    # Auto-detect config if not provided
    if config is None:
      if self._supports_native:
        config = RedisHybridQueryConfig()
      else:
        config = RedisAggregatedHybridQueryConfig()

    # Validate config compatibility with installed version
    self._use_native = isinstance(config, RedisHybridQueryConfig)

    if self._use_native and not self._supports_native:
      raise ValueError(
          "RedisHybridQueryConfig requires RedisVL >= 0.13.0 and Redis >="
          f" 8.4.0. Installed RedisVL version: {_get_redisvl_version()}. Use"
          " RedisAggregatedHybridQueryConfig for older versions."
      )

    if not self._use_native and self._supports_native:
      warnings.warn(
          "RedisAggregatedHybridQueryConfig is deprecated for RedisVL >="
          " 0.13.0. Consider using RedisHybridQueryConfig for native FT.HYBRID"
          " support with better performance. RedisAggregatedHybridQueryConfig"
          " will continue to work but uses client-side score combination.",
          DeprecationWarning,
          stacklevel=2,
      )

    self._config = config
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
  ) -> Any:
    """Build a query for combined vector + text search.

    Args:
        query_text: The original query text for BM25 matching.
        embedding: The vector embedding of the query text.
        **kwargs: Additional parameters (e.g., num_results).

    Returns:
        A HybridQuery or AggregateHybridQuery configured for hybrid search.
    """
    # Allow runtime override of num_results
    num_results = kwargs.get("num_results", self._config.num_results)

    # Get query kwargs from config
    query_kwargs = self._config.to_query_kwargs(
        text=query_text,
        vector=embedding,
        return_fields=self._return_fields,
        filter_expression=self._filter_expression,
    )
    query_kwargs["num_results"] = num_results

    if self._use_native:
      from redisvl.query import HybridQuery

      return HybridQuery(**query_kwargs)
    else:
      from redisvl.query import AggregateHybridQuery

      return AggregateHybridQuery(**query_kwargs)
