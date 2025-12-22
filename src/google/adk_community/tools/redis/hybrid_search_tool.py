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
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union

from google.genai import types
from redisvl.index import AsyncSearchIndex
from redisvl.index import SearchIndex
from redisvl.query import HybridQuery
from redisvl.utils.vectorize import BaseVectorizer

from .base_search_tool import BaseRedisSearchTool


class RedisHybridSearchTool(BaseRedisSearchTool):
  """Hybrid search tool combining vector similarity and BM25 text search.

  This tool performs a hybrid search that combines semantic vector similarity
  with keyword-based BM25 text matching. This is useful when you want to
  leverage both the semantic understanding of embeddings and the precision
  of keyword matching.

  Example:
      ```python
      from redisvl.index import SearchIndex
      from redisvl.utils.vectorize import HFTextVectorizer
      from google.adk_community.tools.redis import RedisHybridSearchTool

      index = SearchIndex.from_yaml("schema.yaml")
      vectorizer = HFTextVectorizer(model="redis/langcache-embed-v2")

      tool = RedisHybridSearchTool(
          index=index,
          vectorizer=vectorizer,
          text_field_name="content",
          linear_alpha=0.7,  # 70% text, 30% vector
          num_results=10,
          return_fields=["title", "content"],
      )

      agent = Agent(model="gemini-2.0-flash", tools=[tool])
      ```
  """

  def __init__(
      self,
      *,
      index: Union[SearchIndex, AsyncSearchIndex],
      vectorizer: BaseVectorizer,
      text_field_name: str = "content",
      vector_field_name: str = "embedding",
      text_scorer: str = "BM25STD",
      combination_method: Optional[str] = None,
      linear_alpha: float = 0.3,
      rrf_window: int = 20,
      rrf_constant: int = 60,
      num_results: int = 10,
      return_fields: Optional[List[str]] = None,
      filter_expression: Optional[Any] = None,
      dtype: str = "float32",
      stopwords: Optional[Union[str, Set[str]]] = "english",
      text_weights: Optional[Dict[str, float]] = None,
      name: str = "redis_hybrid_search",
      description: str = "Search using both semantic similarity and keyword matching.",
  ):
    """Initialize the hybrid search tool.

    Args:
        index: The RedisVL SearchIndex or AsyncSearchIndex to query.
        vectorizer: The vectorizer for embedding queries.
        text_field_name: The name of the text field for BM25 search.
        vector_field_name: The name of the vector field for similarity search.
        text_scorer: The text scoring algorithm (default: "BM25STD").
        combination_method: Score combination method - "RRF" or "LINEAR".
        linear_alpha: Weight of text score when using LINEAR (default: 0.3).
        rrf_window: Window size for RRF combination (default: 20).
        rrf_constant: Constant for RRF combination (default: 60).
        num_results: Default number of results to return (default: 10).
        return_fields: Optional list of fields to return in results.
        filter_expression: Optional filter expression to narrow results.
        dtype: The dtype of the vector (default: "float32").
        stopwords: Stopwords to remove from query (default: "english").
        text_weights: Optional field weights for text scoring.
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
    self._text_field_name = text_field_name
    self._vector_field_name = vector_field_name
    self._text_scorer = text_scorer
    self._combination_method = combination_method
    self._linear_alpha = linear_alpha
    self._rrf_window = rrf_window
    self._rrf_constant = rrf_constant
    self._num_results = num_results
    self._filter_expression = filter_expression
    self._dtype = dtype
    self._stopwords = stopwords
    self._text_weights = text_weights

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
  ) -> HybridQuery:
    """Build a HybridQuery for combined vector + text search.

    Args:
        query_text: The original query text for BM25 matching.
        embedding: The vector embedding of the query text.
        **kwargs: Additional parameters (e.g., num_results).

    Returns:
        A HybridQuery configured for hybrid search.
    """
    num_results = kwargs.get("num_results", self._num_results)

    return HybridQuery(
        text=query_text,
        text_field_name=self._text_field_name,
        vector=embedding,
        vector_field_name=self._vector_field_name,
        text_scorer=self._text_scorer,
        combination_method=self._combination_method,
        linear_alpha=self._linear_alpha,
        rrf_window=self._rrf_window,
        rrf_constant=self._rrf_constant,
        filter_expression=self._filter_expression,
        dtype=self._dtype,
        num_results=num_results,
        return_fields=self._return_fields,
        stopwords=self._stopwords,
        text_weights=self._text_weights,
    )
