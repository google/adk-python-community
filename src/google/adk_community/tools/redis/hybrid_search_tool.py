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

from .base_search_tool import VectorizedSearchTool


class RedisHybridSearchTool(VectorizedSearchTool):
  """Hybrid search tool combining vector similarity and BM25 text search.

  This tool performs a hybrid search that combines semantic vector similarity
  with keyword-based BM25 text matching using Redis's native FT.HYBRID command.
  This is useful when you want to leverage both the semantic understanding of
  embeddings and the precision of keyword matching.

  Requirements:
      - Redis >= 8.4.0 (for native FT.HYBRID command support)
      - redis-py >= 7.1.0

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

      agent = Agent(model="gemini-2.5-flash", tools=[tool])
      ```
  """

  def __init__(
      self,
      *,
      index: Union[SearchIndex, AsyncSearchIndex],
      vectorizer: BaseVectorizer,
      text_field_name: str = "content",
      vector_field_name: str = "embedding",
      vector_param_name: str = "vector",
      text_scorer: str = "BM25STD",
      yield_text_score_as: Optional[str] = None,
      vector_search_method: Optional[str] = None,
      knn_ef_runtime: int = 10,
      range_radius: Optional[float] = None,
      range_epsilon: float = 0.01,
      yield_vsim_score_as: Optional[str] = None,
      combination_method: Optional[str] = None,
      linear_alpha: float = 0.3,
      rrf_window: int = 20,
      rrf_constant: int = 60,
      yield_combined_score_as: Optional[str] = None,
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
        vector_param_name: Name of the parameter substitution for vector blob.
        text_scorer: The text scoring algorithm (default: "BM25STD").
        yield_text_score_as: Field name to yield the text score as.
        vector_search_method: Vector search method - "KNN" or "RANGE".
        knn_ef_runtime: Exploration factor for HNSW when using KNN (default: 10).
        range_radius: Search radius when using RANGE vector search.
        range_epsilon: Epsilon for RANGE search accuracy (default: 0.01).
        yield_vsim_score_as: Field name to yield the vector similarity score as.
        combination_method: Score combination method - "RRF" or "LINEAR".
        linear_alpha: Weight of text score when using LINEAR (default: 0.3).
        rrf_window: Window size for RRF combination (default: 20).
        rrf_constant: Constant for RRF combination (default: 60).
        yield_combined_score_as: Field name to yield the combined score as.
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
    self._vector_param_name = vector_param_name
    self._text_scorer = text_scorer
    self._yield_text_score_as = yield_text_score_as
    self._vector_search_method = vector_search_method
    self._knn_ef_runtime = knn_ef_runtime
    self._range_radius = range_radius
    self._range_epsilon = range_epsilon
    self._yield_vsim_score_as = yield_vsim_score_as
    self._combination_method = combination_method
    self._linear_alpha = linear_alpha
    self._rrf_window = rrf_window
    self._rrf_constant = rrf_constant
    self._yield_combined_score_as = yield_combined_score_as
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
        vector_param_name=self._vector_param_name,
        text_scorer=self._text_scorer,
        yield_text_score_as=self._yield_text_score_as,
        vector_search_method=self._vector_search_method,
        knn_ef_runtime=self._knn_ef_runtime,
        range_radius=self._range_radius,
        range_epsilon=self._range_epsilon,
        yield_vsim_score_as=self._yield_vsim_score_as,
        filter_expression=self._filter_expression,
        combination_method=self._combination_method,
        rrf_window=self._rrf_window,
        rrf_constant=self._rrf_constant,
        linear_alpha=self._linear_alpha,
        yield_combined_score_as=self._yield_combined_score_as,
        dtype=self._dtype,
        num_results=num_results,
        return_fields=self._return_fields,
        stopwords=self._stopwords,
        text_weights=self._text_weights,
    )
