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

"""Configuration classes for Redis search tools.

This module provides Pydantic configuration classes for Redis search tools,
following the pattern established by BigQueryToolConfig in the upstream
google/adk-python repository and OpenMemoryServiceConfig in this repository.
"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

# Type alias for sort specification
SortSpec = Optional[Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]]]


class RedisVectorQueryConfig(BaseModel):
  """Configuration for Redis vector similarity search queries.

  This config groups all query-specific parameters for VectorQuery,
  separating them from tool-level concerns like index and vectorizer.

  Attributes:
      vector_field_name: Name of the vector field in the index.
      num_results: Number of results to return (default: 10).
      dtype: Data type of the vector (default: "float32").
      return_score: Whether to return vector distance scores.
      dialect: RediSearch query dialect version.
      sort_by: Field(s) to order results by.
      in_order: Require query terms in same order as document.
      normalize_vector_distance: Convert distance to 0-1 similarity score.
      hybrid_policy: Filter application policy - "BATCHES" or "ADHOC_BF".
      batch_size: Batch size when hybrid_policy is "BATCHES".
      ef_runtime: HNSW exploration factor at query time.
      epsilon: Range search approximation factor for HNSW/SVS-VAMANA.
      search_window_size: SVS-VAMANA search window size.
      use_search_history: SVS-VAMANA history mode - "OFF", "ON", or "AUTO".
      search_buffer_capacity: SVS-VAMANA 2-level compression tuning.
  """

  model_config = ConfigDict(extra="forbid")

  # Core query parameters
  vector_field_name: str = Field(default="embedding")
  num_results: int = Field(default=10, ge=1)
  dtype: str = Field(default="float32")
  return_score: bool = Field(default=True)
  dialect: int = Field(default=2, ge=1)
  sort_by: SortSpec = Field(default=None)
  in_order: bool = Field(default=False)
  normalize_vector_distance: bool = Field(default=False)

  # Version-dependent parameters (RedisVL 0.13.2+)
  # These are excluded from query kwargs when None for backward compatibility
  hybrid_policy: Optional[str] = Field(default=None)
  batch_size: Optional[int] = Field(default=None, ge=1)
  ef_runtime: Optional[int] = Field(default=None, ge=1)
  epsilon: Optional[float] = Field(default=None, ge=0.0)
  search_window_size: Optional[int] = Field(default=None, ge=1)
  use_search_history: Optional[str] = Field(default=None)
  search_buffer_capacity: Optional[int] = Field(default=None, ge=1)

  def to_query_kwargs(
      self, vector: List[float], filter_expression: Optional[Any] = None
  ) -> dict[str, Any]:
    """Convert config to VectorQuery kwargs, excluding None version-dependent params.

    Args:
        vector: The query vector embedding.
        filter_expression: Optional filter expression to apply.

    Returns:
        Dictionary of kwargs suitable for VectorQuery constructor.
    """
    # Core parameters always included
    kwargs: dict[str, Any] = {
        "vector": vector,
        "vector_field_name": self.vector_field_name,
        "num_results": self.num_results,
        "dtype": self.dtype,
        "return_score": self.return_score,
        "dialect": self.dialect,
        "sort_by": self.sort_by,
        "in_order": self.in_order,
        "normalize_vector_distance": self.normalize_vector_distance,
        "filter_expression": filter_expression,
    }

    # Version-dependent parameters: only include if not None
    version_dependent = {
        "hybrid_policy": self.hybrid_policy,
        "batch_size": self.batch_size,
        "ef_runtime": self.ef_runtime,
        "epsilon": self.epsilon,
        "search_window_size": self.search_window_size,
        "use_search_history": self.use_search_history,
        "search_buffer_capacity": self.search_buffer_capacity,
    }
    for key, value in version_dependent.items():
      if value is not None:
        kwargs[key] = value

    return kwargs


class RedisRangeQueryConfig(BaseModel):
  """Configuration for Redis vector range search queries.

  Range search finds all documents within a specified distance threshold
  from the query vector, rather than returning a fixed number of results.

  Attributes:
      vector_field_name: Name of the vector field in the index.
      distance_threshold: Maximum distance for results (default: 0.2).
      num_results: Maximum number of results to return.
      dtype: Data type of the vector (default: "float32").
      return_score: Whether to return vector distance scores.
      dialect: RediSearch query dialect version.
      sort_by: Field(s) to order results by.
      in_order: Require query terms in same order as document.
      normalize_vector_distance: Convert distance to 0-1 similarity score.
      epsilon: Range search approximation factor for HNSW/SVS-VAMANA.
  """

  model_config = ConfigDict(extra="forbid")

  vector_field_name: str = Field(default="embedding")
  distance_threshold: float = Field(default=0.2, ge=0.0)
  num_results: int = Field(default=10, ge=1)
  dtype: str = Field(default="float32")
  return_score: bool = Field(default=True)
  dialect: int = Field(default=2, ge=1)
  sort_by: SortSpec = Field(default=None)
  in_order: bool = Field(default=False)
  normalize_vector_distance: bool = Field(default=False)

  # Version-dependent parameter
  epsilon: Optional[float] = Field(default=None, ge=0.0)

  def to_query_kwargs(
      self, vector: List[float], filter_expression: Optional[Any] = None
  ) -> dict[str, Any]:
    """Convert config to VectorRangeQuery kwargs.

    Args:
        vector: The query vector embedding.
        filter_expression: Optional filter expression to apply.

    Returns:
        Dictionary of kwargs suitable for VectorRangeQuery constructor.
    """
    kwargs: dict[str, Any] = {
        "vector": vector,
        "vector_field_name": self.vector_field_name,
        "distance_threshold": self.distance_threshold,
        "num_results": self.num_results,
        "dtype": self.dtype,
        "return_score": self.return_score,
        "dialect": self.dialect,
        "sort_by": self.sort_by,
        "in_order": self.in_order,
        "normalize_vector_distance": self.normalize_vector_distance,
        "filter_expression": filter_expression,
    }

    # Version-dependent: only include if not None
    if self.epsilon is not None:
      kwargs["epsilon"] = self.epsilon

    return kwargs


class RedisTextQueryConfig(BaseModel):
  """Configuration for Redis full-text search queries.

  This config groups all query-specific parameters for TextQuery,
  using BM25 scoring for keyword-based search.

  Attributes:
      text_field_name: Name of the text field to search.
      text_scorer: Text scoring algorithm (default: "BM25STD").
      num_results: Number of results to return (default: 10).
      return_score: Whether to return the text score.
      dialect: RediSearch query dialect version.
      sort_by: Field(s) to order results by.
      in_order: Require query terms in same order as document.
      stopwords: Stopwords to remove from query (default: "english").
  """

  model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

  text_field_name: str = Field(default="content")
  text_scorer: str = Field(default="BM25STD")
  num_results: int = Field(default=10, ge=1)
  return_score: bool = Field(default=True)
  dialect: int = Field(default=2, ge=1)
  sort_by: SortSpec = Field(default=None)
  in_order: bool = Field(default=False)
  stopwords: Optional[Union[str, Set[str]]] = Field(default="english")

  def to_query_kwargs(
      self,
      text: str,
      return_fields: Optional[List[str]] = None,
      filter_expression: Optional[Any] = None,
  ) -> dict[str, Any]:
    """Convert config to TextQuery kwargs.

    Args:
        text: The query text for BM25 matching.
        return_fields: Optional list of fields to return.
        filter_expression: Optional filter expression to apply.

    Returns:
        Dictionary of kwargs suitable for TextQuery constructor.
    """
    return {
        "text": text,
        "text_field_name": self.text_field_name,
        "text_scorer": self.text_scorer,
        "num_results": self.num_results,
        "return_score": self.return_score,
        "dialect": self.dialect,
        "sort_by": self.sort_by,
        "in_order": self.in_order,
        "stopwords": self.stopwords,
        "return_fields": return_fields,
        "filter_expression": filter_expression,
    }


class RedisHybridQueryConfig(BaseModel):
  """Configuration for native Redis hybrid search queries.

  Uses Redis's native FT.HYBRID command for server-side hybrid search
  combining semantic vector similarity with keyword-based BM25 text matching.

  Requirements:
      - RedisVL >= 0.13.0
      - Redis >= 8.4.0
      - redis-py >= 7.1.0

  For older Redis/RedisVL versions, use RedisAggregatedHybridQueryConfig instead.

  Example:
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
      tool = RedisHybridSearchTool(index=index, vectorizer=vectorizer, config=config)
      ```

  Attributes:
      text_field_name: Name of the text field for BM25 search.
      vector_field_name: Name of the vector field for similarity search.
      vector_param_name: Name of the parameter substitution for vector blob.
      text_scorer: Text scoring algorithm (default: "BM25STD").
      yield_text_score_as: Field name to yield the text score as.
      vector_search_method: Vector search method - "KNN" or "RANGE".
      knn_ef_runtime: Exploration factor for HNSW when using KNN.
      range_radius: Search radius when using RANGE vector search.
      range_epsilon: Epsilon for RANGE search accuracy.
      yield_vsim_score_as: Field name to yield vector similarity score as.
      combination_method: Score combination method - "RRF" or "LINEAR".
      linear_alpha: Weight of text score when using LINEAR (0.0-1.0).
      rrf_window: Window size for RRF combination.
      rrf_constant: Constant for RRF combination.
      yield_combined_score_as: Field name to yield combined score as.
      num_results: Number of results to return.
      dtype: Data type of the vector.
      stopwords: Stopwords to remove from query.
      text_weights: Optional field weights for text scoring.
  """

  model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

  # Text search parameters
  text_field_name: str = Field(default="content")
  text_scorer: str = Field(default="BM25STD")
  yield_text_score_as: Optional[str] = Field(default=None)
  stopwords: Optional[Union[str, Set[str]]] = Field(default="english")
  text_weights: Optional[Dict[str, float]] = Field(default=None)

  # Vector search parameters
  vector_field_name: str = Field(default="embedding")
  vector_param_name: str = Field(default="vector")
  vector_search_method: Optional[str] = Field(default=None)
  knn_ef_runtime: int = Field(default=10, ge=1)
  range_radius: Optional[float] = Field(default=None)
  range_epsilon: float = Field(default=0.01, ge=0.0)
  yield_vsim_score_as: Optional[str] = Field(default=None)
  dtype: str = Field(default="float32")

  # Score combination parameters
  combination_method: Optional[str] = Field(default=None)
  linear_alpha: float = Field(default=0.3, ge=0.0, le=1.0)
  rrf_window: int = Field(default=20, ge=1)
  rrf_constant: int = Field(default=60, ge=1)
  yield_combined_score_as: Optional[str] = Field(default=None)

  # Result parameters
  num_results: int = Field(default=10, ge=1)

  def to_query_kwargs(
      self,
      text: str,
      vector: List[float],
      return_fields: Optional[List[str]] = None,
      filter_expression: Optional[Any] = None,
  ) -> dict[str, Any]:
    """Convert config to native HybridQuery kwargs.

    Args:
        text: The query text for BM25 matching.
        vector: The query vector embedding.
        return_fields: Optional list of fields to return.
        filter_expression: Optional filter expression to apply.

    Returns:
        Dictionary of kwargs suitable for HybridQuery constructor.
    """
    return {
        "text": text,
        "text_field_name": self.text_field_name,
        "vector": vector,
        "vector_field_name": self.vector_field_name,
        "vector_param_name": self.vector_param_name,
        "text_scorer": self.text_scorer,
        "yield_text_score_as": self.yield_text_score_as,
        "vector_search_method": self.vector_search_method,
        "knn_ef_runtime": self.knn_ef_runtime,
        "range_radius": self.range_radius,
        "range_epsilon": self.range_epsilon,
        "yield_vsim_score_as": self.yield_vsim_score_as,
        "filter_expression": filter_expression,
        "combination_method": self.combination_method,
        "rrf_window": self.rrf_window,
        "rrf_constant": self.rrf_constant,
        "linear_alpha": self.linear_alpha,
        "yield_combined_score_as": self.yield_combined_score_as,
        "dtype": self.dtype,
        "num_results": self.num_results,
        "return_fields": return_fields,
        "stopwords": self.stopwords,
        "text_weights": self.text_weights,
    }


class RedisAggregatedHybridQueryConfig(BaseModel):
  """Configuration for aggregated (client-side) Redis hybrid search queries.

  .. deprecated::
      This config is for older Redis/RedisVL versions. For newer setups
      (RedisVL >= 0.13.0, Redis >= 8.4.0), prefer RedisHybridQueryConfig
      which uses native server-side hybrid search for better performance.

  This config uses AggregateHybridQuery which performs client-side hybrid
  search using FT.AGGREGATE with weighted score combination. It works with
  any Redis version that has RediSearch installed.

  Recommended for:
      - Redis < 8.4.0
      - RedisVL < 0.13.0
      - Environments where native FT.HYBRID is not available

  Example:
      ```python
      from google.adk_community.tools.redis import (
          RedisHybridSearchTool,
          RedisAggregatedHybridQueryConfig,
      )

      config = RedisAggregatedHybridQueryConfig(
          text_field_name="content",
          alpha=0.7,  # 70% text, 30% vector
      )
      tool = RedisHybridSearchTool(index=index, vectorizer=vectorizer, config=config)
      ```

  Attributes:
      text_field_name: Name of the text field for BM25 search.
      vector_field_name: Name of the vector field for similarity search.
      text_scorer: Text scoring algorithm (default: "BM25STD").
      alpha: Weight for text score (default: 0.7). Higher values favor
          text matching over vector similarity. Combined score is:
          alpha * text_score + (1 - alpha) * vector_score
      num_results: Number of results to return.
      dtype: Data type of the vector.
      stopwords: Stopwords to remove from query.
      dialect: RediSearch query dialect version.
      text_weights: Optional field weights for text scoring.
  """

  model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

  text_field_name: str = Field(default="content")
  vector_field_name: str = Field(default="embedding")
  text_scorer: str = Field(default="BM25STD")
  alpha: float = Field(default=0.7, ge=0.0, le=1.0)
  num_results: int = Field(default=10, ge=1)
  dtype: str = Field(default="float32")
  stopwords: Optional[Union[str, Set[str]]] = Field(default="english")
  dialect: int = Field(default=2, ge=1)
  text_weights: Optional[Dict[str, float]] = Field(default=None)

  def to_query_kwargs(
      self,
      text: str,
      vector: List[float],
      return_fields: Optional[List[str]] = None,
      filter_expression: Optional[Any] = None,
  ) -> dict[str, Any]:
    """Convert config to AggregateHybridQuery kwargs.

    Args:
        text: The query text for BM25 matching.
        vector: The query vector embedding.
        return_fields: Optional list of fields to return.
        filter_expression: Optional filter expression to apply.

    Returns:
        Dictionary of kwargs suitable for AggregateHybridQuery constructor.
    """
    return {
        "text": text,
        "text_field_name": self.text_field_name,
        "vector": vector,
        "vector_field_name": self.vector_field_name,
        "text_scorer": self.text_scorer,
        "alpha": self.alpha,
        "dtype": self.dtype,
        "num_results": self.num_results,
        "return_fields": return_fields,
        "stopwords": self.stopwords,
        "dialect": self.dialect,
        "text_weights": self.text_weights,
        "filter_expression": filter_expression,
    }

