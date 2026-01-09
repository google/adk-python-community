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

"""Tests for RedisHybridSearchTool."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

# Skip all tests if redisvl is not installed
pytest.importorskip("redisvl")

from redisvl.index import SearchIndex
from redisvl.query import HybridQuery
from redisvl.utils.vectorize import BaseVectorizer

from google.adk_community.tools.redis import RedisHybridQueryConfig
from google.adk_community.tools.redis import RedisHybridSearchTool


@pytest.fixture
def mock_vectorizer():
  """Mock RedisVL vectorizer."""
  vectorizer = MagicMock(spec=BaseVectorizer)
  vectorizer.embed = MagicMock(return_value=[0.1] * 384)
  vectorizer.aembed = AsyncMock(return_value=[0.1] * 384)
  return vectorizer


@pytest.fixture
def mock_index():
  """Mock RedisVL SearchIndex."""
  index = MagicMock(spec=SearchIndex)
  index.query = MagicMock(
      return_value=[
          {"title": "Test Doc", "content": "Test content", "score": 0.9}
      ]
  )
  return index


@pytest.fixture
def hybrid_search_tool(mock_index, mock_vectorizer):
  """Create RedisHybridSearchTool instance for testing."""
  config = RedisHybridQueryConfig(
      text_field_name="content",
      num_results=5,
  )
  return RedisHybridSearchTool(
      index=mock_index,
      vectorizer=mock_vectorizer,
      config=config,
      return_fields=["title", "content"],
  )


class TestRedisHybridSearchToolInit:
  """Tests for RedisHybridSearchTool initialization."""

  def test_default_parameters(self, mock_index, mock_vectorizer):
    """Test default parameter values with default config."""
    tool = RedisHybridSearchTool(
        index=mock_index,
        vectorizer=mock_vectorizer,
    )
    # Config defaults
    assert tool._config.text_field_name == "content"
    assert tool._config.vector_field_name == "embedding"
    assert tool._config.text_scorer == "BM25STD"
    assert tool._config.combination_method is None
    assert tool._config.linear_alpha == 0.3
    assert tool._config.rrf_window == 20
    assert tool._config.rrf_constant == 60
    assert tool._config.num_results == 10
    assert tool._config.dtype == "float32"
    assert tool._config.stopwords == "english"
    # Tool-level defaults
    assert tool._filter_expression is None

  def test_custom_parameters_via_config(self, mock_index, mock_vectorizer):
    """Test custom parameter values via config object."""
    config = RedisHybridQueryConfig(
        text_field_name="description",
        vector_field_name="vec",
        text_scorer="TFIDF",
        combination_method="LINEAR",
        linear_alpha=0.7,
        rrf_window=30,
        rrf_constant=80,
        num_results=20,
        dtype="float64",
        stopwords={"the", "a", "an"},
    )
    tool = RedisHybridSearchTool(
        index=mock_index,
        vectorizer=mock_vectorizer,
        config=config,
        return_fields=["title", "url"],
    )
    assert tool._config.text_field_name == "description"
    assert tool._config.vector_field_name == "vec"
    assert tool._config.text_scorer == "TFIDF"
    assert tool._config.combination_method == "LINEAR"
    assert tool._config.linear_alpha == 0.7
    assert tool._config.rrf_window == 30
    assert tool._config.rrf_constant == 80
    assert tool._config.num_results == 20
    assert tool._return_fields == ["title", "url"]
    assert tool._config.dtype == "float64"
    assert tool._config.stopwords == {"the", "a", "an"}

  def test_custom_name_and_description(self, mock_index, mock_vectorizer):
    """Test custom tool name and description."""
    tool = RedisHybridSearchTool(
        index=mock_index,
        vectorizer=mock_vectorizer,
        name="custom_hybrid",
        description="Custom hybrid search",
    )
    assert tool.name == "custom_hybrid"
    assert tool.description == "Custom hybrid search"


def _hybrid_query_available():
  """Check if HybridQuery dependencies are available."""
  try:
    from redis.commands.search.hybrid_query import CombineResultsMethod
    from redis.commands.search.hybrid_query import HybridPostProcessingConfig

    return True
  except (ImportError, ModuleNotFoundError):
    return False


class TestRedisHybridSearchToolBuildQuery:
  """Tests for _build_query method."""

  @pytest.mark.skipif(
      not _hybrid_query_available(),
      reason="HybridQuery requires redis-py>=7.1.0 and Redis>=8.4.0",
  )
  def test_build_query_basic(self, hybrid_search_tool):
    """Test basic query building."""
    embedding = [0.1] * 384
    query = hybrid_search_tool._build_query("test query", embedding)

    assert isinstance(query, HybridQuery)

  @pytest.mark.skipif(
      not _hybrid_query_available(),
      reason="HybridQuery requires redis-py>=7.1.0 and Redis>=8.4.0",
  )
  def test_build_query_with_num_results_override(self, hybrid_search_tool):
    """Test query building with num_results override."""
    embedding = [0.1] * 384
    query = hybrid_search_tool._build_query(
        "test query", embedding, num_results=15
    )

    assert query._num_results == 15


class TestRedisHybridSearchToolDeclaration:
  """Tests for _get_declaration method."""

  def test_get_declaration(self, hybrid_search_tool):
    """Test function declaration generation."""
    declaration = hybrid_search_tool._get_declaration()

    assert declaration.name == "redis_hybrid_search"
    assert "query" in declaration.parameters.properties
    assert "num_results" in declaration.parameters.properties
    assert "query" in declaration.parameters.required
