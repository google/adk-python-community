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

"""Tests for RedisRangeSearchTool."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

# Skip all tests if redisvl is not installed
pytest.importorskip("redisvl")

from redisvl.index import SearchIndex
from redisvl.query import VectorRangeQuery
from redisvl.utils.vectorize import BaseVectorizer

from google.adk_community.tools.redis import RedisRangeQueryConfig
from google.adk_community.tools.redis import RedisRangeSearchTool


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
      return_value=[{
          "title": "Test Doc",
          "content": "Test content",
          "vector_distance": 0.1,
      }]
  )
  return index


@pytest.fixture
def range_search_tool(mock_index, mock_vectorizer):
  """Create RedisRangeSearchTool instance for testing."""
  config = RedisRangeQueryConfig(
      distance_threshold=0.3,
      num_results=5,
  )
  return RedisRangeSearchTool(
      index=mock_index,
      vectorizer=mock_vectorizer,
      config=config,
      return_fields=["title", "content"],
  )


class TestRedisRangeSearchToolInit:
  """Tests for RedisRangeSearchTool initialization."""

  def test_default_parameters(self, mock_index, mock_vectorizer):
    """Test default parameter values with default config."""
    tool = RedisRangeSearchTool(
        index=mock_index,
        vectorizer=mock_vectorizer,
    )
    # Config defaults
    assert tool._config.vector_field_name == "embedding"
    assert tool._config.distance_threshold == 0.2
    assert tool._config.num_results == 10
    assert tool._config.dtype == "float32"
    assert tool._config.return_score is True
    assert tool._config.dialect == 2
    assert tool._config.in_order is False
    assert tool._config.normalize_vector_distance is False
    assert tool._config.sort_by is None
    assert tool._config.epsilon is None
    # Tool-level defaults
    assert tool._filter_expression is None

  def test_custom_parameters_via_config(self, mock_index, mock_vectorizer):
    """Test custom parameter values via config object."""
    config = RedisRangeQueryConfig(
        vector_field_name="vec",
        distance_threshold=0.5,
        num_results=20,
        dtype="float64",
        return_score=False,
        dialect=3,
        in_order=True,
        normalize_vector_distance=True,
        epsilon=0.01,
    )
    tool = RedisRangeSearchTool(
        index=mock_index,
        vectorizer=mock_vectorizer,
        config=config,
        return_fields=["title", "url"],
    )
    assert tool._config.vector_field_name == "vec"
    assert tool._config.distance_threshold == 0.5
    assert tool._config.num_results == 20
    assert tool._return_fields == ["title", "url"]
    assert tool._config.dtype == "float64"
    assert tool._config.return_score is False
    assert tool._config.dialect == 3
    assert tool._config.in_order is True
    assert tool._config.normalize_vector_distance is True
    assert tool._config.epsilon == 0.01

  def test_custom_name_and_description(self, mock_index, mock_vectorizer):
    """Test custom tool name and description."""
    tool = RedisRangeSearchTool(
        index=mock_index,
        vectorizer=mock_vectorizer,
        name="custom_range",
        description="Custom range search",
    )
    assert tool.name == "custom_range"
    assert tool.description == "Custom range search"


class TestRedisRangeSearchToolBuildQuery:
  """Tests for _build_query method."""

  def test_build_query_basic(self, range_search_tool):
    """Test basic query building."""
    embedding = [0.1] * 384
    query = range_search_tool._build_query("test query", embedding)

    assert isinstance(query, VectorRangeQuery)

  def test_build_query_with_threshold_override(self, range_search_tool):
    """Test query building with distance_threshold override."""
    embedding = [0.1] * 384
    query = range_search_tool._build_query(
        "test query", embedding, distance_threshold=0.8
    )

    assert query._distance_threshold == 0.8


class TestRedisRangeSearchToolDeclaration:
  """Tests for _get_declaration method."""

  def test_get_declaration(self, range_search_tool):
    """Test function declaration generation."""
    declaration = range_search_tool._get_declaration()

    assert declaration.name == "redis_range_search"
    assert "query" in declaration.parameters.properties
    assert "distance_threshold" in declaration.parameters.properties
    assert "query" in declaration.parameters.required
