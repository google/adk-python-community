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
  return RedisRangeSearchTool(
      index=mock_index,
      vectorizer=mock_vectorizer,
      distance_threshold=0.3,
      num_results=5,
      return_fields=["title", "content"],
  )


class TestRedisRangeSearchToolInit:
  """Tests for RedisRangeSearchTool initialization."""

  def test_default_parameters(self, mock_index, mock_vectorizer):
    """Test default parameter values."""
    tool = RedisRangeSearchTool(
        index=mock_index,
        vectorizer=mock_vectorizer,
    )
    assert tool._vector_field_name == "embedding"
    assert tool._distance_threshold == 0.2
    assert tool._num_results == 10
    assert tool._dtype == "float32"
    assert tool._return_score is True
    assert tool._dialect == 2
    assert tool._in_order is False
    assert tool._normalize_vector_distance is False
    assert tool._filter_expression is None
    assert tool._sort_by is None
    assert tool._epsilon is None

  def test_custom_parameters(self, mock_index, mock_vectorizer):
    """Test custom parameter values."""
    tool = RedisRangeSearchTool(
        index=mock_index,
        vectorizer=mock_vectorizer,
        vector_field_name="vec",
        distance_threshold=0.5,
        num_results=20,
        return_fields=["title", "url"],
        dtype="float64",
        return_score=False,
        dialect=3,
        in_order=True,
        normalize_vector_distance=True,
        epsilon=0.01,
    )
    assert tool._vector_field_name == "vec"
    assert tool._distance_threshold == 0.5
    assert tool._num_results == 20
    assert tool._return_fields == ["title", "url"]
    assert tool._dtype == "float64"
    assert tool._return_score is False
    assert tool._dialect == 3
    assert tool._in_order is True
    assert tool._normalize_vector_distance is True
    assert tool._epsilon == 0.01

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
