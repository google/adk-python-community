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

"""Tests for RedisTextSearchTool."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# Skip all tests if redisvl is not installed
pytest.importorskip("redisvl")

from redisvl.index import SearchIndex

from google.adk_community.tools.redis import RedisTextQueryConfig
from google.adk_community.tools.redis import RedisTextSearchTool


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
def text_search_tool(mock_index):
  """Create RedisTextSearchTool instance for testing."""
  config = RedisTextQueryConfig(
      text_field_name="content",
      num_results=5,
      stopwords=None,  # Avoid nltk dependency in tests
  )
  return RedisTextSearchTool(
      index=mock_index,
      config=config,
      return_fields=["title", "content"],
  )


class TestRedisTextSearchToolInit:
  """Tests for RedisTextSearchTool initialization."""

  def test_default_parameters(self, mock_index):
    """Test default parameter values via config."""
    tool = RedisTextSearchTool(index=mock_index)
    assert tool._config.text_field_name == "content"
    assert tool._config.text_scorer == "BM25STD"
    assert tool._config.num_results == 10
    assert tool._config.return_score is True
    assert tool._config.dialect == 2
    assert tool._config.in_order is False
    assert tool._config.stopwords == "english"
    assert tool._filter_expression is None
    assert tool._config.sort_by is None
    assert tool._return_fields is None

  def test_custom_parameters_via_config(self, mock_index):
    """Test custom parameter values via config object."""
    config = RedisTextQueryConfig(
        text_field_name="description",
        text_scorer="TFIDF",
        num_results=20,
        return_score=False,
        dialect=3,
        in_order=True,
        stopwords={"the", "a", "an"},
    )
    tool = RedisTextSearchTool(
        index=mock_index,
        config=config,
        return_fields=["title", "url"],
    )
    assert tool._config.text_field_name == "description"
    assert tool._config.text_scorer == "TFIDF"
    assert tool._config.num_results == 20
    assert tool._return_fields == ["title", "url"]
    assert tool._config.return_score is False
    assert tool._config.dialect == 3
    assert tool._config.in_order is True
    assert tool._config.stopwords == {"the", "a", "an"}

  def test_custom_name_and_description(self, mock_index):
    """Test custom tool name and description."""
    tool = RedisTextSearchTool(
        index=mock_index,
        name="custom_text",
        description="Custom text search",
    )
    assert tool.name == "custom_text"
    assert tool.description == "Custom text search"

  def test_no_vectorizer_required(self, mock_index):
    """Test that TextSearchTool does not require a vectorizer."""
    # This should work without any vectorizer
    tool = RedisTextSearchTool(index=mock_index)
    assert not hasattr(tool, "_vectorizer")


class TestRedisTextSearchToolDeclaration:
  """Tests for _get_declaration method."""

  def test_get_declaration(self, text_search_tool):
    """Test function declaration generation."""
    declaration = text_search_tool._get_declaration()

    assert declaration.name == "redis_text_search"
    assert "query" in declaration.parameters.properties
    assert "num_results" in declaration.parameters.properties
    assert "query" in declaration.parameters.required


class TestRedisTextSearchToolRunAsync:
  """Tests for run_async method."""

  @pytest.mark.asyncio
  @patch("google.adk_community.tools.redis.text_search_tool.asyncio.to_thread")
  async def test_run_async_success(
      self, mock_to_thread, text_search_tool, mock_index
  ):
    """Test successful search execution."""
    mock_to_thread.return_value = [
        {"title": "Test Doc", "content": "Test content", "score": 0.9}
    ]
    mock_context = MagicMock()
    result = await text_search_tool.run_async(
        args={"query": "test query"},
        tool_context=mock_context,
    )

    assert result["status"] == "success"
    assert result["count"] == 1
    assert len(result["results"]) == 1
    mock_to_thread.assert_called_once()

  @pytest.mark.asyncio
  async def test_run_async_empty_query(self, text_search_tool):
    """Test error handling for empty query."""
    mock_context = MagicMock()
    result = await text_search_tool.run_async(
        args={"query": ""},
        tool_context=mock_context,
    )

    assert result["status"] == "error"
    assert "required" in result["error"].lower()

  @pytest.mark.asyncio
  @patch("google.adk_community.tools.redis.text_search_tool.asyncio.to_thread")
  async def test_run_async_with_num_results(
      self, mock_to_thread, text_search_tool, mock_index
  ):
    """Test search with custom num_results."""
    mock_to_thread.return_value = [
        {"title": "Test Doc", "content": "Test content", "score": 0.9}
    ]
    mock_context = MagicMock()
    await text_search_tool.run_async(
        args={"query": "test", "num_results": 15},
        tool_context=mock_context,
    )

    # Verify to_thread was called
    mock_to_thread.assert_called_once()
