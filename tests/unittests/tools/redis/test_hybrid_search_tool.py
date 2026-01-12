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
from unittest.mock import patch
import warnings

import pytest

# Skip all tests if redisvl is not installed
pytest.importorskip("redisvl")

from redisvl.index import SearchIndex
from redisvl.utils.vectorize import BaseVectorizer

from google.adk_community.tools.redis import RedisAggregatedHybridQueryConfig
from google.adk_community.tools.redis import RedisHybridQueryConfig
from google.adk_community.tools.redis import RedisHybridSearchTool
from google.adk_community.tools.redis.hybrid_search_tool import _get_redisvl_version
from google.adk_community.tools.redis.hybrid_search_tool import _supports_native_hybrid


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


def _native_hybrid_available():
  """Check if native HybridQuery dependencies are available."""
  try:
    from redis.commands.search.hybrid_query import CombineResultsMethod
    from redis.commands.search.hybrid_query import HybridPostProcessingConfig

    return True
  except (ImportError, ModuleNotFoundError):
    return False


class TestVersionDetection:
  """Tests for version detection functions."""

  def test_get_redisvl_version(self):
    """Test version string retrieval."""
    version = _get_redisvl_version()
    assert isinstance(version, str)
    # Should be a valid version string like "0.13.0" or "0.0.0"
    assert len(version.split(".")) >= 2

  def test_supports_native_hybrid(self):
    """Test native hybrid support detection."""
    result = _supports_native_hybrid()
    assert isinstance(result, bool)


class TestRedisHybridQueryConfig:
  """Tests for RedisHybridQueryConfig (native mode)."""

  def test_default_values(self):
    """Test default config values."""
    config = RedisHybridQueryConfig()
    assert config.text_field_name == "content"
    assert config.vector_field_name == "embedding"
    assert config.text_scorer == "BM25STD"
    assert config.combination_method is None
    assert config.linear_alpha == 0.3
    assert config.rrf_window == 20
    assert config.rrf_constant == 60
    assert config.num_results == 10
    assert config.dtype == "float32"
    assert config.stopwords == "english"

  def test_to_query_kwargs(self):
    """Test conversion to query kwargs."""
    config = RedisHybridQueryConfig(
        text_field_name="content",
        combination_method="LINEAR",
        linear_alpha=0.7,
    )
    kwargs = config.to_query_kwargs(
        text="test query",
        vector=[0.1] * 384,
        return_fields=["title"],
    )
    assert kwargs["text"] == "test query"
    assert kwargs["text_field_name"] == "content"
    assert kwargs["combination_method"] == "LINEAR"
    assert kwargs["linear_alpha"] == 0.7
    assert kwargs["return_fields"] == ["title"]


class TestRedisAggregatedHybridQueryConfig:
  """Tests for RedisAggregatedHybridQueryConfig (legacy mode)."""

  def test_default_values(self):
    """Test default config values."""
    config = RedisAggregatedHybridQueryConfig()
    assert config.text_field_name == "content"
    assert config.vector_field_name == "embedding"
    assert config.text_scorer == "BM25STD"
    assert config.alpha == 0.7
    assert config.num_results == 10
    assert config.dtype == "float32"
    assert config.dialect == 2

  def test_to_query_kwargs(self):
    """Test conversion to query kwargs."""
    config = RedisAggregatedHybridQueryConfig(
        text_field_name="content",
        alpha=0.8,
    )
    kwargs = config.to_query_kwargs(
        text="test query",
        vector=[0.1] * 384,
        return_fields=["title"],
    )
    assert kwargs["text"] == "test query"
    assert kwargs["text_field_name"] == "content"
    assert kwargs["alpha"] == 0.8
    assert kwargs["return_fields"] == ["title"]
    assert "linear_alpha" not in kwargs
    assert "combination_method" not in kwargs


class TestRedisHybridSearchToolInit:
  """Tests for RedisHybridSearchTool initialization."""

  def test_auto_detect_config_native(self, mock_index, mock_vectorizer):
    """Test auto-detection uses native config when supported."""
    with patch(
        "google.adk_community.tools.redis.hybrid_search_tool._supports_native_hybrid",
        return_value=True,
    ):
      tool = RedisHybridSearchTool(
          index=mock_index,
          vectorizer=mock_vectorizer,
      )
      assert isinstance(tool._config, RedisHybridQueryConfig)
      assert tool._use_native is True

  def test_auto_detect_config_aggregate(self, mock_index, mock_vectorizer):
    """Test auto-detection uses aggregate config when native not supported."""
    with patch(
        "google.adk_community.tools.redis.hybrid_search_tool._supports_native_hybrid",
        return_value=False,
    ):
      tool = RedisHybridSearchTool(
          index=mock_index,
          vectorizer=mock_vectorizer,
      )
      assert isinstance(tool._config, RedisAggregatedHybridQueryConfig)
      assert tool._use_native is False

  def test_native_config_on_old_version_raises(
      self, mock_index, mock_vectorizer
  ):
    """Test that using native config on old version raises ValueError."""
    with patch(
        "google.adk_community.tools.redis.hybrid_search_tool._supports_native_hybrid",
        return_value=False,
    ):
      with pytest.raises(ValueError, match="RedisHybridQueryConfig requires"):
        RedisHybridSearchTool(
            index=mock_index,
            vectorizer=mock_vectorizer,
            config=RedisHybridQueryConfig(),
        )

  def test_aggregate_config_on_new_version_warns(
      self, mock_index, mock_vectorizer
  ):
    """Test that using aggregate config on new version emits deprecation warning."""
    with patch(
        "google.adk_community.tools.redis.hybrid_search_tool._supports_native_hybrid",
        return_value=True,
    ):
      with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tool = RedisHybridSearchTool(
            index=mock_index,
            vectorizer=mock_vectorizer,
            config=RedisAggregatedHybridQueryConfig(),
        )
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()
        assert tool._use_native is False

  def test_custom_name_and_description(self, mock_index, mock_vectorizer):
    """Test custom tool name and description."""
    with patch(
        "google.adk_community.tools.redis.hybrid_search_tool._supports_native_hybrid",
        return_value=True,
    ):
      tool = RedisHybridSearchTool(
          index=mock_index,
          vectorizer=mock_vectorizer,
          name="custom_hybrid",
          description="Custom hybrid search",
      )
      assert tool.name == "custom_hybrid"
      assert tool.description == "Custom hybrid search"


class TestRedisHybridSearchToolBuildQuery:
  """Tests for _build_query method."""

  @pytest.mark.skipif(
      not _native_hybrid_available(),
      reason="HybridQuery requires redis-py>=7.1.0 and Redis>=8.4.0",
  )
  def test_build_query_native(self, mock_index, mock_vectorizer):
    """Test query building with native HybridQuery."""
    from redisvl.query import HybridQuery

    with patch(
        "google.adk_community.tools.redis.hybrid_search_tool._supports_native_hybrid",
        return_value=True,
    ):
      tool = RedisHybridSearchTool(
          index=mock_index,
          vectorizer=mock_vectorizer,
          config=RedisHybridQueryConfig(num_results=5),
      )
      embedding = [0.1] * 384
      query = tool._build_query("test query", embedding)
      assert isinstance(query, HybridQuery)

  def test_build_query_aggregate(self, mock_index, mock_vectorizer):
    """Test query building with AggregateHybridQuery."""
    from redisvl.query import AggregateHybridQuery

    with patch(
        "google.adk_community.tools.redis.hybrid_search_tool._supports_native_hybrid",
        return_value=False,
    ):
      tool = RedisHybridSearchTool(
          index=mock_index,
          vectorizer=mock_vectorizer,
          config=RedisAggregatedHybridQueryConfig(num_results=5),
      )
      embedding = [0.1] * 384
      query = tool._build_query("test query", embedding)
      assert isinstance(query, AggregateHybridQuery)


class TestRedisHybridSearchToolDeclaration:
  """Tests for _get_declaration method."""

  def test_get_declaration(self, mock_index, mock_vectorizer):
    """Test function declaration generation."""
    with patch(
        "google.adk_community.tools.redis.hybrid_search_tool._supports_native_hybrid",
        return_value=True,
    ):
      tool = RedisHybridSearchTool(
          index=mock_index,
          vectorizer=mock_vectorizer,
          config=RedisHybridQueryConfig(num_results=5),
      )
      declaration = tool._get_declaration()

      assert declaration.name == "redis_hybrid_search"
      assert "query" in declaration.parameters.properties
      assert "num_results" in declaration.parameters.properties
      assert "query" in declaration.parameters.required
