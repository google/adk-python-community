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

import sys
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types
import pytest

from google.adk_community.memory.redis_long_term_memory_service import RedisLongTermMemoryService
from google.adk_community.memory.redis_long_term_memory_service import RedisLongTermMemoryServiceConfig


# Create mock classes for agent_memory_client.models
class MockMemoryMessage:

  def __init__(self, role, content):
    self.role = role
    self.content = content


class MockMemoryStrategyConfig:

  def __init__(self, strategy, config=None):
    self.strategy = strategy
    self.config = config


class MockWorkingMemory:

  def __init__(
      self, session_id, namespace, user_id, messages, long_term_memory_strategy
  ):
    self.session_id = session_id
    self.namespace = namespace
    self.user_id = user_id
    self.messages = messages
    self.long_term_memory_strategy = long_term_memory_strategy


class MockRecencyConfig:

  def __init__(
      self,
      recency_boost,
      semantic_weight,
      recency_weight,
      freshness_weight,
      novelty_weight,
      half_life_last_access_days,
      half_life_created_days,
  ):
    self.recency_boost = recency_boost
    self.semantic_weight = semantic_weight
    self.recency_weight = recency_weight
    self.freshness_weight = freshness_weight
    self.novelty_weight = novelty_weight
    self.half_life_last_access_days = half_life_last_access_days
    self.half_life_created_days = half_life_created_days


MOCK_APP_NAME = "test-app"
MOCK_USER_ID = "test-user"
MOCK_SESSION_ID = "session-1"

MOCK_SESSION = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id=MOCK_SESSION_ID,
    last_update_time=1000,
    events=[
        Event(
            id="event-1",
            invocation_id="inv-1",
            author="user",
            timestamp=12345,
            content=types.Content(
                parts=[types.Part(text="Hello, I like Python.")]
            ),
        ),
        Event(
            id="event-2",
            invocation_id="inv-2",
            author="model",
            timestamp=12346,
            content=types.Content(
                parts=[
                    types.Part(text="Python is a great programming language.")
                ]
            ),
        ),
        # Empty event, should be ignored
        Event(
            id="event-3",
            invocation_id="inv-3",
            author="user",
            timestamp=12347,
        ),
    ],
)

MOCK_SESSION_WITH_EMPTY_EVENTS = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id=MOCK_SESSION_ID,
    last_update_time=1000,
)


class TestRedisLongTermMemoryService:
  """Tests for RedisLongTermMemoryService."""

  @pytest.fixture(autouse=True)
  def mock_agent_memory_models(self):
    """Mock agent_memory_client.models module."""
    mock_models = MagicMock()
    mock_models.MemoryMessage = MockMemoryMessage
    mock_models.MemoryStrategyConfig = MockMemoryStrategyConfig
    mock_models.WorkingMemory = MockWorkingMemory
    mock_models.RecencyConfig = MockRecencyConfig

    with patch.dict(sys.modules, {"agent_memory_client.models": mock_models}):
      yield mock_models

  @pytest.fixture
  def mock_memory_client(self):
    """Create a mock MemoryAPIClient."""
    mock_client = MagicMock()
    mock_client.put_working_memory = AsyncMock()
    mock_client.search_long_term_memory = AsyncMock()
    mock_client.close = AsyncMock()
    return mock_client

  @pytest.fixture
  def memory_service(self, mock_memory_client):
    """Create RedisLongTermMemoryService with mocked client."""
    service = RedisLongTermMemoryService()
    # Inject the mock client by setting it in __dict__ to bypass cached_property
    service.__dict__["_client"] = mock_memory_client
    return service

  @pytest.fixture
  def memory_service_with_config(self, mock_memory_client):
    """Create RedisLongTermMemoryService with custom config."""
    config = RedisLongTermMemoryServiceConfig(
        default_namespace="custom_namespace",
        search_top_k=5,
        recency_boost=True,
        extraction_strategy="preferences",
    )
    service = RedisLongTermMemoryService(config=config)
    # Inject the mock client by setting it in __dict__ to bypass cached_property
    service.__dict__["_client"] = mock_memory_client
    return service

  @pytest.mark.asyncio
  async def test_add_session_to_memory_success(
      self, memory_service, mock_memory_client
  ):
    """Test successful addition of session to memory."""
    mock_response = MagicMock()
    mock_response.context_percentage_total_used = 25.0
    mock_memory_client.put_working_memory.return_value = mock_response

    await memory_service.add_session_to_memory(MOCK_SESSION)

    mock_memory_client.put_working_memory.assert_called_once()
    call_args = mock_memory_client.put_working_memory.call_args
    assert call_args.kwargs["session_id"] == MOCK_SESSION_ID
    assert call_args.kwargs["user_id"] == MOCK_USER_ID

    working_memory = call_args.kwargs["memory"]
    assert len(working_memory.messages) == 2
    assert working_memory.messages[0].role == "user"
    assert working_memory.messages[0].content == "Hello, I like Python."
    assert working_memory.messages[1].role == "assistant"
    assert (
        working_memory.messages[1].content
        == "Python is a great programming language."
    )

  @pytest.mark.asyncio
  async def test_add_session_filters_empty_events(
      self, memory_service, mock_memory_client
  ):
    """Test that events without content are filtered out."""
    await memory_service.add_session_to_memory(MOCK_SESSION_WITH_EMPTY_EVENTS)

    mock_memory_client.put_working_memory.assert_not_called()

  @pytest.mark.asyncio
  async def test_add_session_uses_config_namespace(
      self, memory_service_with_config, mock_memory_client
  ):
    """Test that namespace from config is used."""
    mock_response = MagicMock()
    mock_response.context_percentage_total_used = 10.0
    mock_memory_client.put_working_memory.return_value = mock_response

    await memory_service_with_config.add_session_to_memory(MOCK_SESSION)

    call_args = mock_memory_client.put_working_memory.call_args
    working_memory = call_args.kwargs["memory"]
    assert working_memory.namespace == "custom_namespace"

  @pytest.mark.asyncio
  async def test_add_session_uses_extraction_strategy(
      self, memory_service_with_config, mock_memory_client
  ):
    """Test that extraction strategy from config is used."""
    mock_response = MagicMock()
    mock_response.context_percentage_total_used = 10.0
    mock_memory_client.put_working_memory.return_value = mock_response

    await memory_service_with_config.add_session_to_memory(MOCK_SESSION)

    call_args = mock_memory_client.put_working_memory.call_args
    working_memory = call_args.kwargs["memory"]
    assert working_memory.long_term_memory_strategy.strategy == "preferences"

  @pytest.mark.asyncio
  async def test_add_session_error_handling(
      self, memory_service, mock_memory_client
  ):
    """Test error handling during memory addition."""
    mock_memory_client.put_working_memory.side_effect = Exception("API Error")

    # Should not raise exception, just log error
    await memory_service.add_session_to_memory(MOCK_SESSION)

  @pytest.mark.asyncio
  async def test_search_memory_success(
      self, memory_service, mock_memory_client
  ):
    """Test successful memory search."""
    mock_memory = MagicMock()
    mock_memory.text = "Python is a great language"
    mock_results = MagicMock()
    mock_results.memories = [mock_memory]
    mock_memory_client.search_long_term_memory.return_value = mock_results

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="Python programming"
    )

    mock_memory_client.search_long_term_memory.assert_called_once()
    call_args = mock_memory_client.search_long_term_memory.call_args
    assert call_args.kwargs["text"] == "Python programming"
    assert call_args.kwargs["namespace"] == {"eq": MOCK_APP_NAME}
    assert call_args.kwargs["user_id"] == {"eq": MOCK_USER_ID}

    assert len(result.memories) == 1
    assert (
        result.memories[0].content.parts[0].text == "Python is a great language"
    )

  @pytest.mark.asyncio
  async def test_search_memory_with_recency_boost(
      self, memory_service, mock_memory_client
  ):
    """Test that recency config is passed when enabled."""
    mock_results = MagicMock()
    mock_results.memories = []
    mock_memory_client.search_long_term_memory.return_value = mock_results

    await memory_service.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="test query"
    )

    call_args = mock_memory_client.search_long_term_memory.call_args
    recency = call_args.kwargs["recency"]
    assert recency is not None
    assert recency.recency_boost is True
    assert recency.semantic_weight == 0.8
    assert recency.recency_weight == 0.2

  @pytest.mark.asyncio
  async def test_search_memory_without_recency_boost(self, mock_memory_client):
    """Test that recency config is None when disabled."""
    config = RedisLongTermMemoryServiceConfig(recency_boost=False)
    service = RedisLongTermMemoryService(config=config)
    # Inject the mock client by setting it in __dict__ to bypass cached_property
    service.__dict__["_client"] = mock_memory_client

    mock_results = MagicMock()
    mock_results.memories = []
    mock_memory_client.search_long_term_memory.return_value = mock_results

    await service.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="test query"
    )

    call_args = mock_memory_client.search_long_term_memory.call_args
    assert call_args.kwargs["recency"] is None

  @pytest.mark.asyncio
  async def test_search_memory_respects_top_k(
      self, memory_service_with_config, mock_memory_client
  ):
    """Test that config.search_top_k is used."""
    mock_results = MagicMock()
    mock_results.memories = []
    mock_memory_client.search_long_term_memory.return_value = mock_results

    await memory_service_with_config.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="test query"
    )

    call_args = mock_memory_client.search_long_term_memory.call_args
    assert call_args.kwargs["limit"] == 5

  @pytest.mark.asyncio
  async def test_search_memory_error_handling(
      self, memory_service, mock_memory_client
  ):
    """Test graceful error handling during memory search."""
    mock_memory_client.search_long_term_memory.side_effect = Exception(
        "API Error"
    )

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="test query"
    )

    assert len(result.memories) == 0

  @pytest.mark.asyncio
  async def test_close(self, memory_service, mock_memory_client):
    """Test closing the service."""
    await memory_service.close()

    mock_memory_client.close.assert_called_once()
    assert (
        not hasattr(memory_service, "client")
        or "client" not in memory_service.__dict__
    )

  def test_import_error_handling(self):
    """Test that ImportError is raised when agent-memory-client is not installed."""
    service = RedisLongTermMemoryService()

    with patch.dict("sys.modules", {"agent_memory_client": None}):
      with pytest.raises(ImportError, match="agent-memory-client"):
        # Access the client property which will trigger the import
        _ = service._client
