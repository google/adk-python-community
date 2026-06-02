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

import json
from unittest.mock import AsyncMock, MagicMock

from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.adk_community.memory.valkey_memory_service import (
    ValkeyMemoryService,
    ValkeyMemoryServiceConfig,
)
from google.genai import types
import pytest

MOCK_APP_NAME = 'test-app'
MOCK_USER_ID = 'test-user'
MOCK_SESSION_ID = 'session-1'

MOCK_SESSION = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id=MOCK_SESSION_ID,
    last_update_time=1000,
    events=[
        Event(
            id='event-1',
            invocation_id='inv-1',
            author='user',
            timestamp=12345,
            content=types.Content(
                parts=[types.Part(text='Hello, I like Python.')]
            ),
        ),
        Event(
            id='event-2',
            invocation_id='inv-2',
            author='model',
            timestamp=12346,
            content=types.Content(
                parts=[
                    types.Part(text='Python is a great programming language.')
                ]
            ),
        ),
        # Empty event, should be ignored
        Event(
            id='event-3',
            invocation_id='inv-3',
            author='user',
            timestamp=12347,
        ),
        # Function call event, should be ignored
        Event(
            id='event-4',
            invocation_id='inv-4',
            author='agent',
            timestamp=12348,
            content=types.Content(
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name='test_function')
                    )
                ]
            ),
        ),
    ],
)

MOCK_SESSION_WITH_EMPTY_EVENTS = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id=MOCK_SESSION_ID,
    last_update_time=1000,
)


@pytest.fixture
def mock_valkey_client():
  """Mock valkey-glide client for testing."""
  client = AsyncMock()
  client.rpush = AsyncMock(return_value=1)
  client.lrange = AsyncMock(return_value=[])
  client.expire = AsyncMock(return_value=True)
  return client


@pytest.fixture
def memory_service(mock_valkey_client):
  """Create ValkeyMemoryService instance for testing."""
  return ValkeyMemoryService(client=mock_valkey_client)


@pytest.fixture
def memory_service_with_config(mock_valkey_client):
  """Create ValkeyMemoryService with custom config."""
  config = ValkeyMemoryServiceConfig(
      search_top_k=5,
      key_prefix="custom:mem",
      ttl_seconds=3600,
  )
  return ValkeyMemoryService(client=mock_valkey_client, config=config)


class TestValkeyMemoryServiceConfig:
  """Tests for ValkeyMemoryServiceConfig."""

  def test_default_config(self):
    """Test default configuration values."""
    config = ValkeyMemoryServiceConfig()
    assert config.search_top_k == 10
    assert config.key_prefix == "adk:memory"
    assert config.ttl_seconds is None

  def test_custom_config(self):
    """Test custom configuration values."""
    config = ValkeyMemoryServiceConfig(
        search_top_k=20,
        key_prefix="my:prefix",
        ttl_seconds=7200,
    )
    assert config.search_top_k == 20
    assert config.key_prefix == "my:prefix"
    assert config.ttl_seconds == 7200

  def test_config_validation_search_top_k(self):
    """Test search_top_k validation."""
    with pytest.raises(Exception):
      ValkeyMemoryServiceConfig(search_top_k=0)

    with pytest.raises(Exception):
      ValkeyMemoryServiceConfig(search_top_k=101)


class TestValkeyMemoryServiceInit:
  """Tests for ValkeyMemoryService initialization."""

  def test_client_required(self):
    """Test that client is required."""
    with pytest.raises(ValueError, match="client is required"):
      ValkeyMemoryService(client=None)

  def test_init_with_client(self, mock_valkey_client):
    """Test initialization with a valid client."""
    service = ValkeyMemoryService(client=mock_valkey_client)
    assert service._client is mock_valkey_client
    assert service._config.search_top_k == 10

  def test_init_with_config(self, mock_valkey_client):
    """Test initialization with custom config."""
    config = ValkeyMemoryServiceConfig(search_top_k=5)
    service = ValkeyMemoryService(
        client=mock_valkey_client, config=config
    )
    assert service._config.search_top_k == 5


class TestValkeyMemoryServiceAddSession:
  """Tests for add_session_to_memory."""

  @pytest.mark.asyncio
  async def test_add_session_success(
      self, memory_service, mock_valkey_client
  ):
    """Test successful addition of session memories."""
    await memory_service.add_session_to_memory(MOCK_SESSION)

    # Should make 2 rpush calls (one per valid event with text)
    assert mock_valkey_client.rpush.call_count == 2

    # Check first call
    first_call = mock_valkey_client.rpush.call_args_list[0]
    key = first_call[0][0]
    assert key == "adk:memory:test-app:test-user:entries"

    value = first_call[0][1][0]
    data = json.loads(value)
    assert data["content"] == "Hello, I like Python."
    assert data["author"] == "user"
    assert data["session_id"] == MOCK_SESSION_ID
    assert data["event_id"] == "event-1"

    # Check second call
    second_call = mock_valkey_client.rpush.call_args_list[1]
    value = second_call[0][1][0]
    data = json.loads(value)
    assert data["content"] == "Python is a great programming language."
    assert data["author"] == "model"

  @pytest.mark.asyncio
  async def test_add_session_filters_empty_events(
      self, memory_service, mock_valkey_client
  ):
    """Test that events without text content are filtered out."""
    await memory_service.add_session_to_memory(
        MOCK_SESSION_WITH_EMPTY_EVENTS
    )
    assert mock_valkey_client.rpush.call_count == 0

  @pytest.mark.asyncio
  async def test_add_session_with_ttl(
      self, memory_service_with_config, mock_valkey_client
  ):
    """Test that TTL is set when configured."""
    await memory_service_with_config.add_session_to_memory(MOCK_SESSION)

    mock_valkey_client.expire.assert_called_once_with(
        "custom:mem:test-app:test-user:entries", 3600
    )

  @pytest.mark.asyncio
  async def test_add_session_no_ttl_by_default(
      self, memory_service, mock_valkey_client
  ):
    """Test that no TTL is set when not configured."""
    await memory_service.add_session_to_memory(MOCK_SESSION)
    mock_valkey_client.expire.assert_not_called()

  @pytest.mark.asyncio
  async def test_add_session_error_handling(
      self, memory_service, mock_valkey_client
  ):
    """Test error handling during memory addition."""
    mock_valkey_client.rpush.side_effect = Exception("Connection error")

    # Should not raise exception, just log error
    await memory_service.add_session_to_memory(MOCK_SESSION)
    assert mock_valkey_client.rpush.call_count == 2

  @pytest.mark.asyncio
  async def test_add_session_custom_key_prefix(
      self, memory_service_with_config, mock_valkey_client
  ):
    """Test that custom key prefix is used."""
    await memory_service_with_config.add_session_to_memory(MOCK_SESSION)

    first_call = mock_valkey_client.rpush.call_args_list[0]
    key = first_call[0][0]
    assert key == "custom:mem:test-app:test-user:entries"


class TestValkeyMemoryServiceSearch:
  """Tests for search_memory."""

  @pytest.mark.asyncio
  async def test_search_memory_success(
      self, memory_service, mock_valkey_client
  ):
    """Test successful memory search."""
    stored_memories = [
        json.dumps({
            "content": "I love Python programming",
            "author": "user",
            "timestamp": 12345,
        }).encode(),
        json.dumps({
            "content": "Java is also popular",
            "author": "model",
            "timestamp": 12346,
        }).encode(),
        json.dumps({
            "content": "Python has great libraries",
            "author": "user",
            "timestamp": 12347,
        }).encode(),
    ]
    mock_valkey_client.lrange = AsyncMock(return_value=stored_memories)

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query="Python",
    )

    assert len(result.memories) == 2
    assert result.memories[0].content.parts[0].text == (
        "I love Python programming"
    )
    assert result.memories[0].author == "user"
    assert result.memories[1].content.parts[0].text == (
        "Python has great libraries"
    )

  @pytest.mark.asyncio
  async def test_search_memory_no_results(
      self, memory_service, mock_valkey_client
  ):
    """Test search with no matching memories."""
    stored_memories = [
        json.dumps({
            "content": "Hello world",
            "author": "user",
            "timestamp": 12345,
        }).encode(),
    ]
    mock_valkey_client.lrange = AsyncMock(return_value=stored_memories)

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query="Rust language",
    )

    assert len(result.memories) == 0

  @pytest.mark.asyncio
  async def test_search_memory_empty_store(
      self, memory_service, mock_valkey_client
  ):
    """Test search when no memories are stored."""
    mock_valkey_client.lrange = AsyncMock(return_value=[])

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query="anything",
    )

    assert len(result.memories) == 0

  @pytest.mark.asyncio
  async def test_search_memory_none_response(
      self, memory_service, mock_valkey_client
  ):
    """Test search when lrange returns None."""
    mock_valkey_client.lrange = AsyncMock(return_value=None)

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query="anything",
    )

    assert len(result.memories) == 0

  @pytest.mark.asyncio
  async def test_search_memory_respects_top_k(
      self, memory_service_with_config, mock_valkey_client
  ):
    """Test that search respects search_top_k config."""
    # Create more memories than top_k (5)
    stored_memories = [
        json.dumps({
            "content": f"Python tip number {i}",
            "author": "user",
            "timestamp": 12345 + i,
        }).encode()
        for i in range(10)
    ]
    mock_valkey_client.lrange = AsyncMock(return_value=stored_memories)

    result = await memory_service_with_config.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query="Python",
    )

    # Should return at most 5 (search_top_k)
    assert len(result.memories) == 5

  @pytest.mark.asyncio
  async def test_search_memory_case_insensitive(
      self, memory_service, mock_valkey_client
  ):
    """Test that search is case-insensitive."""
    stored_memories = [
        json.dumps({
            "content": "PYTHON is great",
            "author": "user",
            "timestamp": 12345,
        }).encode(),
    ]
    mock_valkey_client.lrange = AsyncMock(return_value=stored_memories)

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query="python",
    )

    assert len(result.memories) == 1

  @pytest.mark.asyncio
  async def test_search_memory_error_handling(
      self, memory_service, mock_valkey_client
  ):
    """Test graceful error handling during search."""
    mock_valkey_client.lrange.side_effect = Exception("Connection error")

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query="test",
    )

    assert len(result.memories) == 0

  @pytest.mark.asyncio
  async def test_search_memory_handles_corrupt_entries(
      self, memory_service, mock_valkey_client
  ):
    """Test that corrupt entries are skipped gracefully."""
    stored_memories = [
        b"not valid json",
        json.dumps({
            "content": "Valid Python memory",
            "author": "user",
            "timestamp": 12345,
        }).encode(),
    ]
    mock_valkey_client.lrange = AsyncMock(return_value=stored_memories)

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query="Python",
    )

    assert len(result.memories) == 1
    assert result.memories[0].content.parts[0].text == (
        "Valid Python memory"
    )

  @pytest.mark.asyncio
  async def test_search_memory_multi_term_query(
      self, memory_service, mock_valkey_client
  ):
    """Test search with multiple terms (any term matches)."""
    stored_memories = [
        json.dumps({
            "content": "I love Python",
            "author": "user",
            "timestamp": 12345,
        }).encode(),
        json.dumps({
            "content": "Java is enterprise",
            "author": "model",
            "timestamp": 12346,
        }).encode(),
        json.dumps({
            "content": "Rust is fast",
            "author": "user",
            "timestamp": 12347,
        }).encode(),
    ]
    mock_valkey_client.lrange = AsyncMock(return_value=stored_memories)

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query="Python Java",
    )

    # Both "Python" and "Java" memories should match
    assert len(result.memories) == 2

  @pytest.mark.asyncio
  async def test_search_memory_correct_key(
      self, memory_service, mock_valkey_client
  ):
    """Test that the correct Valkey key is queried."""
    mock_valkey_client.lrange = AsyncMock(return_value=[])

    await memory_service.search_memory(
        app_name="my-app",
        user_id="user-123",
        query="test",
    )

    mock_valkey_client.lrange.assert_called_once_with(
        "adk:memory:my-app:user-123:entries", 0, -1
    )


class TestValkeyMemoryServiceClose:
  """Tests for close method."""

  @pytest.mark.asyncio
  async def test_close_does_not_close_client(
      self, memory_service, mock_valkey_client
  ):
    """Test that close does not close the underlying client."""
    await memory_service.close()
    # Client's close should NOT be called
    mock_valkey_client.close.assert_not_called()
