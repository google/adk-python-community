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

from unittest.mock import AsyncMock
from unittest.mock import patch

from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types
import pytest

from google.adk_community.memory.valkey_memory_service import ValkeyMemoryService
from google.adk_community.memory.valkey_memory_service import ValkeyMemoryServiceConfig

MOCK_APP_NAME = "test-app"
MOCK_USER_ID = "test-user"
MOCK_SESSION_ID = "session-1"


async def _mock_embed_fn(texts: list[str]) -> list[list[float]]:
  """Simple mock embedding function returning fixed-dim vectors."""
  return [[0.1] * 768 for _ in texts]


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
        # Function call event, should be ignored
        Event(
            id="event-4",
            invocation_id="inv-4",
            author="agent",
            timestamp=12348,
            content=types.Content(
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name="test_function")
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
  client.hset = AsyncMock(return_value=1)
  client.expire = AsyncMock(return_value=True)
  return client


@pytest.fixture
def memory_service(mock_valkey_client):
  """Create ValkeyMemoryService instance for testing."""
  service = ValkeyMemoryService(
      client=mock_valkey_client,
      embedding_function=_mock_embed_fn,
  )
  service._index_created = True
  return service


@pytest.fixture
def memory_service_with_config(mock_valkey_client):
  """Create ValkeyMemoryService with custom config."""
  config = ValkeyMemoryServiceConfig(
      similarity_top_k=5,
      key_prefix="custom:mem",
      index_name="custom_idx",
      ttl_seconds=3600,
      embedding_dimensions=768,
      vector_distance_threshold=0.5,
  )
  service = ValkeyMemoryService(
      client=mock_valkey_client,
      embedding_function=_mock_embed_fn,
      config=config,
  )
  service._index_created = True
  return service


class TestValkeyMemoryServiceConfig:
  """Tests for ValkeyMemoryServiceConfig."""

  def test_default_config(self):
    """Test default configuration values."""
    config = ValkeyMemoryServiceConfig()
    assert config.similarity_top_k == 10
    assert config.key_prefix == "adk:memory"
    assert config.index_name == "adk_memory_idx"
    assert config.ttl_seconds is None
    assert config.embedding_dimensions == 768
    assert config.distance_metric == "COSINE"
    assert config.vector_distance_threshold is None

  def test_custom_config(self):
    """Test custom configuration values."""
    config = ValkeyMemoryServiceConfig(
        similarity_top_k=20,
        key_prefix="my:prefix",
        index_name="my_index",
        ttl_seconds=7200,
        embedding_dimensions=1536,
        distance_metric="L2",
        vector_distance_threshold=0.8,
    )
    assert config.similarity_top_k == 20
    assert config.key_prefix == "my:prefix"
    assert config.index_name == "my_index"
    assert config.ttl_seconds == 7200
    assert config.embedding_dimensions == 1536
    assert config.distance_metric == "L2"
    assert config.vector_distance_threshold == 0.8

  def test_config_validation_top_k(self):
    """Test similarity_top_k validation."""
    with pytest.raises(Exception):
      ValkeyMemoryServiceConfig(similarity_top_k=0)

    with pytest.raises(Exception):
      ValkeyMemoryServiceConfig(similarity_top_k=1001)


class TestValkeyMemoryServiceInit:
  """Tests for ValkeyMemoryService initialization."""

  def test_client_required(self):
    """Test that client is required."""
    with pytest.raises(ValueError, match="client is required"):
      ValkeyMemoryService(client=None, embedding_function=_mock_embed_fn)

  def test_embedding_function_required(self, mock_valkey_client):
    """Test that embedding_function is required."""
    with pytest.raises(ValueError, match="embedding_function is required"):
      ValkeyMemoryService(client=mock_valkey_client, embedding_function=None)

  def test_init_with_defaults(self, mock_valkey_client):
    """Test initialization with default config."""
    service = ValkeyMemoryService(
        client=mock_valkey_client,
        embedding_function=_mock_embed_fn,
    )
    assert service._client is mock_valkey_client
    assert service._config.similarity_top_k == 10
    assert service._index_created is False

  def test_init_with_config(self, mock_valkey_client):
    """Test initialization with custom config."""
    config = ValkeyMemoryServiceConfig(similarity_top_k=5)
    service = ValkeyMemoryService(
        client=mock_valkey_client,
        embedding_function=_mock_embed_fn,
        config=config,
    )
    assert service._config.similarity_top_k == 5


class TestValkeyMemoryServiceCreateIndex:
  """Tests for create_index."""

  @pytest.mark.asyncio
  async def test_create_index_success(self, mock_valkey_client):
    """Test successful index creation."""
    service = ValkeyMemoryService(
        client=mock_valkey_client,
        embedding_function=_mock_embed_fn,
    )

    with patch("glide.ft.create", new_callable=AsyncMock) as mock_create:
      mock_create.return_value = "OK"
      await service.create_index()

      assert service._index_created is True
      mock_create.assert_called_once()

  @pytest.mark.asyncio
  async def test_create_index_already_exists(self, mock_valkey_client):
    """Test that existing index is handled gracefully."""
    service = ValkeyMemoryService(
        client=mock_valkey_client,
        embedding_function=_mock_embed_fn,
    )

    with patch("glide.ft.create", new_callable=AsyncMock) as mock_create:
      mock_create.side_effect = Exception("Index already exists")
      await service.create_index()
      assert service._index_created is True

  @pytest.mark.asyncio
  async def test_create_index_unexpected_error(self, mock_valkey_client):
    """Test that unexpected errors are raised."""
    service = ValkeyMemoryService(
        client=mock_valkey_client,
        embedding_function=_mock_embed_fn,
    )

    with patch("glide.ft.create", new_callable=AsyncMock) as mock_create:
      mock_create.side_effect = Exception("Connection refused")
      with pytest.raises(Exception, match="Connection refused"):
        await service.create_index()
      assert service._index_created is False


class TestValkeyMemoryServiceAddSession:
  """Tests for add_session_to_memory."""

  @pytest.mark.asyncio
  async def test_add_session_success(self, memory_service, mock_valkey_client):
    """Test successful addition of session memories."""
    await memory_service.add_session_to_memory(MOCK_SESSION)

    # Should make 2 hset calls (one per valid event with text)
    assert mock_valkey_client.hset.call_count == 2

    # Check first call stores correct fields including embedding
    first_call = mock_valkey_client.hset.call_args_list[0]
    key = first_call[0][0]
    assert key.startswith("adk:memory:")
    fields = first_call[0][1]
    assert fields["content"] == "Hello, I like Python."
    assert fields["author"] == "user"
    assert fields["app_name"] == MOCK_APP_NAME
    assert fields["user_id"] == MOCK_USER_ID
    assert "embedding" in fields
    assert isinstance(fields["embedding"], bytes)

  @pytest.mark.asyncio
  async def test_add_session_filters_empty_events(
      self, memory_service, mock_valkey_client
  ):
    """Test that events without text content are filtered out."""
    await memory_service.add_session_to_memory(MOCK_SESSION_WITH_EMPTY_EVENTS)
    assert mock_valkey_client.hset.call_count == 0

  @pytest.mark.asyncio
  async def test_add_session_with_ttl(
      self, memory_service_with_config, mock_valkey_client
  ):
    """Test that TTL is set when configured."""
    await memory_service_with_config.add_session_to_memory(MOCK_SESSION)

    assert mock_valkey_client.expire.call_count == 2
    expire_call = mock_valkey_client.expire.call_args_list[0]
    assert expire_call[0][1] == 3600

  @pytest.mark.asyncio
  async def test_add_session_no_ttl_by_default(
      self, memory_service, mock_valkey_client
  ):
    """Test that no TTL is set when not configured."""
    await memory_service.add_session_to_memory(MOCK_SESSION)
    mock_valkey_client.expire.assert_not_called()

  @pytest.mark.asyncio
  async def test_add_session_embedding_error(self, mock_valkey_client):
    """Test handling of embedding function failure."""

    async def _failing_embed(texts):
      raise RuntimeError("Embedding service unavailable")

    service = ValkeyMemoryService(
        client=mock_valkey_client,
        embedding_function=_failing_embed,
    )
    service._index_created = True

    # Should not raise, just log error
    await service.add_session_to_memory(MOCK_SESSION)
    mock_valkey_client.hset.assert_not_called()

  @pytest.mark.asyncio
  async def test_add_session_hset_error(
      self, memory_service, mock_valkey_client
  ):
    """Test error handling during hset."""
    mock_valkey_client.hset.side_effect = Exception("Connection error")

    await memory_service.add_session_to_memory(MOCK_SESSION)
    assert mock_valkey_client.hset.call_count == 2

  @pytest.mark.asyncio
  async def test_add_session_custom_key_prefix(
      self, memory_service_with_config, mock_valkey_client
  ):
    """Test that custom key prefix is used."""
    await memory_service_with_config.add_session_to_memory(MOCK_SESSION)

    first_call = mock_valkey_client.hset.call_args_list[0]
    key = first_call[0][0]
    assert key.startswith("custom:mem:")

  @pytest.mark.asyncio
  async def test_add_session_creates_index_if_needed(self, mock_valkey_client):
    """Test that create_index is called if not yet created."""
    service = ValkeyMemoryService(
        client=mock_valkey_client,
        embedding_function=_mock_embed_fn,
    )

    with patch("glide.ft.create", new_callable=AsyncMock) as mock_create:
      mock_create.return_value = "OK"
      await service.add_session_to_memory(MOCK_SESSION)
      mock_create.assert_called_once()
      assert service._index_created is True


class TestValkeyMemoryServiceAddEvents:
  """Tests for add_events_to_memory."""

  @pytest.mark.asyncio
  async def test_add_events_success(self, memory_service, mock_valkey_client):
    """Test incremental event ingestion."""
    events = [
        Event(
            id="ev-1",
            invocation_id="inv-1",
            author="user",
            timestamp=100,
            content=types.Content(parts=[types.Part(text="Hello world")]),
        ),
    ]

    await memory_service.add_events_to_memory(
        app_name="myapp",
        user_id="user1",
        events=events,
        session_id="sess-1",
    )

    assert mock_valkey_client.hset.call_count == 1
    fields = mock_valkey_client.hset.call_args_list[0][0][1]
    assert fields["content"] == "Hello world"
    assert fields["app_name"] == "myapp"
    assert fields["user_id"] == "user1"
    assert fields["session_id"] == "sess-1"

  @pytest.mark.asyncio
  async def test_add_events_filters_empty(
      self, memory_service, mock_valkey_client
  ):
    """Test that empty events are skipped."""
    events = [
        Event(id="ev-empty", invocation_id="inv-1", author="user"),
    ]

    await memory_service.add_events_to_memory(
        app_name="myapp",
        user_id="user1",
        events=events,
    )

    mock_valkey_client.hset.assert_not_called()


class TestValkeyMemoryServiceSearch:
  """Tests for search_memory."""

  @pytest.mark.asyncio
  async def test_search_memory_success(
      self, memory_service, mock_valkey_client
  ):
    """Test successful memory search using KNN."""
    search_result = [
        2,
        {
            b"adk:memory:abc123": {
                b"content": b"I love Python programming",
                b"author": b"user",
                b"timestamp": b"12345",
                b"__embedding_score": b"0.15",
            },
            b"adk:memory:def456": {
                b"content": b"Python has great libraries",
                b"author": b"model",
                b"timestamp": b"12346",
                b"__embedding_score": b"0.25",
            },
        },
    ]

    with patch("glide.ft.search", new_callable=AsyncMock) as mock_search:
      mock_search.return_value = search_result

      result = await memory_service.search_memory(
          app_name=MOCK_APP_NAME,
          user_id=MOCK_USER_ID,
          query="Python",
      )

      assert len(result.memories) == 2
      assert (
          result.memories[0].content.parts[0].text
          == "I love Python programming"
      )
      assert result.memories[0].author == "user"

  @pytest.mark.asyncio
  async def test_search_memory_empty_result(
      self, memory_service, mock_valkey_client
  ):
    """Test search when no results."""
    with patch("glide.ft.search", new_callable=AsyncMock) as mock_search:
      mock_search.return_value = [0, {}]

      result = await memory_service.search_memory(
          app_name=MOCK_APP_NAME,
          user_id=MOCK_USER_ID,
          query="anything",
      )
      assert len(result.memories) == 0

  @pytest.mark.asyncio
  async def test_search_memory_uses_knn_query(
      self, memory_service, mock_valkey_client
  ):
    """Test that search builds a KNN query with TAG filters."""
    with patch("glide.ft.search", new_callable=AsyncMock) as mock_search:
      mock_search.return_value = [0, {}]

      await memory_service.search_memory(
          app_name="my-app",
          user_id="user-123",
          query="test query",
      )

      call_args = mock_search.call_args
      query_str = call_args[0][2]
      assert "my\\-app" in query_str
      assert "user\\-123" in query_str
      assert "KNN" in query_str
      assert "@embedding" in query_str

  @pytest.mark.asyncio
  async def test_search_memory_passes_query_vec(
      self, memory_service, mock_valkey_client
  ):
    """Test that query embedding is passed as params."""
    with patch("glide.ft.search", new_callable=AsyncMock) as mock_search:
      mock_search.return_value = [0, {}]

      await memory_service.search_memory(
          app_name=MOCK_APP_NAME,
          user_id=MOCK_USER_ID,
          query="test",
      )

      call_args = mock_search.call_args
      options = call_args[0][3]
      assert "query_vec" in options.params
      assert isinstance(options.params["query_vec"], bytes)

  @pytest.mark.asyncio
  async def test_search_memory_distance_threshold(
      self, memory_service_with_config, mock_valkey_client
  ):
    """Test that distance threshold filters results."""
    search_result = [
        2,
        {
            b"adk:memory:close": {
                b"content": b"Close match",
                b"author": b"user",
                b"timestamp": b"1",
                b"__embedding_score": b"0.3",
            },
            b"adk:memory:far": {
                b"content": b"Far match",
                b"author": b"user",
                b"timestamp": b"2",
                b"__embedding_score": b"0.9",
            },
        },
    ]

    with patch("glide.ft.search", new_callable=AsyncMock) as mock_search:
      mock_search.return_value = search_result

      # Threshold is 0.5, so only "Close match" (0.3) should pass
      result = await memory_service_with_config.search_memory(
          app_name=MOCK_APP_NAME,
          user_id=MOCK_USER_ID,
          query="test",
      )

      assert len(result.memories) == 1
      assert result.memories[0].content.parts[0].text == "Close match"

  @pytest.mark.asyncio
  async def test_search_memory_embedding_error(self, mock_valkey_client):
    """Test graceful handling of embedding failure during search."""

    async def _failing_embed(texts):
      raise RuntimeError("Embedding service down")

    service = ValkeyMemoryService(
        client=mock_valkey_client,
        embedding_function=_failing_embed,
    )
    service._index_created = True

    result = await service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query="test",
    )
    assert len(result.memories) == 0

  @pytest.mark.asyncio
  async def test_search_memory_ft_search_error(
      self, memory_service, mock_valkey_client
  ):
    """Test graceful handling of FT.SEARCH failure."""
    with patch("glide.ft.search", new_callable=AsyncMock) as mock_search:
      mock_search.side_effect = Exception("Connection error")

      result = await memory_service.search_memory(
          app_name=MOCK_APP_NAME,
          user_id=MOCK_USER_ID,
          query="test",
      )
      assert len(result.memories) == 0


class TestValkeyMemoryServiceBuildQuery:
  """Tests for _build_knn_query."""

  def test_basic_knn_query(self, memory_service):
    """Test KNN query construction."""
    query = memory_service._build_knn_query("myapp", "user1", 10)
    assert "@app_name:{myapp}" in query
    assert "@user_id:{user1}" in query
    assert "KNN 10 @embedding" in query

  def test_hyphenated_values(self, memory_service):
    """Test escaping of hyphens in TAG values."""
    query = memory_service._build_knn_query("my-app", "user-1", 5)
    assert "my\\-app" in query
    assert "user\\-1" in query


class TestValkeyMemoryServiceClose:
  """Tests for close method."""

  @pytest.mark.asyncio
  async def test_close_does_not_close_client(
      self, memory_service, mock_valkey_client
  ):
    """Test that close does not close the underlying client."""
    await memory_service.close()
    mock_valkey_client.close.assert_not_called()
