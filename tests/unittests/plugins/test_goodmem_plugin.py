# Copyright 2026 pairsys.ai (DBA Goodmem.ai)
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
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest
from google.genai import types

from google.adk_community.plugins.goodmem import GoodmemClient
from google.adk_community.plugins.goodmem.goodmem_plugin import GoodmemChatPlugin


# Mock constants
MOCK_BASE_URL = "https://api.goodmem.ai"
MOCK_API_KEY = "test-api-key"
MOCK_EMBEDDER_ID = "test-embedder-id"
MOCK_SPACE_ID = "test-space-id"
MOCK_SPACE_NAME = "adk_chat_test_user"
MOCK_USER_ID = "test_user"
MOCK_SESSION_ID = "test_session"
MOCK_MEMORY_ID = "test-memory-id"


class TestGoodmemClient:
  """Tests for GoodmemClient."""


  @pytest.fixture
  def mock_httpx_client(self) -> MagicMock:
    """Mock httpx.Client for testing."""
    with patch('google.adk_community.plugins.goodmem.client.httpx') as mock_httpx:
      mock_client = MagicMock()
      mock_httpx.Client.return_value = mock_client
      yield mock_client

  @pytest.fixture
  def goodmem_client(self, mock_httpx_client: MagicMock) -> GoodmemClient:
    """Create GoodmemClient instance for testing."""
    return GoodmemClient(base_url=MOCK_BASE_URL, api_key=MOCK_API_KEY)

  def test_client_initialization(self, goodmem_client: GoodmemClient) -> None:
    """Test client initialization."""
    assert goodmem_client._base_url == MOCK_BASE_URL
    assert goodmem_client._api_key == MOCK_API_KEY
    assert goodmem_client._headers["x-api-key"] == MOCK_API_KEY
    assert goodmem_client._headers["Content-Type"] == "application/json"

  def test_create_space(self, goodmem_client: GoodmemClient, mock_httpx_client: MagicMock) -> None:
    """Test creating a new space."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"spaceId": MOCK_SPACE_ID}
    mock_response.raise_for_status = MagicMock()
    mock_httpx_client.post.return_value = mock_response

    result = goodmem_client.create_space(MOCK_SPACE_NAME, MOCK_EMBEDDER_ID)

    assert result["spaceId"] == MOCK_SPACE_ID
    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    assert call_args.args[0] == "/v1/spaces"
    assert call_args.kwargs["json"]["name"] == MOCK_SPACE_NAME
    assert call_args.kwargs["json"]["spaceEmbedders"][0]["embedderId"] == MOCK_EMBEDDER_ID

  def test_insert_memory(self, goodmem_client: GoodmemClient, mock_httpx_client: MagicMock) -> None:
    """Test inserting a text memory."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "memoryId": MOCK_MEMORY_ID,
        "processingStatus": "COMPLETED"
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx_client.post.return_value = mock_response

    content = "Test memory content"
    metadata = {"session_id": MOCK_SESSION_ID, "user_id": MOCK_USER_ID}
    result = goodmem_client.insert_memory(
        MOCK_SPACE_ID, content, "text/plain", metadata
    )

    assert result["memoryId"] == MOCK_MEMORY_ID
    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    assert call_args.args[0] == "/v1/memories"
    assert call_args.kwargs["json"]["spaceId"] == MOCK_SPACE_ID
    assert call_args.kwargs["json"]["originalContent"] == content
    assert call_args.kwargs["json"]["metadata"] == metadata

  def test_insert_memory_binary(self, goodmem_client: GoodmemClient, mock_httpx_client: MagicMock) -> None:
    """Test inserting a binary memory using multipart upload."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "memoryId": MOCK_MEMORY_ID,
        "processingStatus": "COMPLETED"
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx_client.post.return_value = mock_response

    file_bytes = b"test file content"
    metadata = {"filename": "test.pdf", "user_id": MOCK_USER_ID}

    result = goodmem_client.insert_memory_binary(
        MOCK_SPACE_ID, file_bytes, "application/pdf", metadata
    )

    assert result["memoryId"] == MOCK_MEMORY_ID
    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args

    # Verify multipart form data was used
    assert "data" in call_args.kwargs
    assert "files" in call_args.kwargs
    data = call_args.kwargs["data"]
    files = call_args.kwargs["files"]

    # Check request metadata (in data parameter)
    assert "request" in data
    request_json = json.loads(data["request"])
    assert request_json["spaceId"] == MOCK_SPACE_ID
    assert request_json["contentType"] == "application/pdf"
    assert request_json["metadata"] == metadata

    # Check file content (in files parameter)
    assert "file" in files
    assert files["file"][1] == file_bytes
    assert files["file"][2] == "application/pdf"

  def test_retrieve_memories(self, goodmem_client: GoodmemClient, mock_httpx_client: MagicMock) -> None:
    """Test retrieving memories."""
    mock_response = MagicMock()
    # Simulate NDJSON response
    ndjson_lines = [
        json.dumps({"retrievedItem": {"chunk": {"chunk": {"chunkText": "chunk 1", "memoryId": "mem1"}}}}),
        json.dumps({"status": "complete"}),
        json.dumps({"retrievedItem": {"chunk": {"chunk": {"chunkText": "chunk 2", "memoryId": "mem2"}}}})
    ]
    mock_response.text = "\n".join(ndjson_lines)
    mock_response.raise_for_status = MagicMock()
    mock_httpx_client.post.return_value = mock_response

    query = "test query"
    space_ids = [MOCK_SPACE_ID]
    result = goodmem_client.retrieve_memories(query, space_ids, request_size=5)

    assert len(result) == 2  # Only items with retrievedItem
    assert result[0]["retrievedItem"]["chunk"]["chunk"]["chunkText"] == "chunk 1"
    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    assert call_args.kwargs["json"]["message"] == query
    assert call_args.kwargs["json"]["requestedSize"] == 5

  def test_list_spaces(self, goodmem_client: GoodmemClient, mock_httpx_client: MagicMock) -> None:
    """Test getting all spaces."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "spaces": [
            {"spaceId": "space1", "name": "Space 1"},
            {"spaceId": "space2", "name": "Space 2"}
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx_client.get.return_value = mock_response

    result = goodmem_client.list_spaces()

    assert len(result) == 2
    assert result[0]["name"] == "Space 1"
    mock_httpx_client.get.assert_called_once_with(
        "/v1/spaces",
        params={"maxResults": 1000},
        timeout=30.0,
    )

  def test_list_spaces_with_name_filter(
      self, goodmem_client: GoodmemClient, mock_httpx_client: MagicMock
  ) -> None:
    """Test filtering spaces by name."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "spaces": [
            {"spaceId": "space1", "name": "adk_chat_test_user"}
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx_client.get.return_value = mock_response

    result = goodmem_client.list_spaces(name=MOCK_SPACE_NAME)

    assert len(result) == 1
    assert result[0]["name"] == "adk_chat_test_user"
    mock_httpx_client.get.assert_called_once_with(
        "/v1/spaces",
        params={"maxResults": 1000, "nameFilter": MOCK_SPACE_NAME},
        timeout=30.0,
    )

  def test_list_embedders(self, goodmem_client: GoodmemClient, mock_httpx_client: MagicMock) -> None:
    """Test listing embedders."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "embedders": [
            {"embedderId": "emb1", "name": "Embedder 1"},
            {"embedderId": "emb2", "name": "Embedder 2"}
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx_client.get.return_value = mock_response

    result = goodmem_client.list_embedders()

    assert len(result) == 2
    assert result[0]["embedderId"] == "emb1"
    mock_httpx_client.get.assert_called_once_with(
        "/v1/embedders",
        timeout=30.0,
    )

  def test_get_memory_by_id(self, goodmem_client: GoodmemClient, mock_httpx_client: MagicMock) -> None:
    """Test getting a memory by ID."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "memoryId": MOCK_MEMORY_ID,
        "metadata": {"user_id": MOCK_USER_ID}
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx_client.get.return_value = mock_response

    result = goodmem_client.get_memory_by_id(MOCK_MEMORY_ID)

    assert result["memoryId"] == MOCK_MEMORY_ID
    assert result["metadata"]["user_id"] == MOCK_USER_ID
    from urllib.parse import quote
    encoded_memory_id = quote(MOCK_MEMORY_ID, safe="")
    mock_httpx_client.get.assert_called_once_with(
        f"/v1/memories/{encoded_memory_id}",
        timeout=30.0,
    )

  def test_get_memories_batch(
      self, goodmem_client: GoodmemClient, mock_httpx_client: MagicMock
  ) -> None:
    """Test batch get of memories (POST /v1/memories:batchGet)."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "memories": [
            {"memoryId": "mem1", "metadata": {"role": "user"}},
            {"memoryId": "mem2", "metadata": {"role": "LLM"}},
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx_client.post.return_value = mock_response

    result = goodmem_client.get_memories_batch(["mem1", "mem2"])

    assert len(result) == 2
    assert result[0]["memoryId"] == "mem1"
    assert result[0]["metadata"]["role"] == "user"
    assert result[1]["memoryId"] == "mem2"
    mock_httpx_client.post.assert_called_once()
    call_args = mock_httpx_client.post.call_args
    assert call_args.args[0] == "/v1/memories:batchGet"
    assert set(call_args.kwargs["json"]["memoryIds"]) == {"mem1", "mem2"}

  def test_get_memories_batch_empty(
      self, goodmem_client: GoodmemClient, mock_httpx_client: MagicMock
  ) -> None:
    """Test get_memories_batch with empty list does not call API."""
    result = goodmem_client.get_memories_batch([])
    assert result == []
    mock_httpx_client.post.assert_not_called()


class TestGoodmemChatPlugin:
  """Tests for GoodmemChatPlugin."""


  @pytest.fixture
  def mock_goodmem_client(self) -> MagicMock:
    """Mock GoodmemClient for testing."""
    with patch('google.adk_community.plugins.goodmem.goodmem_plugin.GoodmemClient') as mock_client_class:
      mock_client = MagicMock()

      # Mock list_embedders
      mock_client.list_embedders.return_value = [
          {"embedderId": MOCK_EMBEDDER_ID, "name": "Test Embedder"}
      ]

      # Mock list_spaces
      mock_client.list_spaces.return_value = []

      # Mock create_space
      mock_client.create_space.return_value = {"spaceId": MOCK_SPACE_ID}

      # Mock insert_memory
      mock_client.insert_memory.return_value = {
          "memoryId": MOCK_MEMORY_ID,
          "processingStatus": "COMPLETED"
      }

      # Mock insert_memory_binary
      mock_client.insert_memory_binary.return_value = {
          "memoryId": MOCK_MEMORY_ID,
          "processingStatus": "COMPLETED"
      }

      # Mock retrieve_memories
      mock_client.retrieve_memories.return_value = []

      # Mock get_memory_by_id (used by tools / single fetch)
      mock_client.get_memory_by_id.return_value = {
          "memoryId": MOCK_MEMORY_ID,
          "metadata": {"user_id": MOCK_USER_ID, "role": "user"}
      }

      # Mock get_memories_batch (used by before_model_callback for metadata)
      mock_client.get_memories_batch.return_value = [
          {"memoryId": MOCK_MEMORY_ID, "metadata": {"user_id": MOCK_USER_ID, "role": "user"}}
      ]

      mock_client_class.return_value = mock_client
      yield mock_client

  @pytest.fixture
  def chat_plugin(self, mock_goodmem_client: MagicMock) -> GoodmemChatPlugin:
    """Create GoodmemChatPlugin instance for testing."""
    return GoodmemChatPlugin(
        base_url=MOCK_BASE_URL,
        api_key=MOCK_API_KEY,
        embedder_id=MOCK_EMBEDDER_ID,
        top_k=5,
        debug=False
    )

  def test_plugin_initialization(self, chat_plugin: GoodmemChatPlugin) -> None:
    """Test plugin initialization."""
    assert chat_plugin.name == "GoodmemChatPlugin"
    assert chat_plugin.embedder_id == MOCK_EMBEDDER_ID
    assert chat_plugin.top_k == 5
    assert chat_plugin.debug is False

  def test_plugin_initialization_no_embedder_id(self, mock_goodmem_client: MagicMock) -> None:
    """Test plugin initialization without embedder_id."""
    plugin = GoodmemChatPlugin(
        base_url=MOCK_BASE_URL,
        api_key=MOCK_API_KEY,
        top_k=5
    )
    # Should use first embedder from API
    assert plugin.embedder_id == MOCK_EMBEDDER_ID

  def test_plugin_initialization_no_embedders_fails(self, mock_goodmem_client: MagicMock) -> None:
    """Test that embedder resolution fails when no embedders available."""
    mock_goodmem_client.list_embedders.return_value = []

    plugin = GoodmemChatPlugin(
        base_url=MOCK_BASE_URL,
        api_key=MOCK_API_KEY
    )
    with pytest.raises(ValueError, match="No embedders available"):
      plugin._get_embedder_id()

  def test_plugin_initialization_invalid_embedder_fails(self, mock_goodmem_client: MagicMock) -> None:
    """Test that embedder resolution fails with invalid embedder_id."""
    plugin = GoodmemChatPlugin(
        base_url=MOCK_BASE_URL,
        api_key=MOCK_API_KEY,
        embedder_id="invalid-embedder-id"
    )
    with pytest.raises(ValueError, match="is not valid"):
      plugin._get_embedder_id()

  def test_plugin_initialization_no_network_call(self, mock_goodmem_client: MagicMock) -> None:
    """Test that __init__ does not call list_embedders (lazy resolution)."""
    GoodmemChatPlugin(
        base_url=MOCK_BASE_URL,
        api_key=MOCK_API_KEY,
        embedder_id=MOCK_EMBEDDER_ID,
    )
    mock_goodmem_client.list_embedders.assert_not_called()

  def test_plugin_initialization_requires_base_url(self) -> None:
    """Test plugin initialization requires base_url."""
    with pytest.raises(ValueError):
      GoodmemChatPlugin(
          base_url=None,
          api_key=MOCK_API_KEY
      )

  def test_plugin_initialization_requires_api_key(self) -> None:
    """Test plugin initialization requires api_key."""
    with pytest.raises(ValueError):
      GoodmemChatPlugin(
          base_url=MOCK_BASE_URL,
          api_key=None
      )

  @pytest.mark.asyncio
  async def test_ensure_chat_space_creates_new_space(self, chat_plugin: GoodmemChatPlugin, mock_goodmem_client: MagicMock) -> None:
    """Test _get_space_id creates a new space when it doesn't exist."""
    mock_goodmem_client.list_spaces.return_value = []

    # Create mock context with session state
    mock_context = MagicMock()
    mock_context.user_id = MOCK_USER_ID
    mock_context.state = {}

    space_id = chat_plugin._get_space_id(mock_context)

    mock_goodmem_client.list_spaces.assert_called_once_with(
        name=MOCK_SPACE_NAME
    )
    mock_goodmem_client.create_space.assert_called_once_with(
        MOCK_SPACE_NAME, MOCK_EMBEDDER_ID
    )
    assert space_id == MOCK_SPACE_ID
    assert mock_context.state['_goodmem_space_id'] == MOCK_SPACE_ID

  @pytest.mark.asyncio
  async def test_ensure_chat_space_uses_existing_space(self, chat_plugin: GoodmemChatPlugin, mock_goodmem_client: MagicMock) -> None:
    """Test _get_space_id uses existing space when found."""
    mock_goodmem_client.list_spaces.return_value = [
        {"spaceId": "existing-space-id", "name": MOCK_SPACE_NAME}
    ]

    # Create mock context with session state
    mock_context = MagicMock()
    mock_context.user_id = MOCK_USER_ID
    mock_context.state = {}

    space_id = chat_plugin._get_space_id(mock_context)

    mock_goodmem_client.list_spaces.assert_called_once_with(
        name=MOCK_SPACE_NAME
    )
    mock_goodmem_client.create_space.assert_not_called()
    assert space_id == "existing-space-id"
    assert mock_context.state['_goodmem_space_id'] == "existing-space-id"

  @pytest.mark.asyncio
  async def test_ensure_chat_space_skips_case_mismatch(
      self, chat_plugin: GoodmemChatPlugin, mock_goodmem_client: MagicMock
  ) -> None:
    """Test _get_space_id prefers exact name match over case-insensitive match."""
    mock_goodmem_client.list_spaces.return_value = [
        {"spaceId": "case-mismatch-id", "name": MOCK_SPACE_NAME.upper()},
        {"spaceId": "exact-match-id", "name": MOCK_SPACE_NAME},
    ]

    mock_context = MagicMock()
    mock_context.user_id = MOCK_USER_ID
    mock_context.state = {}

    space_id = chat_plugin._get_space_id(mock_context)

    mock_goodmem_client.list_spaces.assert_called_once_with(
        name=MOCK_SPACE_NAME
    )
    mock_goodmem_client.create_space.assert_not_called()
    assert space_id == "exact-match-id"
    assert mock_context.state['_goodmem_space_id'] == "exact-match-id"

  @pytest.mark.asyncio
  async def test_ensure_chat_space_uses_cache(self, chat_plugin: GoodmemChatPlugin, mock_goodmem_client: MagicMock) -> None:
    """Test _get_space_id uses session state cache."""
    # Create mock context with cached space_id in session state
    mock_context = MagicMock()
    mock_context.user_id = MOCK_USER_ID
    mock_context.state = {'_goodmem_space_id': 'cached-space-id'}

    space_id = chat_plugin._get_space_id(mock_context)

    mock_goodmem_client.list_spaces.assert_not_called()
    mock_goodmem_client.create_space.assert_not_called()
    assert space_id == "cached-space-id"

  @pytest.mark.asyncio
  async def test_on_user_message_logs_text(self, chat_plugin: GoodmemChatPlugin, mock_goodmem_client: MagicMock) -> None:
    """Test on_user_message_callback logs text messages."""
    # Create mock invocation context with session state
    # Use a real dict object, not a MagicMock, for state
    state_dict = {'_goodmem_space_id': MOCK_SPACE_ID}

    # Create a simple object for session with real dict state
    class MockSession:
      id = MOCK_SESSION_ID
      state = state_dict

    # Use spec_set to prevent MagicMock from having a 'state' attribute
    mock_context = MagicMock(spec=['user_id', 'session'])
    mock_context.user_id = MOCK_USER_ID
    mock_context.session = MockSession()

    # Create user message with text
    user_message = types.Content(
        role="user",
        parts=[types.Part(text="Hello, how are you?")]
    )

    await chat_plugin.on_user_message_callback(
        invocation_context=mock_context,
        user_message=user_message
    )

    # Verify memory was inserted
    mock_goodmem_client.insert_memory.assert_called_once()
    call_args = mock_goodmem_client.insert_memory.call_args
    # Check positional args
    assert MOCK_SPACE_ID in str(call_args)
    assert "User: Hello, how are you?" in str(call_args)
    # Check if metadata was passed (could be positional or keyword arg)
    if len(call_args.args) >= 4:
      metadata = call_args.args[3]
    else:
      metadata = call_args.kwargs.get('metadata')
    assert metadata["user_id"] == MOCK_USER_ID
    assert metadata["role"] == "user"

  @pytest.mark.asyncio
  async def test_on_user_message_logs_file_attachment(self, chat_plugin: GoodmemChatPlugin, mock_goodmem_client: MagicMock) -> None:
    """Test on_user_message_callback logs file attachments."""
    # Use a real dict object, not a MagicMock, for state
    state_dict = {'_goodmem_space_id': MOCK_SPACE_ID}

    # Create a simple object for session with real dict state
    class MockSession:
      id = MOCK_SESSION_ID
      state = state_dict

    # Use spec_set to prevent MagicMock from having a 'state' attribute
    mock_context = MagicMock(spec=['user_id', 'session'])
    mock_context.user_id = MOCK_USER_ID
    mock_context.session = MockSession()

    # Create user message with file attachment
    file_data = b"test file content"
    blob = types.Blob(data=file_data, mime_type="application/pdf")
    blob.display_name = "test.pdf"
    user_message = types.Content(
        role="user",
        parts=[types.Part(inline_data=blob)]
    )

    await chat_plugin.on_user_message_callback(
        invocation_context=mock_context,
        user_message=user_message
    )

    # Verify binary memory was inserted
    mock_goodmem_client.insert_memory_binary.assert_called_once()
    call_args = mock_goodmem_client.insert_memory_binary.call_args
    # Check arguments (could be positional or keyword)
    assert MOCK_SPACE_ID in str(call_args)
    assert "application/pdf" in str(call_args)
    if len(call_args.args) >= 4:
      metadata = call_args.args[3]
    else:
      metadata = call_args.kwargs.get('metadata')
    assert metadata["filename"] == "test.pdf"

  @pytest.mark.asyncio
  async def test_on_user_message_filters_unsupported_mime_types(self, mock_goodmem_client: MagicMock) -> None:
    """Test on_user_message_callback only sends supported MIME types to Goodmem.
    
    Note: Files are NOT filtered for the LLM - all files pass through. Only Goodmem
    storage is filtered. LLM errors must be handled at the application level.
    """
    plugin = GoodmemChatPlugin(
        base_url=MOCK_BASE_URL,
        api_key=MOCK_API_KEY,
        embedder_id=MOCK_EMBEDDER_ID
    )

    # Use a real dict object, not a MagicMock, for state
    state_dict = {'_goodmem_space_id': MOCK_SPACE_ID}

    class MockSession:
      id = MOCK_SESSION_ID
      state = state_dict

    mock_context = MagicMock(spec=['user_id', 'session'])
    mock_context.user_id = MOCK_USER_ID
    mock_context.session = MockSession()

    # Create user message with unsupported image file
    image_data = b"fake image data"
    image_blob = types.Blob(data=image_data, mime_type="image/png")
    image_blob.display_name = "test.png"

    # Create user message with supported PDF file
    pdf_data = b"fake pdf data"
    pdf_blob = types.Blob(data=pdf_data, mime_type="application/pdf")
    pdf_blob.display_name = "test.pdf"

    # Create user message with unsupported video file
    video_data = b"fake video data"
    video_blob = types.Blob(data=video_data, mime_type="video/mp4")
    video_blob.display_name = "test.mp4"

    # Create user message with supported text file
    text_data = b"fake text data"
    text_blob = types.Blob(data=text_data, mime_type="text/plain")
    text_blob.display_name = "test.txt"

    # Create user message with supported JSON file
    json_data = b'{"key": "value"}'
    json_blob = types.Blob(data=json_data, mime_type="application/json")
    json_blob.display_name = "test.json"

    user_message = types.Content(
        role="user",
        parts=[
            types.Part(inline_data=image_blob),
            types.Part(inline_data=pdf_blob),
            types.Part(inline_data=video_blob),
            types.Part(inline_data=text_blob),
            types.Part(inline_data=json_blob)
        ]
    )

    result = await plugin.on_user_message_callback(
        invocation_context=mock_context,
        user_message=user_message
    )

    # Verify only supported files (PDF, text, JSON) were inserted to Goodmem
    assert mock_goodmem_client.insert_memory_binary.call_count == 3
    call_args_list = mock_goodmem_client.insert_memory_binary.call_args_list
    # Check that supported types were inserted
    assert any("application/pdf" in str(call) for call in call_args_list)
    assert any("text/plain" in str(call) for call in call_args_list)
    assert any("application/json" in str(call) for call in call_args_list)
    # Check that unsupported types were not inserted to Goodmem
    assert not any("image/png" in str(call) for call in call_args_list)
    assert not any("video/mp4" in str(call) for call in call_args_list)

    # Verify that the callback returns None (no filtering for LLM - all files pass through)
    assert result is None

  @pytest.mark.asyncio
  async def test_on_user_message_error_handling(self, chat_plugin: GoodmemChatPlugin, mock_goodmem_client: MagicMock) -> None:
    """Test on_user_message_callback error handling."""
    mock_context = MagicMock()
    mock_context.user_id = MOCK_USER_ID

    mock_goodmem_client.insert_memory.side_effect = httpx.RequestError("API Error")

    user_message = types.Content(
        role="user",
        parts=[types.Part(text="Test message")]
    )

    # Should not raise; plugin catches httpx.HTTPError and returns None
    result = await chat_plugin.on_user_message_callback(
        invocation_context=mock_context,
        user_message=user_message
    )

    assert result is None

  def test_is_mime_type_supported(self, mock_goodmem_client: MagicMock) -> None:
    """Test _is_mime_type_supported method with various MIME types."""
    plugin = GoodmemChatPlugin(
        base_url=MOCK_BASE_URL,
        api_key=MOCK_API_KEY,
        embedder_id=MOCK_EMBEDDER_ID
    )

    # Test supported text/* types
    assert plugin._is_mime_type_supported("text/plain") is True
    assert plugin._is_mime_type_supported("text/html") is True
    assert plugin._is_mime_type_supported("text/markdown") is True
    assert plugin._is_mime_type_supported("text/csv") is True

    # Test supported application types
    assert plugin._is_mime_type_supported("application/pdf") is True
    assert plugin._is_mime_type_supported("application/rtf") is True
    assert plugin._is_mime_type_supported("application/msword") is True
    assert plugin._is_mime_type_supported(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ) is True

    # Test XML-based formats (contains "+xml")
    assert plugin._is_mime_type_supported("application/xhtml+xml") is True
    assert plugin._is_mime_type_supported("application/atom+xml") is True

    # Test JSON formats
    assert plugin._is_mime_type_supported("application/json") is True
    assert plugin._is_mime_type_supported("application/vnd.api+json") is True

    # Test unsupported types
    assert plugin._is_mime_type_supported("image/png") is False
    assert plugin._is_mime_type_supported("image/jpeg") is False
    assert plugin._is_mime_type_supported("video/mp4") is False
    assert plugin._is_mime_type_supported("audio/mpeg") is False
    assert plugin._is_mime_type_supported("application/zip") is False
    assert plugin._is_mime_type_supported("application/octet-stream") is False

    # Test edge cases
    assert plugin._is_mime_type_supported("") is False
    # Note: None would cause AttributeError on .lower(), but in practice mime_type
    # comes from blob.mime_type which is always a string or defaults to "application/octet-stream"

  def test_extract_user_content(self, chat_plugin: GoodmemChatPlugin) -> None:
    """Test _extract_user_content extracts text from LLM request."""
    # Create mock LLM request with actual types.Part
    mock_request = MagicMock()
    mock_request.contents = [
        types.Content(role="user", parts=[types.Part(text="User query text")])
    ]

    result = chat_plugin._extract_user_content(mock_request)

    assert result == "User query text"

  def test_format_timestamp(self, chat_plugin: GoodmemChatPlugin) -> None:
    """Test _format_timestamp formats millisecond timestamps."""
    # Test timestamp: 2026-01-18T00:00:00 UTC (1768694400 seconds)
    timestamp_ms = 1768694400000

    result = chat_plugin._format_timestamp(timestamp_ms)

    assert result == "2026-01-18T00:00:00Z"

  def test_format_chunk_context(self, chat_plugin: GoodmemChatPlugin) -> None:
    """Test _format_chunk_context formats chunks with metadata."""
    chunk_content = "User: Hello there"
    memory_id = "mem-123"
    timestamp_ms = 1768694400000
    metadata = {"role": "user", "filename": "test.pdf"}

    result = chat_plugin._format_chunk_context(
        chunk_content, memory_id, timestamp_ms, metadata
    )

    assert "- id: mem-123" in result
    assert "datetime_utc: 2026-01-18T00:00:00Z" in result
    assert "role: user" in result
    assert "filename: test.pdf" in result
    assert "Hello there" in result  # Prefix should be removed

  @pytest.mark.asyncio
  async def test_before_model_callback_augments_request(self, chat_plugin: GoodmemChatPlugin, mock_goodmem_client: MagicMock) -> None:
    """Test before_model_callback augments LLM request with memory (uses batch get)."""
    mock_context = MagicMock()
    mock_context.user_id = MOCK_USER_ID

    # Mock retrieve_memories to return chunks
    mock_goodmem_client.retrieve_memories.return_value = [
        {
            "retrievedItem": {
                "chunk": {
                    "chunk": {
                        "chunkId": "chunk1",
                        "memoryId": "mem1",
                        "chunkText": "User: Previous conversation",
                        "updatedAt": 1768694400000
                    }
                }
            }
        }
    ]

    mock_goodmem_client.get_memories_batch.return_value = [
        {"memoryId": "mem1", "metadata": {"role": "user"}}
    ]

    # Create LLM request
    mock_request = MagicMock()
    mock_part = MagicMock()
    mock_part.text = "Current user query"
    mock_content = MagicMock()
    mock_content.parts = [mock_part]
    mock_request.contents = [mock_content]

    result = await chat_plugin.before_model_callback(
        callback_context=mock_context,
        llm_request=mock_request
    )

    # Verify batch get was called once with the memory id (no N+1)
    mock_goodmem_client.get_memories_batch.assert_called_once()
    call_args = mock_goodmem_client.get_memories_batch.call_args
    assert set(call_args[0][0]) == {"mem1"}

    # Verify request was augmented
    assert "BEGIN MEMORY" in mock_part.text
    assert "END MEMORY" in mock_part.text
    assert "Previous conversation" in mock_part.text
    assert result is None

  @pytest.mark.asyncio
  async def test_before_model_callback_batch_get_multiple_memories(
      self, chat_plugin: GoodmemChatPlugin, mock_goodmem_client: MagicMock
  ) -> None:
    """Test before_model_callback uses single batch get for multiple memory IDs (no N+1)."""
    mock_context = MagicMock()
    mock_context.user_id = MOCK_USER_ID

    # Two chunks from two different memories
    mock_goodmem_client.retrieve_memories.return_value = [
        {
            "retrievedItem": {
                "chunk": {
                    "chunk": {
                        "memoryId": "mem1",
                        "chunkText": "User: First message",
                        "updatedAt": 1768694400000,
                    }
                }
            }
        },
        {
            "retrievedItem": {
                "chunk": {
                    "chunk": {
                        "memoryId": "mem2",
                        "chunkText": "LLM: Second response",
                        "updatedAt": 1768694401000,
                    }
                }
            }
        },
    ]
    mock_goodmem_client.get_memories_batch.return_value = [
        {"memoryId": "mem1", "metadata": {"role": "user"}},
        {"memoryId": "mem2", "metadata": {"role": "LLM"}},
    ]

    mock_request = MagicMock()
    mock_part = MagicMock()
    mock_part.text = "Current query"
    mock_request.contents = [MagicMock(parts=[mock_part])]

    await chat_plugin.before_model_callback(
        callback_context=mock_context,
        llm_request=mock_request,
    )

    # Single batch call with both IDs (no N+1)
    mock_goodmem_client.get_memories_batch.assert_called_once()
    call_args = mock_goodmem_client.get_memories_batch.call_args
    assert set(call_args[0][0]) == {"mem1", "mem2"}
    mock_goodmem_client.get_memory_by_id.assert_not_called()

  @pytest.mark.asyncio
  async def test_before_model_callback_no_chunks(self, chat_plugin: GoodmemChatPlugin, mock_goodmem_client: MagicMock) -> None:
    """Test before_model_callback when no chunks are retrieved."""
    mock_context = MagicMock()
    mock_context.user_id = MOCK_USER_ID

    mock_goodmem_client.retrieve_memories.return_value = []

    mock_request = MagicMock()
    mock_part = MagicMock()
    mock_part.text = "Current user query"
    mock_content = MagicMock()
    mock_content.parts = [mock_part]
    mock_request.contents = [mock_content]

    result = await chat_plugin.before_model_callback(
        callback_context=mock_context,
        llm_request=mock_request
    )

    # When no chunks retrieved, return early without modifying the request
    assert "BEGIN MEMORY" not in mock_part.text
    assert "END MEMORY" not in mock_part.text
    assert mock_part.text == "Current user query"  # Unchanged
    assert result is None

  @pytest.mark.asyncio
  async def test_before_model_callback_no_user_content(
      self, chat_plugin: GoodmemChatPlugin, mock_goodmem_client: MagicMock
  ) -> None:
    """Test before_model_callback when request has no extractable user text."""
    mock_context = MagicMock()
    mock_context.user_id = MOCK_USER_ID
    mock_context.state = {"_goodmem_space_id": MOCK_SPACE_ID}

    # Request with empty parts / no text - _extract_user_content returns ""
    mock_request = MagicMock()
    mock_request.contents = []  # No contents

    result = await chat_plugin.before_model_callback(
        callback_context=mock_context,
        llm_request=mock_request,
    )

    # Should return early without calling retrieve_memories
    mock_goodmem_client.retrieve_memories.assert_not_called()
    assert result is None

  @pytest.mark.asyncio
  async def test_before_model_callback_error_handling(self, chat_plugin: GoodmemChatPlugin, mock_goodmem_client: MagicMock) -> None:
    """Test before_model_callback error handling."""
    mock_context = MagicMock()
    mock_context.user_id = MOCK_USER_ID
    mock_context.state = {'_goodmem_space_id': MOCK_SPACE_ID}

    mock_goodmem_client.retrieve_memories.side_effect = httpx.RequestError("API Error")

    mock_request = MagicMock()
    mock_content = MagicMock()
    mock_content.parts = [types.Part(text="Test")]
    mock_request.contents = [mock_content]

    # Should not raise exception
    result = await chat_plugin.before_model_callback(
        callback_context=mock_context,
        llm_request=mock_request
    )

    assert result is None

  @pytest.mark.asyncio
  async def test_after_model_callback_logs_response(self, chat_plugin: GoodmemChatPlugin, mock_goodmem_client: MagicMock) -> None:
    """Test after_model_callback logs LLM response."""
    mock_context = MagicMock()
    mock_context.user_id = MOCK_USER_ID
    mock_context.session = MagicMock()
    mock_context.session.id = MOCK_SESSION_ID
    mock_context.state = {'_goodmem_space_id': MOCK_SPACE_ID}

    # Create LLM response
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "This is the LLM response"
    mock_response.content = mock_content

    await chat_plugin.after_model_callback(
        callback_context=mock_context,
        llm_response=mock_response
    )

    # Verify memory was inserted
    mock_goodmem_client.insert_memory.assert_called()
    call_args = mock_goodmem_client.insert_memory.call_args
    # Check that the call contains expected values
    assert MOCK_SPACE_ID in str(call_args)
    assert "LLM: This is the LLM response" in str(call_args)
    # Check metadata (could be positional or keyword arg)
    if len(call_args.args) >= 4:
      metadata = call_args.args[3]
    else:
      metadata = call_args.kwargs.get('metadata')
    assert metadata["role"] == "LLM"

  @pytest.mark.asyncio
  async def test_after_model_callback_no_space_id(self, chat_plugin: GoodmemChatPlugin, mock_goodmem_client: MagicMock) -> None:
    """Test after_model_callback when no space_id is cached in session state."""
    mock_context = MagicMock()
    mock_context.user_id = MOCK_USER_ID
    mock_context.session = MagicMock()
    mock_context.session.id = MOCK_SESSION_ID
    mock_context.state = {}  # Empty session state, no cached space_id

    # Mock existing space so _get_space_id will find it
    mock_goodmem_client.list_spaces.return_value = [
        {"name": MOCK_SPACE_NAME, "spaceId": MOCK_SPACE_ID}
    ]

    mock_response = MagicMock()
    mock_response.content = MagicMock()
    mock_response.content.text = "Test response"

    result = await chat_plugin.after_model_callback(
        callback_context=mock_context,
        llm_response=mock_response
    )

    mock_goodmem_client.list_spaces.assert_called_once_with(
        name=MOCK_SPACE_NAME
    )
    # With the fix, _ensure_chat_space is called and space_id is set
    # So insert_memory SHOULD be called
    assert mock_goodmem_client.insert_memory.called
    assert result is None

  @pytest.mark.asyncio
  async def test_after_model_callback_error_handling(self, chat_plugin: GoodmemChatPlugin, mock_goodmem_client: MagicMock) -> None:
    """Test after_model_callback error handling."""
    mock_context = MagicMock()
    mock_context.user_id = MOCK_USER_ID
    mock_context.session = MagicMock()
    mock_context.session.id = MOCK_SESSION_ID
    mock_context.state = {'_goodmem_space_id': MOCK_SPACE_ID}

    mock_goodmem_client.insert_memory.side_effect = httpx.RequestError("API Error")

    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "Response text"
    mock_response.content = mock_content

    # Should not raise; plugin catches httpx.HTTPError and returns None
    result = await chat_plugin.after_model_callback(
        callback_context=mock_context,
        llm_response=mock_response
    )

    assert result is None

  @pytest.mark.asyncio
  async def test_plugin_with_debug_mode(self, mock_goodmem_client: MagicMock) -> None:
    """Test plugin with debug mode enabled."""
    plugin = GoodmemChatPlugin(
        base_url=MOCK_BASE_URL,
        api_key=MOCK_API_KEY,
        embedder_id=MOCK_EMBEDDER_ID,
        debug=True
    )

    assert plugin.debug is True

  @pytest.mark.asyncio
  async def test_full_conversation_flow(self, chat_plugin: GoodmemChatPlugin, mock_goodmem_client: MagicMock) -> None:
    """Test full conversation flow with user message, retrieval, and response logging."""
    shared_state = {}  # Shared state dict for both invocation and callback contexts
    mock_context = MagicMock()
    mock_context.user_id = MOCK_USER_ID
    mock_context.session = MagicMock()
    mock_context.session.id = MOCK_SESSION_ID
    mock_context.session.state = shared_state  # For invocation_context access
    mock_context.state = shared_state  # For callback_context access

    # 1. User sends a message
    user_message = types.Content(
        role="user",
        parts=[types.Part(text="What's the weather?")]
    )

    await chat_plugin.on_user_message_callback(
        invocation_context=mock_context,
        user_message=user_message
    )

    # Verify user message was logged
    assert mock_goodmem_client.insert_memory.called

    # 2. Before model is called, retrieve context
    mock_goodmem_client.retrieve_memories.return_value = [
        {
            "retrievedItem": {
                "chunk": {
                    "chunk": {
                        "memoryId": "mem1",
                        "chunkText": "User: I'm in San Francisco",
                        "updatedAt": 1768694400000
                    }
                }
            }
        }
    ]

    mock_request = MagicMock()
    mock_part = MagicMock()
    mock_part.text = "What's the weather?"
    mock_content = MagicMock()
    mock_content.parts = [mock_part]
    mock_request.contents = [mock_content]

    await chat_plugin.before_model_callback(
        callback_context=mock_context,
        llm_request=mock_request
    )

    # Verify request was augmented with context
    assert "BEGIN MEMORY" in mock_part.text

    # 3. After model responds, log the response
    mock_response = MagicMock()
    mock_response_content = MagicMock()
    mock_response_content.text = "It's sunny in San Francisco"
    mock_response.content = mock_response_content

    await chat_plugin.after_model_callback(
        callback_context=mock_context,
        llm_response=mock_response
    )

    # Verify LLM response was logged
    insert_calls = [call for call in mock_goodmem_client.insert_memory.call_args_list]
    assert len(insert_calls) >= 2  # At least user message and LLM response

  @pytest.mark.asyncio
  async def test_multi_user_isolation(self, mock_goodmem_client: MagicMock) -> None:
    """Test that multiple users don't leak data to each other."""
    plugin = GoodmemChatPlugin(
        base_url=MOCK_BASE_URL,
        api_key=MOCK_API_KEY,
        embedder_id=MOCK_EMBEDDER_ID
    )

    # Mock spaces for two different users
    def list_spaces_side_effect(*, name=None, **kwargs):
      if name == "adk_chat_alice":
        return [{"name": "adk_chat_alice", "spaceId": "space_alice"}]
      if name == "adk_chat_bob":
        return [{"name": "adk_chat_bob", "spaceId": "space_bob"}]
      return []

    mock_goodmem_client.list_spaces.side_effect = list_spaces_side_effect

    # Context for User Alice
    alice_context = MagicMock()
    alice_context.user_id = "alice"
    alice_context.session = MagicMock()
    alice_context.session.id = "session_alice"
    alice_context.state = {}  # Separate session state for Alice

    # Context for User Bob
    bob_context = MagicMock()
    bob_context.user_id = "bob"
    bob_context.session = MagicMock()
    bob_context.session.id = "session_bob"
    bob_context.state = {}  # Separate session state for Bob

    # Alice's response
    alice_response = MagicMock()
    alice_response.content = MagicMock()
    alice_response.content.text = "Alice's secret data"

    # Bob's response
    bob_response = MagicMock()
    bob_response.content = MagicMock()
    bob_response.content.text = "Bob's secret data"

    # Log Alice's response
    await plugin.after_model_callback(
        callback_context=alice_context,
        llm_response=alice_response
    )

    # Verify Alice's data went to Alice's space
    calls = mock_goodmem_client.insert_memory.call_args_list
    assert calls[-1][0][0] == "space_alice"  # First arg is space_id
    assert "Alice's secret data" in calls[-1][0][1]  # Second arg is content

    # Log Bob's response
    await plugin.after_model_callback(
        callback_context=bob_context,
        llm_response=bob_response
    )

    # Verify Bob's data went to Bob's space (NOT Alice's!)
    calls = mock_goodmem_client.insert_memory.call_args_list
    assert calls[-1][0][0] == "space_bob"  # NOT "space_alice"
    assert "Bob's secret data" in calls[-1][0][1]
    assert mock_goodmem_client.list_spaces.call_args_list == [
        call(name="adk_chat_alice"),
        call(name="adk_chat_bob"),
    ]

  @pytest.mark.asyncio
  async def test_debug_mode_empty_retrieval_consistency(self, mock_goodmem_client: MagicMock) -> None:
    """Test that debug mode doesn't alter behavior when retrieval is empty."""

    # Test with debug=False
    plugin_no_debug = GoodmemChatPlugin(
        base_url=MOCK_BASE_URL,
        api_key=MOCK_API_KEY,
        embedder_id=MOCK_EMBEDDER_ID,
        debug=False
    )

    # Test with debug=True
    plugin_debug = GoodmemChatPlugin(
        base_url=MOCK_BASE_URL,
        api_key=MOCK_API_KEY,
        embedder_id=MOCK_EMBEDDER_ID,
        debug=True
    )

    # Mock empty retrieval
    mock_goodmem_client.retrieve_memories.return_value = []
    mock_goodmem_client.list_spaces.return_value = [
        {"name": "adk_chat_test_user", "spaceId": MOCK_SPACE_ID}
    ]

    mock_context = MagicMock()
    mock_context.user_id = MOCK_USER_ID
    mock_context.state = {'_goodmem_space_id': MOCK_SPACE_ID}

    mock_request = MagicMock()
    mock_request.contents = [
        types.Content(role="user", parts=[types.Part(text="Hello")])
    ]

    # Call both plugins with empty retrieval
    result_no_debug = await plugin_no_debug.before_model_callback(
        callback_context=mock_context,
        llm_request=mock_request
    )

    result_debug = await plugin_debug.before_model_callback(
        callback_context=mock_context,
        llm_request=mock_request
    )

    # BOTH should return None (not inject empty memory block)
    assert result_no_debug is None
    assert result_debug is None

    # BOTH should have same behavior - early return, no modification
    # This test would FAIL with the old code because debug=True returns early
    # while debug=False continues and injects empty memory block

  @pytest.mark.asyncio
  @pytest.mark.filterwarnings("ignore:coroutine .* was never awaited:RuntimeWarning")
  async def test_concurrent_user_race_condition(self, mock_goodmem_client: MagicMock) -> None:
    """Test that concurrent requests from different users don't cause data leakage."""
    import asyncio

    plugin = GoodmemChatPlugin(
        base_url=MOCK_BASE_URL,
        api_key=MOCK_API_KEY,
        embedder_id=MOCK_EMBEDDER_ID
    )

    # Mock spaces for two users
    def list_spaces_side_effect(*, name=None, **kwargs):
      if name == "adk_chat_alice":
        return [{"name": "adk_chat_alice", "spaceId": "space_alice"}]
      if name == "adk_chat_bob":
        return [{"name": "adk_chat_bob", "spaceId": "space_bob"}]
      return []

    mock_goodmem_client.list_spaces.side_effect = list_spaces_side_effect

    # Track which space_id was used for each insert_memory call
    insert_memory_calls = []

    def track_insert(space_id, content, *args, **kwargs):
        insert_memory_calls.append({
            "space_id": space_id,
            "content": content
        })
        return {"memoryId": "test-id", "processingStatus": "COMPLETED"}

    mock_goodmem_client.insert_memory.side_effect = track_insert

    # retrieve_memories is called synchronously by the plugin; return [] so
    # before_model_callback completes without error
    mock_goodmem_client.retrieve_memories.return_value = []

    # Alice's context and response
    alice_context = MagicMock()
    alice_context.user_id = "alice"
    alice_context.session = MagicMock()
    alice_context.session.id = "session_alice"
    alice_context.state = {}  # Separate session state for Alice

    alice_response = MagicMock()
    alice_response.content = MagicMock()
    alice_response.content.text = "Alice's confidential message"

    # Bob's context and response
    bob_context = MagicMock()
    bob_context.user_id = "bob"
    bob_context.session = MagicMock()
    bob_context.session.id = "session_bob"
    bob_context.state = {}  # Separate session state for Bob

    bob_response = MagicMock()
    bob_response.content = MagicMock()
    bob_response.content.text = "Bob's confidential message"

    # Simulate concurrent before_model_callback calls (sets self.space_id)
    alice_request = MagicMock()
    alice_request.contents = [types.Content(role="user", parts=[types.Part(text="Hi")])]

    bob_request = MagicMock()
    bob_request.contents = [types.Content(role="user", parts=[types.Part(text="Hey")])]

    # Run callbacks concurrently to trigger race condition
    await asyncio.gather(
        plugin.before_model_callback(callback_context=alice_context, llm_request=alice_request),
        plugin.before_model_callback(callback_context=bob_context, llm_request=bob_request),
    )

    # Now run after_model_callback concurrently
    await asyncio.gather(
        plugin.after_model_callback(callback_context=alice_context, llm_response=alice_response),
        plugin.after_model_callback(callback_context=bob_context, llm_response=bob_response),
    )

    # Verify each user's data went to their own space
    alice_calls = [c for c in insert_memory_calls if "Alice's confidential" in c["content"]]
    bob_calls = [c for c in insert_memory_calls if "Bob's confidential" in c["content"]]

    assert len(alice_calls) == 1, "Alice's message should be logged exactly once"
    assert len(bob_calls) == 1, "Bob's message should be logged exactly once"

    # CRITICAL: Alice's data must NOT go to Bob's space
    assert alice_calls[0]["space_id"] == "space_alice", \
        f"Alice's data leaked to {alice_calls[0]['space_id']} instead of space_alice!"

    # CRITICAL: Bob's data must NOT go to Alice's space
    assert bob_calls[0]["space_id"] == "space_bob", \
        f"Bob's data leaked to {bob_calls[0]['space_id']} instead of space_bob!"
    called_names = {
        kwargs.get("name")
        for _, kwargs in mock_goodmem_client.list_spaces.call_args_list
    }
    assert called_names == {"adk_chat_alice", "adk_chat_bob"}
