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

"""Unit tests for GoodmemClient.

These tests focus on the HTTP request construction, particularly the binary
upload fix where Content-Type must NOT be set to application/json for
multipart requests.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from google.adk_community.plugins.goodmem import GoodmemClient


# Mock constants
MOCK_BASE_URL = "https://api.goodmem.ai"
MOCK_API_KEY = "test-api-key"
MOCK_SPACE_ID = "test-space-id"
MOCK_EMBEDDER_ID = "test-embedder-id"
MOCK_MEMORY_ID = "test-memory-id"


class TestGoodmemClientInit:
    """Tests for GoodmemClient initialization."""

    def test_init_sets_base_url(self) -> None:
        """Test that base_url is set correctly."""
        with patch("google.adk_community.plugins.goodmem.client.httpx.Client"):
            client = GoodmemClient(MOCK_BASE_URL, MOCK_API_KEY)
            assert client._base_url == MOCK_BASE_URL

    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from base_url."""
        with patch("google.adk_community.plugins.goodmem.client.httpx.Client"):
            client = GoodmemClient(f"{MOCK_BASE_URL}/", MOCK_API_KEY)
            assert client._base_url == MOCK_BASE_URL

    def test_init_sets_api_key(self) -> None:
        """Test that api_key is set correctly."""
        with patch("google.adk_community.plugins.goodmem.client.httpx.Client"):
            client = GoodmemClient(MOCK_BASE_URL, MOCK_API_KEY)
            assert client._api_key == MOCK_API_KEY

    def test_init_sets_default_headers(self) -> None:
        """Test that default headers include api key."""
        with patch("google.adk_community.plugins.goodmem.client.httpx.Client"):
            client = GoodmemClient(MOCK_BASE_URL, MOCK_API_KEY)
            assert client._headers["x-api-key"] == MOCK_API_KEY

    def test_init_creates_httpx_client(self) -> None:
        """Test that httpx.Client is created with correct config."""
        with patch("google.adk_community.plugins.goodmem.client.httpx.Client") as mock_client_class:
            GoodmemClient(MOCK_BASE_URL, MOCK_API_KEY)
            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args.kwargs
            assert call_kwargs["base_url"] == MOCK_BASE_URL
            assert call_kwargs["headers"]["x-api-key"] == MOCK_API_KEY

    def test_context_manager(self) -> None:
        """Test context manager closes client."""
        with patch("google.adk_community.plugins.goodmem.client.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            with GoodmemClient(MOCK_BASE_URL, MOCK_API_KEY) as client:
                pass

            mock_client.close.assert_called_once()


class TestGoodmemClientTextMemory:
    """Tests for text memory operations."""

    @pytest.fixture
    def mock_httpx_client(self) -> MagicMock:
        """Mock httpx.Client for testing."""
        with patch("google.adk_community.plugins.goodmem.client.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def client(self, mock_httpx_client: MagicMock) -> GoodmemClient:
        """Create GoodmemClient instance for testing."""
        return GoodmemClient(MOCK_BASE_URL, MOCK_API_KEY)

    def test_insert_memory_sends_json(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test that insert_memory sends JSON request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"memoryId": MOCK_MEMORY_ID}
        mock_httpx_client.post.return_value = mock_response

        result = client.insert_memory(
            MOCK_SPACE_ID, "test content", "text/plain", {"key": "value"}
        )

        assert result["memoryId"] == MOCK_MEMORY_ID
        mock_httpx_client.post.assert_called_once()
        call_kwargs = mock_httpx_client.post.call_args.kwargs
        assert call_kwargs["json"]["spaceId"] == MOCK_SPACE_ID
        assert call_kwargs["json"]["originalContent"] == "test content"
        assert call_kwargs["json"]["contentType"] == "text/plain"
        assert call_kwargs["json"]["metadata"] == {"key": "value"}

    def test_insert_memory_without_metadata(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test insert_memory without metadata."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"memoryId": MOCK_MEMORY_ID}
        mock_httpx_client.post.return_value = mock_response

        client.insert_memory(MOCK_SPACE_ID, "test content", "text/plain")

        call_kwargs = mock_httpx_client.post.call_args.kwargs
        assert "metadata" not in call_kwargs["json"]


class TestGoodmemClientBinaryMemory:
    """Tests for binary memory operations - specifically the Content-Type bug fix."""

    @pytest.fixture
    def mock_httpx_client(self) -> MagicMock:
        """Mock httpx.Client for testing."""
        with patch("google.adk_community.plugins.goodmem.client.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def client(self, mock_httpx_client: MagicMock) -> GoodmemClient:
        """Create GoodmemClient instance for testing."""
        return GoodmemClient(MOCK_BASE_URL, MOCK_API_KEY)

    def test_insert_memory_binary_no_content_type_header(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test that insert_memory_binary does NOT set Content-Type header.

        httpx automatically sets Content-Type: multipart/form-data for multipart
        uploads. We must NOT override this with application/json.
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {"memoryId": MOCK_MEMORY_ID}
        mock_httpx_client.post.return_value = mock_response

        client.insert_memory_binary(
            MOCK_SPACE_ID,
            b"test binary content",
            "application/pdf",
        )

        call_kwargs = mock_httpx_client.post.call_args.kwargs
        headers = call_kwargs.get("headers", {})

        # CRITICAL: Content-Type must NOT be set in headers
        # httpx will auto-set multipart/form-data
        assert "Content-Type" not in headers
        assert "content-type" not in headers

    def test_insert_memory_binary_only_api_key_header(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test that only x-api-key is in headers for binary upload."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"memoryId": MOCK_MEMORY_ID}
        mock_httpx_client.post.return_value = mock_response

        client.insert_memory_binary(
            MOCK_SPACE_ID,
            b"test binary content",
            "application/pdf",
        )

        call_kwargs = mock_httpx_client.post.call_args.kwargs
        headers = call_kwargs.get("headers", {})

        # Only x-api-key should be set (passed explicitly).
        assert headers == {}

    def test_insert_memory_binary_uses_full_url(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test that insert_memory_binary constructs full URL."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"memoryId": MOCK_MEMORY_ID}
        mock_httpx_client.post.return_value = mock_response

        client.insert_memory_binary(
            MOCK_SPACE_ID,
            b"test binary content",
            "application/pdf",
        )

        # First positional arg is the URL path (base_url is configured on client)
        call_args = mock_httpx_client.post.call_args
        assert call_args.args[0] == "/v1/memories"

    def test_insert_memory_binary_multipart_structure(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test that insert_memory_binary sends correct multipart structure."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"memoryId": MOCK_MEMORY_ID}
        mock_httpx_client.post.return_value = mock_response

        file_bytes = b"test binary content"
        metadata = {"filename": "test.pdf", "user_id": "user123"}

        client.insert_memory_binary(
            MOCK_SPACE_ID,
            file_bytes,
            "application/pdf",
            metadata,
        )

        call_kwargs = mock_httpx_client.post.call_args.kwargs

        # Check data field (request JSON)
        assert "data" in call_kwargs
        request_json = json.loads(call_kwargs["data"]["request"])
        assert request_json["spaceId"] == MOCK_SPACE_ID
        assert request_json["contentType"] == "application/pdf"
        assert request_json["metadata"] == metadata

        # Check files field (binary content)
        assert "files" in call_kwargs
        files = call_kwargs["files"]
        assert "file" in files
        assert files["file"][0] == "upload"  # filename
        assert files["file"][1] == file_bytes  # content
        assert files["file"][2] == "application/pdf"  # content type

    def test_insert_memory_binary_without_metadata(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test insert_memory_binary without metadata."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"memoryId": MOCK_MEMORY_ID}
        mock_httpx_client.post.return_value = mock_response

        client.insert_memory_binary(
            MOCK_SPACE_ID,
            b"test binary content",
            "application/pdf",
            # No metadata
        )

        call_kwargs = mock_httpx_client.post.call_args.kwargs
        request_json = json.loads(call_kwargs["data"]["request"])
        assert "metadata" not in request_json

    def test_insert_memory_binary_timeout(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test that insert_memory_binary uses longer timeout for large uploads."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"memoryId": MOCK_MEMORY_ID}
        mock_httpx_client.post.return_value = mock_response

        client.insert_memory_binary(
            MOCK_SPACE_ID,
            b"test binary content",
            "application/pdf",
        )

        call_kwargs = mock_httpx_client.post.call_args.kwargs
        assert call_kwargs["timeout"] == 120.0  # Longer timeout for binary

    def test_insert_memory_binary_raises_on_http_error(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test that insert_memory_binary raises on HTTP errors."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request",
            request=MagicMock(),
            response=MagicMock(status_code=400),
        )
        mock_httpx_client.post.return_value = mock_response

        with pytest.raises(httpx.HTTPStatusError):
            client.insert_memory_binary(
                MOCK_SPACE_ID,
                b"test binary content",
                "application/pdf",
            )


class TestGoodmemClientDebugMode:
    """Tests for debug mode."""

    def test_debug_mode_disabled_by_default(self) -> None:
        """Test that debug mode is disabled by default."""
        with patch("google.adk_community.plugins.goodmem.client.httpx.Client"):
            client = GoodmemClient(MOCK_BASE_URL, MOCK_API_KEY)
            assert client._debug is False

    def test_debug_mode_can_be_enabled(self) -> None:
        """Test that debug mode can be enabled."""
        with patch("google.adk_community.plugins.goodmem.client.httpx.Client"):
            client = GoodmemClient(MOCK_BASE_URL, MOCK_API_KEY, debug=True)
            assert client._debug is True


class TestGoodmemClientSpaces:
    """Tests for space operations."""

    @pytest.fixture
    def mock_httpx_client(self) -> MagicMock:
        """Mock httpx.Client for testing."""
        with patch("google.adk_community.plugins.goodmem.client.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def client(self, mock_httpx_client: MagicMock) -> GoodmemClient:
        """Create GoodmemClient instance for testing."""
        return GoodmemClient(MOCK_BASE_URL, MOCK_API_KEY)

    def test_create_space(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test creating a space."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"spaceId": MOCK_SPACE_ID}
        mock_httpx_client.post.return_value = mock_response

        result = client.create_space("test-space", MOCK_EMBEDDER_ID)

        assert result["spaceId"] == MOCK_SPACE_ID
        call_kwargs = mock_httpx_client.post.call_args.kwargs
        assert call_kwargs["json"]["name"] == "test-space"
        assert call_kwargs["json"]["spaceEmbedders"][0]["embedderId"] == MOCK_EMBEDDER_ID

    def test_list_spaces_no_filter(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test listing spaces without filter."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "spaces": [{"spaceId": "s1"}, {"spaceId": "s2"}]
        }
        mock_httpx_client.get.return_value = mock_response

        result = client.list_spaces()

        assert len(result) == 2
        mock_httpx_client.get.assert_called_once()
        call_kwargs = mock_httpx_client.get.call_args.kwargs
        assert "nameFilter" not in call_kwargs["params"]

    def test_list_spaces_with_name_filter(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test listing spaces with name filter."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "spaces": [{"spaceId": "s1", "name": "test-space"}]
        }
        mock_httpx_client.get.return_value = mock_response

        result = client.list_spaces(name="test-space")

        assert len(result) == 1
        call_kwargs = mock_httpx_client.get.call_args.kwargs
        assert call_kwargs["params"]["nameFilter"] == "test-space"

    def test_list_spaces_pagination(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test listing spaces with pagination."""
        # First page returns nextToken
        mock_response1 = MagicMock()
        mock_response1.json.return_value = {
            "spaces": [{"spaceId": "s1"}],
            "nextToken": "token123",
        }
        # Second page returns no nextToken
        mock_response2 = MagicMock()
        mock_response2.json.return_value = {
            "spaces": [{"spaceId": "s2"}],
        }
        mock_httpx_client.get.side_effect = [mock_response1, mock_response2]

        result = client.list_spaces()

        assert len(result) == 2
        assert mock_httpx_client.get.call_count == 2


class TestGoodmemClientRetrieve:
    """Tests for memory retrieval."""

    @pytest.fixture
    def mock_httpx_client(self) -> MagicMock:
        """Mock httpx.Client for testing."""
        with patch("google.adk_community.plugins.goodmem.client.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def client(self, mock_httpx_client: MagicMock) -> GoodmemClient:
        """Create GoodmemClient instance for testing."""
        return GoodmemClient(MOCK_BASE_URL, MOCK_API_KEY)

    def test_retrieve_memories_parses_ndjson(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test that retrieve_memories correctly parses NDJSON response."""
        mock_response = MagicMock()
        ndjson = "\n".join([
            '{"retrievedItem": {"chunk": {"chunk": {"chunkText": "text1"}}}}',
            '{"status": "complete"}',
            '{"retrievedItem": {"chunk": {"chunk": {"chunkText": "text2"}}}}',
        ])
        mock_response.text = ndjson
        mock_httpx_client.post.return_value = mock_response

        result = client.retrieve_memories("query", [MOCK_SPACE_ID])

        # Only items with retrievedItem should be returned
        assert len(result) == 2

    def test_retrieve_memories_sends_correct_payload(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test retrieve_memories sends correct payload."""
        mock_response = MagicMock()
        mock_response.text = ""
        mock_httpx_client.post.return_value = mock_response

        client.retrieve_memories("test query", ["space1", "space2"], request_size=10)

        call_kwargs = mock_httpx_client.post.call_args.kwargs
        assert call_kwargs["json"]["message"] == "test query"
        assert call_kwargs["json"]["requestedSize"] == 10
        assert call_kwargs["json"]["spaceKeys"] == [
            {"spaceId": "space1"},
            {"spaceId": "space2"},
        ]

    def test_get_memory_by_id_url_encodes(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test that get_memory_by_id URL-encodes the memory ID."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"memoryId": "mem/123"}
        mock_httpx_client.get.return_value = mock_response

        client.get_memory_by_id("mem/123")

        call_args = mock_httpx_client.get.call_args
        # / should be encoded as %2F
        assert "%2F" in call_args.args[0]

    def test_get_memories_batch(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test batch get of memories."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "memories": [{"memoryId": "m1"}, {"memoryId": "m2"}]
        }
        mock_httpx_client.post.return_value = mock_response

        result = client.get_memories_batch(["m1", "m2"])

        assert len(result) == 2
        call_kwargs = mock_httpx_client.post.call_args.kwargs
        assert set(call_kwargs["json"]["memoryIds"]) == {"m1", "m2"}

    def test_get_memories_batch_empty_list(
        self, client: GoodmemClient, mock_httpx_client: MagicMock
    ) -> None:
        """Test batch get with empty list doesn't call API."""
        result = client.get_memories_batch([])

        assert result == []
        mock_httpx_client.post.assert_not_called()
