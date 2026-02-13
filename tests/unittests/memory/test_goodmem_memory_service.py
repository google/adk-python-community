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

"""Tests for GoodmemMemoryService."""

# pylint: disable=protected-access,unused-argument,too-many-public-methods
# pylint: disable=redefined-outer-name

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from google.adk.events.event import Event
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions.session import Session
from google.adk_community.memory.goodmem.goodmem_memory_service import (
    format_memory_block_for_prompt,
    GoodmemMemoryService,
    GoodmemMemoryServiceConfig,
)

# Mock constants
MOCK_BASE_URL = "https://api.goodmem.ai/v1"
MOCK_API_KEY = "test-api-key"
MOCK_EMBEDDER_ID = "test-embedder-id"
MOCK_SPACE_ID = "test-space-id"
MOCK_SPACE_NAME = "adk_memory_test-app_test-user"
MOCK_APP_NAME = "test-app"
MOCK_USER_ID = "test-user"
MOCK_SESSION_ID = "test-session"
MOCK_MEMORY_ID = "test-memory-id"

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


# ---------------------------------------------------------------------------
# GoodmemMemoryServiceConfig
# ---------------------------------------------------------------------------


class TestGoodmemMemoryServiceConfig:
    """Tests for GoodmemMemoryServiceConfig."""

    def test_default_config(self) -> None:
        config = GoodmemMemoryServiceConfig()
        assert config.top_k == 5
        assert config.timeout == 30.0
        assert config.split_turn is False

    def test_custom_config(self) -> None:
        config = GoodmemMemoryServiceConfig(
            top_k=20,
            timeout=10.0,
            split_turn=True,
        )
        assert config.top_k == 20
        assert config.timeout == 10.0
        assert config.split_turn is True

    def test_config_validation_top_k(self) -> None:
        with pytest.raises(Exception):
            GoodmemMemoryServiceConfig(top_k=0)

        with pytest.raises(Exception):
            GoodmemMemoryServiceConfig(top_k=101)


# ---------------------------------------------------------------------------
# GoodmemMemoryService
# ---------------------------------------------------------------------------

_CLIENT_PATCH = (
    "google.adk_community.memory.goodmem.goodmem_memory_service.GoodmemClient"
)


class TestGoodmemMemoryService:
    """Tests for GoodmemMemoryService."""

    @pytest.fixture
    def mock_goodmem_client(self) -> Generator[MagicMock, None, None]:
        """Mock the shared GoodmemClient."""
        with patch(_CLIENT_PATCH) as mock_cls:
            client = MagicMock()
            client.list_embedders.return_value = [
                {"embedderId": MOCK_EMBEDDER_ID, "name": "Test Embedder"}
            ]
            client.list_spaces.return_value = []
            client.create_space.return_value = {"spaceId": MOCK_SPACE_ID}
            client.insert_memory.return_value = {
                "memoryId": MOCK_MEMORY_ID,
                "processingStatus": "COMPLETED",
            }
            client.insert_memory_binary.return_value = {
                "memoryId": MOCK_MEMORY_ID,
                "processingStatus": "PROCESSING",
            }
            client.retrieve_memories.return_value = []
            mock_cls.return_value = client
            yield client

    @pytest.fixture
    def memory_service(
        self, mock_goodmem_client: MagicMock
    ) -> GoodmemMemoryService:
        return GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
        )

    @pytest.fixture
    def memory_service_with_config(
        self, mock_goodmem_client: MagicMock
    ) -> GoodmemMemoryService:
        config = GoodmemMemoryServiceConfig(top_k=5, timeout=10.0)
        return GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            config=config,
        )

    # -- constructor / lazy init --------------------------------------------

    def test_service_initialization_no_network_call(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """Constructor must not call list_embedders or list_spaces."""
        GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
        )
        mock_goodmem_client.list_embedders.assert_not_called()
        mock_goodmem_client.list_spaces.assert_not_called()

    def test_service_initialization_stores_embedder_arg(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
        )
        assert service._embedder_id_arg == MOCK_EMBEDDER_ID
        assert service._resolved_embedder_id is None

    def test_service_initialization_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="api_key is required"):
            GoodmemMemoryService(base_url=MOCK_BASE_URL, api_key="")

    def test_config_with_custom_timeout(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        config = GoodmemMemoryServiceConfig(timeout=60.0)
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            config=config,
        )
        assert service._config.timeout == 60.0

    # -- embedder resolution ------------------------------------------------

    def test_embedder_resolved_on_first_space_creation(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """Embedder is resolved lazily, not in constructor."""
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
        )
        assert service._resolved_embedder_id is None
        mock_goodmem_client.list_embedders.assert_not_called()

        service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)

        assert service._resolved_embedder_id == MOCK_EMBEDDER_ID
        mock_goodmem_client.list_embedders.assert_called_once()

    def test_embedder_uses_first_available(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """When no embedder_id given, first available is used (deterministic)."""
        mock_goodmem_client.list_embedders.return_value = [
            {"embedderId": "first-emb", "name": "First"},
            {"embedderId": "second-emb", "name": "Second"},
        ]
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
        )

        service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)

        assert service._resolved_embedder_id == "first-emb"
        mock_goodmem_client.create_space.assert_called_once_with(
            MOCK_SPACE_NAME, "first-emb"
        )

    def test_no_embedders_fails_on_first_space(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        """Constructor succeeds; error deferred to first space creation."""
        mock_goodmem_client.list_embedders.return_value = []
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
        )
        with pytest.raises(ValueError, match="No embedders available"):
            service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)

    def test_invalid_embedder_fails_on_first_space(
        self, mock_goodmem_client: MagicMock
    ) -> None:
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id="invalid-embedder-id",
        )
        with pytest.raises(ValueError, match="is not valid"):
            service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)

    # -- space management ---------------------------------------------------

    def test_ensure_space_creates_new_space(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        space_id = memory_service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)

        mock_goodmem_client.list_spaces.assert_called_once_with(
            name=MOCK_SPACE_NAME
        )
        mock_goodmem_client.create_space.assert_called_once_with(
            MOCK_SPACE_NAME, MOCK_EMBEDDER_ID
        )
        assert space_id == MOCK_SPACE_ID
        cache_key = f"{MOCK_APP_NAME}:{MOCK_USER_ID}"
        assert memory_service._space_cache[cache_key] == MOCK_SPACE_ID

    def test_ensure_space_uses_existing_space(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        mock_goodmem_client.list_spaces.return_value = [
            {"spaceId": "existing-space-id", "name": MOCK_SPACE_NAME}
        ]

        space_id = memory_service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)

        mock_goodmem_client.create_space.assert_not_called()
        mock_goodmem_client.list_embedders.assert_not_called()
        assert space_id == "existing-space-id"

    def test_ensure_space_uses_cache(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        cache_key = f"{MOCK_APP_NAME}:{MOCK_USER_ID}"
        memory_service._space_cache[cache_key] = "cached-space-id"

        space_id = memory_service._ensure_space(MOCK_APP_NAME, MOCK_USER_ID)

        mock_goodmem_client.list_spaces.assert_not_called()
        mock_goodmem_client.create_space.assert_not_called()
        assert space_id == "cached-space-id"

    # -- add_session_to_memory ----------------------------------------------

    @pytest.mark.asyncio
    async def test_add_session_to_memory_success(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        await memory_service.add_session_to_memory(MOCK_SESSION)

        mock_goodmem_client.insert_memory.assert_called_once()
        call_kw = mock_goodmem_client.insert_memory.call_args.kwargs

        assert "User: Hello, I like Python." in call_kw["content"]
        assert (
            "LLM: Python is a great programming language."
            in call_kw["content"]
        )
        assert call_kw["space_id"] == MOCK_SPACE_ID
        assert call_kw["metadata"]["app_name"] == MOCK_APP_NAME
        assert call_kw["metadata"]["user_id"] == MOCK_USER_ID
        assert call_kw["metadata"]["session_id"] == MOCK_SESSION_ID
        assert call_kw["metadata"]["source"] == "adk_session"

    @pytest.mark.asyncio
    async def test_add_session_filters_empty_events(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        await memory_service.add_session_to_memory(
            MOCK_SESSION_WITH_EMPTY_EVENTS
        )
        mock_goodmem_client.insert_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_session_error_handling(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        mock_goodmem_client.insert_memory.side_effect = Exception("API Error")
        await memory_service.add_session_to_memory(MOCK_SESSION)
        mock_goodmem_client.insert_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_session_separate_user_llm_memories(
        self,
        mock_goodmem_client: MagicMock,
    ) -> None:
        """With split_turn=True, two memories per turn."""
        config = GoodmemMemoryServiceConfig(
            split_turn=True,
        )
        service = GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
            config=config,
        )
        await service.add_session_to_memory(MOCK_SESSION)

        assert mock_goodmem_client.insert_memory.call_count == 2
        calls = mock_goodmem_client.insert_memory.call_args_list
        user_call = calls[0].kwargs
        llm_call = calls[1].kwargs
        assert user_call["content"] == "User: Hello, I like Python."
        assert user_call["metadata"].get("role") == "user"
        assert llm_call["content"] == "LLM: Python is a great programming language."
        assert llm_call["metadata"].get("role") == "LLM"

    # -- search_memory ------------------------------------------------------

    @pytest.mark.asyncio
    async def test_search_memory_success(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        mock_goodmem_client.retrieve_memories.return_value = [
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkText": (
                                "User: What is Python?\n"
                                "LLM: Python is great"
                            ),
                            "memoryId": "mem-1",
                        }
                    }
                }
            },
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkText": (
                                "User: Do you like coding?\n"
                                "LLM: I like programming"
                            ),
                            "memoryId": "mem-2",
                        }
                    }
                }
            },
        ]

        result = await memory_service.search_memory(
            app_name=MOCK_APP_NAME,
            user_id=MOCK_USER_ID,
            query="Python programming",
        )

        mock_goodmem_client.retrieve_memories.assert_called_once_with(
            query="Python programming",
            space_ids=[MOCK_SPACE_ID],
            request_size=5,
        )

        assert len(result.memories) == 2
        assert "Python is great" in result.memories[0].content.parts[0].text
        assert result.memories[0].author == "conversation"
        assert result.memories[0].id == "mem-1"
        assert (
            "I like programming" in result.memories[1].content.parts[0].text
        )
        assert result.memories[1].id == "mem-2"

    @pytest.mark.asyncio
    async def test_search_memory_respects_top_k(
        self,
        memory_service_with_config: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        await memory_service_with_config.search_memory(
            app_name=MOCK_APP_NAME,
            user_id=MOCK_USER_ID,
            query="test query",
        )

        call_kw = mock_goodmem_client.retrieve_memories.call_args.kwargs
        assert call_kw["request_size"] == 5

    @pytest.mark.asyncio
    async def test_search_memory_error_handling(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        mock_goodmem_client.retrieve_memories.side_effect = Exception(
            "API Error"
        )

        result = await memory_service.search_memory(
            app_name=MOCK_APP_NAME,
            user_id=MOCK_USER_ID,
            query="test query",
        )
        assert len(result.memories) == 0

    @pytest.mark.asyncio
    async def test_search_memory_empty_response(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        result = await memory_service.search_memory(
            app_name=MOCK_APP_NAME,
            user_id=MOCK_USER_ID,
            query="test query",
        )
        assert len(result.memories) == 0

    # -- close --------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_close_calls_client_close(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        await memory_service.close()
        mock_goodmem_client.close.assert_called_once()

    # -- full flow ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_full_memory_flow(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        # Add session
        await memory_service.add_session_to_memory(MOCK_SESSION)
        mock_goodmem_client.insert_memory.assert_called_once()

        # Search
        mock_goodmem_client.retrieve_memories.return_value = [
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkText": (
                                "User: Hello\nLLM: I like Python."
                            ),
                            "memoryId": "mem-1",
                        }
                    }
                }
            }
        ]

        result = await memory_service.search_memory(
            app_name=MOCK_APP_NAME,
            user_id=MOCK_USER_ID,
            query="Python",
        )

        mock_goodmem_client.retrieve_memories.assert_called_once()
        assert len(result.memories) == 1
        assert "Python" in result.memories[0].content.parts[0].text


# ---------------------------------------------------------------------------
# Binary attachments via add_session_to_memory
# ---------------------------------------------------------------------------


class TestSessionWithBinaryAttachments:
    """Tests for add_session_to_memory with PDF/image attachments."""

    @pytest.fixture
    def mock_goodmem_client(self) -> Generator[MagicMock, None, None]:
        with patch(_CLIENT_PATCH) as mock_cls:
            client = MagicMock()
            client.list_embedders.return_value = [
                {"embedderId": MOCK_EMBEDDER_ID, "name": "Test Embedder"}
            ]
            client.list_spaces.return_value = []
            client.create_space.return_value = {"spaceId": MOCK_SPACE_ID}
            client.insert_memory.return_value = {
                "memoryId": MOCK_MEMORY_ID,
                "processingStatus": "COMPLETED",
            }
            client.insert_memory_binary.return_value = {
                "memoryId": MOCK_MEMORY_ID,
                "processingStatus": "PROCESSING",
            }
            mock_cls.return_value = client
            yield client

    @pytest.fixture
    def memory_service(
        self, mock_goodmem_client: MagicMock
    ) -> GoodmemMemoryService:
        return GoodmemMemoryService(
            base_url=MOCK_BASE_URL,
            api_key=MOCK_API_KEY,
            embedder_id=MOCK_EMBEDDER_ID,
        )

    @pytest.mark.asyncio
    async def test_session_with_pdf_attachment_only(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        """User uploads PDF without text; LLM responds."""
        pdf_blob = types.Blob(
            data=b"%PDF-1.4 fake pdf content",
            mime_type="application/pdf",
        )
        pdf_blob.display_name = "document.pdf"

        session = Session(
            app_name=MOCK_APP_NAME,
            user_id=MOCK_USER_ID,
            id=MOCK_SESSION_ID,
            last_update_time=1000,
            events=[
                Event(
                    id="event-pdf",
                    invocation_id="inv-1",
                    author="user",
                    timestamp=12345,
                    content=types.Content(
                        parts=[types.Part(inline_data=pdf_blob)]
                    ),
                ),
                Event(
                    id="event-response",
                    invocation_id="inv-1",
                    author="model",
                    timestamp=12346,
                    content=types.Content(
                        parts=[
                            types.Part(
                                text="This PDF contains information about..."
                            )
                        ]
                    ),
                ),
            ],
        )

        await memory_service.add_session_to_memory(session)

        # Binary attachment saved via shared client.
        mock_goodmem_client.insert_memory_binary.assert_called_once()
        bin_kw = mock_goodmem_client.insert_memory_binary.call_args.kwargs
        assert bin_kw["content_bytes"] == b"%PDF-1.4 fake pdf content"
        assert bin_kw["content_type"] == "application/pdf"
        assert bin_kw["metadata"]["filename"] == "document.pdf"

        # LLM response saved as text (no user text prefix).
        mock_goodmem_client.insert_memory.assert_called_once()
        txt_kw = mock_goodmem_client.insert_memory.call_args.kwargs
        assert "LLM: This PDF contains information about" in txt_kw["content"]

    @pytest.mark.asyncio
    async def test_session_with_image_attachment_and_text(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        """User uploads image with a text question."""
        image_blob = types.Blob(
            data=b"\x89PNG\r\n\x1a\n fake png",
            mime_type="image/png",
        )
        image_blob.display_name = "screenshot.png"

        session = Session(
            app_name=MOCK_APP_NAME,
            user_id=MOCK_USER_ID,
            id=MOCK_SESSION_ID,
            last_update_time=1000,
            events=[
                Event(
                    id="event-upload",
                    invocation_id="inv-1",
                    author="user",
                    timestamp=12345,
                    content=types.Content(
                        parts=[
                            types.Part(inline_data=image_blob),
                            types.Part(text="What is in this image?"),
                        ]
                    ),
                ),
                Event(
                    id="event-response",
                    invocation_id="inv-1",
                    author="model",
                    timestamp=12346,
                    content=types.Content(
                        parts=[
                            types.Part(text="The image shows a chart.")
                        ]
                    ),
                ),
            ],
        )

        await memory_service.add_session_to_memory(session)

        # Image saved as binary.
        mock_goodmem_client.insert_memory_binary.assert_called_once()
        bin_kw = mock_goodmem_client.insert_memory_binary.call_args.kwargs
        assert bin_kw["content_type"] == "image/png"

        # Text conversation paired.
        mock_goodmem_client.insert_memory.assert_called_once()
        txt_kw = mock_goodmem_client.insert_memory.call_args.kwargs
        assert "User: What is in this image?" in txt_kw["content"]
        assert "LLM: The image shows a chart." in txt_kw["content"]

    @pytest.mark.asyncio
    async def test_session_with_multiple_attachments(
        self,
        memory_service: GoodmemMemoryService,
        mock_goodmem_client: MagicMock,
    ) -> None:
        """Multiple attachments in a single user event."""
        pdf_blob = types.Blob(
            data=b"%PDF-1.4 pdf1", mime_type="application/pdf"
        )
        pdf_blob.display_name = "doc1.pdf"

        img_blob = types.Blob(
            data=b"\xff\xd8\xff jpeg", mime_type="image/jpeg"
        )
        img_blob.display_name = "photo.jpg"

        session = Session(
            app_name=MOCK_APP_NAME,
            user_id=MOCK_USER_ID,
            id=MOCK_SESSION_ID,
            last_update_time=1000,
            events=[
                Event(
                    id="event-uploads",
                    invocation_id="inv-1",
                    author="user",
                    timestamp=12345,
                    content=types.Content(
                        parts=[
                            types.Part(inline_data=pdf_blob),
                            types.Part(inline_data=img_blob),
                        ]
                    ),
                ),
                Event(
                    id="event-response",
                    invocation_id="inv-1",
                    author="model",
                    timestamp=12346,
                    content=types.Content(
                        parts=[types.Part(text="I see two files.")]
                    ),
                ),
            ],
        )

        await memory_service.add_session_to_memory(session)

        # Both attachments saved.
        assert mock_goodmem_client.insert_memory_binary.call_count == 2

        # LLM response saved as text.
        mock_goodmem_client.insert_memory.assert_called_once()


# ---------------------------------------------------------------------------
# format_memory_block_for_prompt
# ---------------------------------------------------------------------------


class TestFormatMemoryBlockForPrompt:
    """Tests for format_memory_block_for_prompt."""

    def test_empty_response(self) -> None:
        """Empty response still produces header and footer."""
        response = SearchMemoryResponse(memories=[])
        block = format_memory_block_for_prompt(response)
        assert "BEGIN MEMORY" in block
        assert "END MEMORY" in block
        assert "RETRIEVED MEMORIES:" in block
        assert "Usage rules:" in block

    def test_one_chunk_with_timestamp(self) -> None:
        """One memory with timestamp produces id, time, content."""
        entry = MemoryEntry(
            id="mem-123",
            content=types.Content(
                parts=[types.Part(text="User: My favorite color is blue.\nLLM: I'll remember.")]
            ),
            timestamp="2025-02-05 14:30",
        )
        response = SearchMemoryResponse(memories=[entry])
        block = format_memory_block_for_prompt(response)
        assert "BEGIN MEMORY" in block
        assert "END MEMORY" in block
        assert "- id: mem-123" in block
        assert "  time: 2025-02-05 14:30" in block
        assert "User: My favorite color is blue." in block
        assert "LLM: I'll remember." in block
        assert "role:" not in block

    def test_chunk_without_timestamp(self) -> None:
        """Chunk without timestamp omits time line."""
        entry = MemoryEntry(
            id="mem-456",
            content=types.Content(parts=[types.Part(text="User: Hello.")]),
            timestamp=None,
        )
        response = SearchMemoryResponse(memories=[entry])
        block = format_memory_block_for_prompt(response)
        assert "- id: mem-456" in block
        assert "  content: |" in block
        assert "User: Hello." in block
        # No time line when timestamp is None
        assert "  time: " not in block

    def test_multiple_chunks(self) -> None:
        """Multiple memories appear in order."""
        entries = [
            MemoryEntry(
                id="mem-a",
                content=types.Content(parts=[types.Part(text="User: A.\nLLM: B.")]),
                timestamp="2025-02-05 14:30",
            ),
            MemoryEntry(
                id="mem-b",
                content=types.Content(parts=[types.Part(text="User: C.")]),
                timestamp="2025-02-05 14:32",
            ),
        ]
        response = SearchMemoryResponse(memories=entries)
        block = format_memory_block_for_prompt(response)
        assert block.index("mem-a") < block.index("mem-b")
        assert "  time: 2025-02-05 14:30" in block
        assert "  time: 2025-02-05 14:32" in block
