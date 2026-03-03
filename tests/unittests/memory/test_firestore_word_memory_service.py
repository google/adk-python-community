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

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.adk_community.memory.firestore_word_memory_service import (
    FirestoreWordMemoryService,
)
from google.genai import types

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
            author="user",
            timestamp=12345,
            content=types.Content(parts=[types.Part(text="Hello, I like Python.")]),
        ),
        Event(
            id="event-2",
            author="agent",
            timestamp=12346,
            content=types.Content(
                parts=[types.Part(text="Python is a great programming language.")]
            ),
        ),
        # Empty event, should be ignored
        Event(
            id="event-3",
            author="user",
            timestamp=12347,
        ),
    ],
)


@pytest.fixture
def mock_firestore_class():
    with patch("google.cloud.firestore.AsyncClient") as mock_client_class:
        yield mock_client_class


@pytest.fixture
def mock_firestore(mock_firestore_class):
    mock_client = MagicMock()
    mock_firestore_class.return_value = mock_client

    # Chainable mocks
    mock_collection = MagicMock()
    mock_document = MagicMock()
    mock_subcollection = MagicMock()

    mock_client.collection.return_value = mock_collection
    mock_collection.document.return_value = mock_document
    mock_document.collection.return_value = mock_subcollection
    mock_subcollection.where.return_value = mock_subcollection
    mock_subcollection.limit.return_value = mock_subcollection

    # Batch mock
    mock_batch = MagicMock()
    mock_batch.commit = AsyncMock()
    mock_client.batch.return_value = mock_batch

    yield mock_client


@pytest.fixture
def mock_auth():
    with patch("google.auth.default") as mock_default:
        mock_default.return_value = (MagicMock(), "test-project")
        yield mock_default


@pytest.fixture
def service(mock_auth, mock_firestore):
    return FirestoreWordMemoryService(
        collection_name="test_memories", database="test-db"
    )


class TestFirestoreWordMemoryService:
    def test_init(self, mock_auth, mock_firestore_class):
        service = FirestoreWordMemoryService(
            collection_name="custom_col", database="custom_db"
        )
        assert service.collection_name == "custom_col"
        mock_firestore_class.assert_called_once()
        # Check if database was passed to AsyncClient
        _, kwargs = mock_firestore_class.call_args
        assert kwargs["database"] == "custom_db"
        assert kwargs["project"] == "test-project"

    @pytest.mark.asyncio
    async def test_add_session_to_memory(self, service, mock_firestore):
        await service.add_session_to_memory(MOCK_SESSION)

        mock_collection = mock_firestore.collection
        mock_collection.assert_called_with("test_memories")

        user_key = f"{MOCK_APP_NAME}:{MOCK_USER_ID}"
        mock_collection.return_value.document.assert_called_with(user_key)

        mock_subcollection = (
            mock_collection.return_value.document.return_value.collection
        )
        mock_subcollection.assert_called_with("events")

        mock_batch = mock_firestore.batch.return_value
        # Should be 2 calls for the 2 events with content
        assert mock_batch.set.call_count == 2
        mock_batch.commit.assert_called_once()

        # Verify first event data
        call_args = mock_batch.set.call_args_list[0]
        event_data = call_args.args[1]
        assert event_data["session_id"] == MOCK_SESSION_ID
        assert event_data["author"] == "user"
        assert "python" in event_data["words"]
        assert "hello" in event_data["words"]
        assert event_data["content"]["parts"][0]["text"] == "Hello, I like Python."

    @pytest.mark.asyncio
    async def test_search_memory_empty_query(self, service):
        response = await service.search_memory(
            app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query=""
        )
        assert len(response.memories) == 0

    @pytest.mark.asyncio
    async def test_search_memory_success(self, service, mock_firestore):
        # Setup mock for stream()
        mock_query = MagicMock()
        mock_query.limit.return_value = mock_query

        # Create mock document snapshots
        mock_doc1 = MagicMock()
        mock_doc1.to_dict.return_value = {
            "content": {"parts": [{"text": "Python is fun"}]},
            "author": "user",
            "timestamp": 12345,
        }

        async def mock_stream():
            yield mock_doc1

        mock_query.stream.return_value = mock_stream()

        # Setup chain for search
        mock_collection = mock_firestore.collection
        mock_document = mock_collection.return_value.document
        mock_events = mock_document.return_value.collection
        mock_events.return_value.where.return_value = mock_query

        response = await service.search_memory(
            app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="python"
        )

        assert len(response.memories) == 1
        assert response.memories[0].content.parts[0].text == "Python is fun"
        assert response.memories[0].author == "user"

        # Verify query was constructed correctly
        mock_events.return_value.where.assert_called_once()
        _, kwargs = mock_events.return_value.where.call_args
        filter_obj = kwargs["filter"]
        # In some versions of firestore, these are private or stored in _to_pb()
        # For testing, we just check that it's a FieldFilter
        from google.cloud.firestore_v1.base_query import FieldFilter

        assert isinstance(filter_obj, FieldFilter)
