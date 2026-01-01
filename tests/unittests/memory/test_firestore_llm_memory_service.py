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
import json
import pytest

from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.adk_community.memory.firestore_llm_memory_service import (
    FirestoreLLMMemoryService,
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
            content=types.Content(parts=[types.Part(text="I love hiking.")]),
        ),
    ],
)


@pytest.fixture
def mock_auth():
    with patch("google.auth.default") as mock_default:
        mock_default.return_value = (MagicMock(), "test-project")
        yield mock_default


@pytest.fixture
def mock_firestore():
    with patch("google.cloud.firestore.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_collection = MagicMock()
        mock_document = MagicMock()
        mock_subcollection = MagicMock()

        mock_client.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_document
        mock_document.collection.return_value = mock_subcollection
        mock_subcollection.order_by.return_value = mock_subcollection
        mock_subcollection.limit.return_value = mock_subcollection

        # Batch mock
        mock_batch = MagicMock()
        mock_batch.commit = AsyncMock()
        mock_client.batch.return_value = mock_batch

        yield mock_client


@pytest.fixture
def service(mock_auth, mock_firestore):
    with patch(
        "google.adk.agents.llm_agent.Agent.canonical_model",
        new_callable=MagicMock,
    ) as mock_model:
        mock_model.model = "gemini-2.0-flash"
        service = FirestoreLLMMemoryService(collection_name="test_facts")
        yield service


class TestFirestoreLLMMemoryService:
    def test_init(self, mock_auth, mock_firestore):
        service = FirestoreLLMMemoryService(
            collection_name="custom_facts", model="gemini-2.0-flash"
        )
        assert service.collection_name == "custom_facts"
        assert service._memory_agent.model == "gemini-2.0-flash"

    @pytest.mark.asyncio
    async def test_add_session_to_memory_success(self, service, mock_firestore):
        # 1. Mock existing facts in Firestore
        mock_doc = MagicMock()
        mock_doc.id = "fact-1"
        mock_doc.to_dict.return_value = {"text": "Old fact"}

        async def mock_stream():
            yield mock_doc

        mock_subcollection = (
            mock_firestore.collection.return_value.document.return_value.collection.return_value
        )
        mock_subcollection.stream.return_value = mock_stream()

        # 2. Mock LLM Response
        llm_response_json = {
            "add": ["Likes hiking"],
            "update": [{"id": "fact-1", "text": "Actually dislikes hiking"}],
            "delete": [],
        }

        mock_response = MagicMock()
        mock_response.partial = False
        mock_response.content.parts = [types.Part(text=json.dumps(llm_response_json))]

        async def mock_generate(*args, **kwargs):
            yield mock_response

        service._memory_agent.canonical_model.generate_content_async.side_effect = (
            mock_generate
        )

        await service.add_session_to_memory(MOCK_SESSION)

        # 3. Verify Batch Operations
        mock_batch = mock_firestore.batch.return_value
        assert mock_batch.set.call_count == 1
        assert mock_batch.update.call_count == 1
        assert mock_batch.delete.call_count == 0
        mock_batch.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_memory_success(self, service, mock_firestore):
        # 1. Mock stored facts
        mock_doc = MagicMock()
        mock_doc.id = "fact-1"
        mock_doc.to_dict.return_value = {
            "text": "User likes hiking",
            "timestamp": MagicMock(timestamp=lambda: 1234567.0),
            "source_session_id": "session-1",
        }

        async def mock_stream():
            yield mock_doc

        # Setup the subcollection mock chain
        mock_subcollection = MagicMock()
        mock_firestore.collection.return_value.document.return_value.collection.return_value = (
            mock_subcollection
        )
        mock_subcollection.order_by.return_value = mock_subcollection
        mock_subcollection.limit.return_value = mock_subcollection
        mock_subcollection.stream.return_value = mock_stream()

        # 2. Mock LLM Filtering Response
        mock_response = MagicMock()
        mock_response.partial = False
        mock_response.content.parts = [types.Part(text='["fact-1"]')]

        async def mock_generate(*args, **kwargs):
            yield mock_response

        service._memory_agent.canonical_model.generate_content_async.side_effect = (
            mock_generate
        )

        response = await service.search_memory(
            app_name=MOCK_APP_NAME,
            user_id=MOCK_USER_ID,
            query="What does the user like?",
        )

        assert len(response.memories) == 1
        assert response.memories[0].content.parts[0].text == "User likes hiking"
        assert response.memories[0].author == "memory_manager"

    @pytest.mark.asyncio
    async def test_add_session_malformed_json(self, service, mock_firestore):
        # Mock LLM returning garbage
        mock_response = MagicMock()
        mock_response.partial = False
        mock_response.content.parts = [types.Part(text="Not JSON")]

        async def mock_generate(*args, **kwargs):
            yield mock_response

        service._memory_agent.canonical_model.generate_content_async.side_effect = (
            mock_generate
        )

        # Should not raise exception
        await service.add_session_to_memory(MOCK_SESSION)

        mock_batch = mock_firestore.batch.return_value
        assert mock_batch.commit.call_count == 0

    @pytest.mark.asyncio
    async def test_add_session_structurally_invalid_json(self, service, mock_firestore):
        # Mock LLM returning valid JSON but with missing/incorrect keys
        mock_response = MagicMock()
        mock_response.partial = False
        # Case: missing 'text' in update
        mock_response.content.parts = [
            types.Part(text='{"update": [{"id": "fact-1"}]}')
        ]

        async def mock_generate(*args, **kwargs):
            yield mock_response

        service._memory_agent.canonical_model.generate_content_async.side_effect = (
            mock_generate
        )

        # The method should handle this gracefully without crashing
        await service.add_session_to_memory(MOCK_SESSION)

        # Verify that no commit happened because no valid operations were found
        mock_batch = mock_firestore.batch.return_value
        assert mock_batch.commit.call_count == 0
