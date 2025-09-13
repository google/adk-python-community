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
from datetime import datetime, timezone
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk_community.sessions import RedisMemorySessionService
from google.genai import types


class TestRedisMemorySessionService:
    """Test cases for RedisMemorySessionService."""

    @pytest_asyncio.fixture
    async def redis_service(self):
        """Create a Redis session service for testing."""
        with patch("redis.asyncio.Redis") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            service = RedisMemorySessionService()
            service.cache = mock_client
            yield service

    def _setup_redis_mocks(self, redis_service, sessions_data=None):
        """Helper to set up Redis mocks for the new storage strategy."""
        if sessions_data is None:
            sessions_data = {}

        session_ids = list(sessions_data.keys())
        redis_service.cache.smembers = AsyncMock(
            return_value={sid.encode() for sid in session_ids}
        )
        session_values = [
            json.dumps(sessions_data[sid]).encode() if sid in sessions_data else None
            for sid in session_ids
        ]
        redis_service.cache.mget = AsyncMock(return_value=session_values)
        redis_service.cache.srem = AsyncMock()
        redis_service.cache.get = AsyncMock(return_value=None)  # Default to no session

        # Mock pipeline - make sure pipeline() returns the mock_pipe directly, not a coroutine
        mock_pipe = MagicMock()
        mock_pipe.set = MagicMock(return_value=mock_pipe)  # Allow chaining
        mock_pipe.sadd = MagicMock(return_value=mock_pipe)
        mock_pipe.expire = MagicMock(return_value=mock_pipe)
        mock_pipe.delete = MagicMock(return_value=mock_pipe)
        mock_pipe.srem = MagicMock(return_value=mock_pipe)
        mock_pipe.hset = MagicMock(return_value=mock_pipe)
        mock_pipe.execute = AsyncMock(return_value=[])
        redis_service.cache.pipeline = MagicMock(return_value=mock_pipe)

        redis_service.cache.hgetall = AsyncMock(return_value={})
        redis_service.cache.hset = AsyncMock()

    @pytest.mark.asyncio
    async def test_get_empty_session(self, redis_service):
        """Test getting a non-existent session."""
        self._setup_redis_mocks(redis_service)

        session = await redis_service.get_session(
            app_name="test_app", user_id="test_user", session_id="nonexistent"
        )

        assert session is None

    @pytest.mark.asyncio
    async def test_create_get_session(self, redis_service):
        """Test session creation and retrieval."""
        app_name = "test_app"
        user_id = "test_user"
        state = {"key": "value"}

        self._setup_redis_mocks(redis_service)

        session = await redis_service.create_session(
            app_name=app_name, user_id=user_id, state=state
        )

        assert session.app_name == app_name
        assert session.user_id == user_id
        assert session.id is not None
        assert session.state == state
        assert (
            session.last_update_time
            <= datetime.now().astimezone(timezone.utc).timestamp()
        )

        # Mock individual session retrieval
        redis_service.cache.get = AsyncMock(
            return_value=session.model_dump_json().encode()
        )

        got_session = await redis_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session.id
        )

        assert got_session.app_name == session.app_name
        assert got_session.user_id == session.user_id
        assert got_session.id == session.id
        assert got_session.state == session.state

    @pytest.mark.asyncio
    async def test_create_and_list_sessions(self, redis_service):
        """Test creating multiple sessions and listing them."""
        app_name = "test_app"
        user_id = "test_user"

        self._setup_redis_mocks(redis_service)

        session_ids = ["session" + str(i) for i in range(3)]
        sessions_data = {}

        for session_id in session_ids:
            session = await redis_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                state={"key": "value" + session_id},
            )
            sessions_data[session_id] = session.model_dump()

        self._setup_redis_mocks(redis_service, sessions_data)

        list_sessions_response = await redis_service.list_sessions(
            app_name=app_name, user_id=user_id
        )
        sessions = list_sessions_response.sessions

        assert len(sessions) == len(session_ids)
        returned_session_ids = {session.id for session in sessions}
        assert returned_session_ids == set(session_ids)
        for session in sessions:
            # Note: list_sessions removes state for performance
            assert session.state == {}

    @pytest.mark.asyncio
    async def test_session_state_management(self, redis_service):
        """Test session state management with app, user, and temp state."""
        app_name = "test_app"
        user_id = "test_user"
        session_id = "test_session"

        self._setup_redis_mocks(redis_service)

        session = await redis_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state={"initial_key": "initial_value"},
        )

        event = Event(
            invocation_id="invocation",
            author="user",
            content=types.Content(role="user", parts=[types.Part(text="text")]),
            actions=EventActions(
                state_delta={
                    "app:key": "app_value",
                    "user:key1": "user_value",
                    "temp:key": "temp_value",
                    "initial_key": "updated_value",
                }
            ),
        )

        redis_service.cache.get = AsyncMock(
            return_value=session.model_dump_json().encode()
        )

        await redis_service.append_event(session=session, event=event)

        assert session.state.get("app:key") == "app_value"
        assert session.state.get("user:key1") == "user_value"
        assert session.state.get("initial_key") == "updated_value"
        assert session.state.get("temp:key") is None  # Temp state filtered

        redis_service.cache.pipeline().hset.assert_any_call(
            "app:test_app", "key", json.dumps("app_value")
        )
        redis_service.cache.pipeline().hset.assert_any_call(
            "user:test_app:test_user", "key1", json.dumps("user_value")
        )

    @pytest.mark.asyncio
    async def test_append_event_with_bytes(self, redis_service):
        """Test appending events with binary content and serialization roundtrip."""
        app_name = "test_app"
        user_id = "test_user"

        self._setup_redis_mocks(redis_service)

        session = await redis_service.create_session(app_name=app_name, user_id=user_id)

        test_content = types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=b"test_image_data", mime_type="image/png"),
            ],
        )
        test_grounding_metadata = types.GroundingMetadata(
            search_entry_point=types.SearchEntryPoint(sdk_blob=b"test_sdk_blob")
        )
        event = Event(
            invocation_id="invocation",
            author="user",
            content=test_content,
            grounding_metadata=test_grounding_metadata,
        )

        redis_service.cache.get = AsyncMock(
            return_value=session.model_dump_json().encode()
        )

        await redis_service.append_event(session=session, event=event)

        # Verify the event was appended to in-memory session
        assert len(session.events) == 1
        assert session.events[0].content == test_content
        assert session.events[0].grounding_metadata == test_grounding_metadata

        # Test serialization/deserialization roundtrip to ensure binary data is preserved
        # Simulate what happens when session is stored and retrieved from Redis
        serialized_session = session.model_dump_json()

        redis_service.cache.get = AsyncMock(
            return_value=serialized_session.encode()
        )

        retrieved_session = await redis_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session.id
        )

        assert retrieved_session is not None
        assert len(retrieved_session.events) == 1

        # Verify the binary content was preserved through serialization
        retrieved_event = retrieved_session.events[0]
        assert retrieved_event.content.parts[0].inline_data.data == b"test_image_data"
        assert retrieved_event.content.parts[0].inline_data.mime_type == "image/png"
        assert retrieved_event.grounding_metadata.search_entry_point.sdk_blob == b"test_sdk_blob"

    @pytest.mark.asyncio
    async def test_get_session_with_config(self, redis_service):
        """Test getting session with configuration filters."""
        app_name = "test_app"
        user_id = "test_user"

        self._setup_redis_mocks(redis_service)

        session = await redis_service.create_session(app_name=app_name, user_id=user_id)

        # Add multiple events with different timestamps
        num_test_events = 5
        for i in range(1, num_test_events + 1):
            event = Event(author="user", timestamp=float(i))
            session.events.append(event)

        redis_service.cache.get = AsyncMock(
            return_value=session.model_dump_json().encode()
        )

        # Test num_recent_events filter
        config = GetSessionConfig(num_recent_events=3)
        filtered_session = await redis_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session.id, config=config
        )

        assert len(filtered_session.events) == 3
        assert filtered_session.events[0].timestamp == 3.0  # Last 3 events

        # Test after_timestamp filter
        config = GetSessionConfig(after_timestamp=3.0)
        filtered_session = await redis_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session.id, config=config
        )

        assert len(filtered_session.events) == 3  # Events 3, 4, 5
        assert filtered_session.events[0].timestamp == 3.0

    @pytest.mark.asyncio
    async def test_delete_session(self, redis_service):
        """Test session deletion."""
        app_name = "test_app"
        user_id = "test_user"
        session_id = "test_session"

        self._setup_redis_mocks(redis_service)  # Empty sessions
        await redis_service.delete_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        redis_service.cache.pipeline().execute.assert_called()

        redis_service.cache.pipeline().execute.reset_mock()
        self._setup_redis_mocks(redis_service)

        await redis_service.delete_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )

        redis_service.cache.pipeline().execute.assert_called()
