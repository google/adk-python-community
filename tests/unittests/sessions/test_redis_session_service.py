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

import orjson
from datetime import datetime, timezone
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk_community.sessions.redis_session_service import RedisSessionService
from google.adk.cli.fast_api import get_fast_api_app
from google.genai import types


class TestRedisSessionService:
    """Test cases for RedisSessionService."""

    @pytest_asyncio.fixture
    async def redis_service(self):
        """Create a Redis session service for testing."""
        with patch("redis.asyncio.Redis") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            service = RedisSessionService()
            service.cache = mock_client
            yield service

    @pytest_asyncio.fixture
    async def redis_cluster_service(self):
        """Create a Redis cluster session service for testing."""
        with patch("redis.asyncio.RedisCluster.from_url") as mock_redis_cluster:
            mock_client = AsyncMock()
            mock_redis_cluster.return_value = mock_client
            cluster_uri = "redis://redis-node1:6379"
            service = RedisSessionService(cluster_uri=cluster_uri)
            service.cache = mock_client
            yield service

    @pytest_asyncio.fixture
    async def redis_cluster_uri_service(self):
        """Create a Redis cluster session service using URI for testing."""
        with patch("redis.asyncio.RedisCluster.from_url") as mock_redis_cluster:
            mock_client = AsyncMock()
            mock_redis_cluster.return_value = mock_client
            cluster_uri = "redis://node1:6379,node2:6379"
            service = RedisSessionService(cluster_uri=cluster_uri)
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

        # Mock the new cluster-aware pipeline approach
        session_values = [
            orjson.dumps(sessions_data[sid]) if sid in sessions_data else None
            for sid in session_ids
        ]

        # For backward compatibility with mget approach (still used in some tests)
        redis_service.cache.mget = AsyncMock(return_value=session_values)

        # Mock pipeline for the new cluster approach
        if session_ids:
            # Group sessions as the actual implementation does
            results_per_group = []
            for i in range(len(session_ids)):
                results_per_group.append([session_values[i]])

            mock_context_manager = MagicMock()
            mock_pipe = MagicMock()
            mock_pipe.get = MagicMock(return_value=mock_pipe)
            mock_pipe.execute = AsyncMock(side_effect=results_per_group)
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_pipe)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)
            redis_service.cache.pipeline = MagicMock(return_value=mock_context_manager)
        else:
            mock_context_manager = MagicMock()
            mock_pipe = MagicMock()
            mock_pipe.get = MagicMock(return_value=mock_pipe)
            mock_pipe.execute = AsyncMock(return_value=[])
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_pipe)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)
            redis_service.cache.pipeline = MagicMock(return_value=mock_context_manager)

        redis_service.cache.srem = AsyncMock()
        redis_service.cache.get = AsyncMock(return_value=None)  # Default to no session

        # Additional pipeline operations for create/update operations
        if not session_ids:
            mock_context_manager = MagicMock()
            mock_pipe = MagicMock()
            mock_pipe.set = MagicMock(return_value=mock_pipe)  # Allow chaining
            mock_pipe.sadd = MagicMock(return_value=mock_pipe)
            mock_pipe.expire = MagicMock(return_value=mock_pipe)
            mock_pipe.delete = MagicMock(return_value=mock_pipe)
            mock_pipe.srem = MagicMock(return_value=mock_pipe)
            mock_pipe.hset = MagicMock(return_value=mock_pipe)
            mock_pipe.get = MagicMock(return_value=mock_pipe)
            mock_pipe.execute = AsyncMock(return_value=[])
            mock_context_manager.__aenter__ = AsyncMock(return_value=mock_pipe)
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)
            redis_service.cache.pipeline = MagicMock(return_value=mock_context_manager)

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

        # Allow tiny float/clock rounding differences (~1ms)
        assert (
            session.last_update_time
            <= datetime.now().astimezone(timezone.utc).timestamp() + 0.001
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
        """Test creating multiple sessions and listing them.

        list_sessions() is expected to return lightweight session summaries,
        i.e., with events and state stripped for performance.
        """
        app_name = "test_app"
        user_id = "test_user"

        self._setup_redis_mocks(redis_service)

        session_ids = ["session" + str(i) for i in range(3)]
        sessions_data = {}

        for i, session_id in enumerate(session_ids):
            session = await redis_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                state={"key": "value" + session_id},
            )
            # Add at least one event to ensure list_sessions actually strips them.
            session.events.append(Event(author="user", timestamp=float(i + 1)))
            sessions_data[session_id] = session.model_dump()

        # Now mock Redis to return those sessions (with events present in storage)
        self._setup_redis_mocks(redis_service, sessions_data)

        list_sessions_response = await redis_service.list_sessions(
            app_name=app_name, user_id=user_id
        )
        sessions = list_sessions_response.sessions

        assert len(sessions) == len(session_ids)
        returned_session_ids = {s.id for s in sessions}
        assert returned_session_ids == set(session_ids)

        for s in sessions:
            # list_sessions returns summaries: events and state removed for perf.
            assert len(s.events) == 0
            assert s.state == {}

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

        pipeline_mock = redis_service.cache.pipeline.return_value
        pipe_mock = await pipeline_mock.__aenter__()
        pipe_mock.hset.assert_any_call("app:test_app", "key", orjson.dumps("app_value"))
        pipe_mock.hset.assert_any_call(
            "user:test_app:test_user", "key1", orjson.dumps("user_value")
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
                types.Part.from_bytes(
                    data=b"test_image_data", mime_type="image/png"
                ),
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

        redis_service.cache.get = AsyncMock(return_value=serialized_session.encode())

        retrieved_session = await redis_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session.id
        )

        assert retrieved_session is not None
        assert len(retrieved_session.events) == 1

        # Verify the binary content was preserved through serialization
        retrieved_event = retrieved_session.events[0]
        assert retrieved_event.content.parts[0].inline_data.data == b"test_image_data"
        assert (
            retrieved_event.content.parts[0].inline_data.mime_type
            == "image/png"
        )
        assert (
            retrieved_event.grounding_metadata.search_entry_point.sdk_blob
            == b"test_sdk_blob"
        )

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
            app_name=app_name,
            user_id=user_id,
            session_id=session.id,
            config=config,
        )

        assert len(filtered_session.events) == 3
        assert filtered_session.events[0].timestamp == 3.0  # Last 3 events

        # Test after_timestamp filter
        config = GetSessionConfig(after_timestamp=3.0)
        filtered_session = await redis_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session.id,
            config=config,
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
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        pipeline_mock = redis_service.cache.pipeline.return_value
        pipe_mock = await pipeline_mock.__aenter__()
        pipe_mock.execute.assert_called()

        redis_service.cache.pipeline.reset_mock()
        self._setup_redis_mocks(redis_service)

        await redis_service.delete_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )

        pipeline_mock = redis_service.cache.pipeline.return_value
        pipe_mock = await pipeline_mock.__aenter__()
        pipe_mock.execute.assert_called()

    @pytest.mark.asyncio
    async def test_cluster_health_check(self, redis_cluster_service):
        """Test health check for Redis cluster."""
        redis_cluster_service.cache.ping = AsyncMock(return_value=True)

        result = await redis_cluster_service.health_check()
        assert result is True
        redis_cluster_service.cache.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_cluster_health_check_failure(self, redis_cluster_service):
        """Test health check failure for Redis cluster."""
        from redis import RedisError

        redis_cluster_service.cache.ping = AsyncMock(
            side_effect=RedisError("Connection failed")
        )

        result = await redis_cluster_service.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_cluster_create_and_get_session(self, redis_cluster_service):
        """Test session creation and retrieval in cluster mode."""
        app_name = "cluster_test_app"
        user_id = "cluster_test_user"
        state = {"cluster_key": "cluster_value"}

        self._setup_redis_mocks(redis_cluster_service)

        session = await redis_cluster_service.create_session(
            app_name=app_name, user_id=user_id, state=state
        )

        assert session.app_name == app_name
        assert session.user_id == user_id
        assert session.id is not None
        assert session.state == state

        # Mock individual session retrieval
        redis_cluster_service.cache.get = AsyncMock(
            return_value=session.model_dump_json().encode()
        )

        got_session = await redis_cluster_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session.id
        )

        assert got_session.app_name == session.app_name
        assert got_session.user_id == session.user_id
        assert got_session.id == session.id
        assert got_session.state == session.state

    @pytest.mark.asyncio
    async def test_cluster_uri_initialization(self, redis_cluster_uri_service):
        """Test Redis cluster initialization with URI."""
        assert redis_cluster_uri_service.cache is not None

    @pytest.mark.asyncio
    async def test_cluster_error_handling(self, redis_cluster_service):
        """Test error handling in cluster operations."""
        from redis import RedisError

        app_name = "test_app"
        user_id = "test_user"

        # Mock Redis error during session loading
        redis_cluster_service.cache.smembers = AsyncMock(
            side_effect=RedisError("Cluster error")
        )

        sessions_response = await redis_cluster_service.list_sessions(
            app_name=app_name, user_id=user_id
        )

        assert len(sessions_response.sessions) == 0

    @pytest.mark.asyncio
    async def test_cluster_connection_validation(self):
        """Test cluster connection validation during initialization."""
        cluster_uri = "redis://redis-node1:6379"

        with patch("redis.asyncio.RedisCluster.from_url") as mock_redis_cluster:
            mock_client = AsyncMock()
            mock_redis_cluster.return_value = mock_client

            service = RedisSessionService(cluster_uri=cluster_uri)
            assert service.cache is not None
            mock_redis_cluster.assert_called_once()

    @pytest.mark.asyncio
    async def test_cluster_session_cleanup_on_error(self, redis_cluster_service):
        """Test session cleanup when corrupted data is found in cluster."""
        app_name = "test_app"
        user_id = "test_user"

        # Setup mock with corrupted session data
        valid_session_data = {
            "app_name": "test_app",
            "user_id": "test_user",
            "id": "session1",
            "state": {},
            "events": [],
            "last_update_time": 1234567890,
        }
        redis_cluster_service.cache.smembers = AsyncMock(
            return_value={b"session1", b"session2"}
        )

        # Mock the pipeline for cluster approach
        mock_context_manager = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.get = MagicMock(return_value=mock_pipe)
        mock_pipe.execute = AsyncMock(
            side_effect=[
                [orjson.dumps(valid_session_data)],  # session1 result
                [None],  # session2 result (missing)
            ]
        )
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_pipe)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        redis_cluster_service.cache.pipeline = MagicMock(
            return_value=mock_context_manager
        )
        redis_cluster_service.cache.srem = AsyncMock()
        redis_cluster_service.cache.hgetall = AsyncMock(return_value={})

        sessions_response = await redis_cluster_service.list_sessions(
            app_name=app_name, user_id=user_id
        )

        redis_cluster_service.cache.srem.assert_called()
        assert len(sessions_response.sessions) == 1

    @pytest.mark.asyncio
    async def test_decode_responses_handling(self, redis_service):
        """Test proper handling of decode_responses setting."""
        app_name = "test_app"
        user_id = "test_user"
        session_id = "test_session"

        # Test with bytes response (decode_responses=False)
        session_data = (
            '{"app_name": "test_app", "user_id": "test_user", "id": "test_session", '
            '"state": {}, "events": [], "last_update_time": 1234567890}'
        )
        redis_service.cache.get = AsyncMock(return_value=session_data.encode())
        redis_service.cache.hgetall = AsyncMock(return_value={})

        session = await redis_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )

        assert session is not None
        assert session.app_name == app_name
        assert session.user_id == user_id

    @pytest.mark.asyncio
    async def test_get_fast_api_app_with_redis_uri(self, monkeypatch):
        """Test get_fast_api_app integration with Redis URI."""
        from google.adk.cli.service_registry import get_service_registry

        def redis_session_factory(uri: str, **kwargs):
            kwargs_copy = kwargs.copy()
            kwargs_copy.pop("agents_dir", None)
            return RedisSessionService(uri=uri, **kwargs_copy)

        service_registry = get_service_registry()
        monkeypatch.setitem(
            service_registry._session_factories, "redis", redis_session_factory
        )

        with patch("redis.asyncio.Redis.from_url") as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_redis.return_value = mock_client

            redis_uri = "redis://localhost:6379"
            app = get_fast_api_app(
                agents_dir=".",
                session_service_uri=redis_uri,
                web=False,
            )

            assert app is not None
            assert hasattr(app, "routes")

            mock_redis.assert_called_once()
            call_args = mock_redis.call_args
            assert redis_uri in str(call_args)
