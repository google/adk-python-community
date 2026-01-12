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
from unittest.mock import AsyncMock, MagicMock, patch

from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types
import pytest

from google.adk_community.memory.redis_agent_memory_service import (
    RedisAgentMemoryService,
    RedisAgentMemoryServiceConfig,
)


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
    def __init__(self, session_id, namespace, user_id, messages, long_term_memory_strategy):
        self.session_id = session_id
        self.namespace = namespace
        self.user_id = user_id
        self.messages = messages
        self.long_term_memory_strategy = long_term_memory_strategy


class MockRecencyConfig:
    def __init__(self, recency_boost, semantic_weight, recency_weight,
                 freshness_weight, novelty_weight, half_life_last_access_days,
                 half_life_created_days):
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
                parts=[types.Part(text="Python is a great programming language.")]
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


class TestRedisAgentMemoryServiceConfig:
    """Tests for RedisAgentMemoryServiceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RedisAgentMemoryServiceConfig()
        assert config.api_base_url == "http://localhost:8000"
        assert config.timeout == 30.0
        assert config.default_namespace is None
        assert config.search_top_k == 10
        assert config.distance_threshold is None
        assert config.recency_boost is True
        assert config.semantic_weight == 0.8
        assert config.recency_weight == 0.2
        assert config.freshness_weight == 0.6
        assert config.novelty_weight == 0.4
        assert config.half_life_last_access_days == 7.0
        assert config.half_life_created_days == 30.0
        assert config.extraction_strategy == "discrete"
        assert config.extraction_strategy_config == {}
        assert config.model_name is None
        assert config.context_window_max is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RedisAgentMemoryServiceConfig(
            api_base_url="http://memory-server:9000",
            timeout=60.0,
            default_namespace="my_app",
            search_top_k=20,
            distance_threshold=0.5,
            recency_boost=False,
            semantic_weight=0.7,
            recency_weight=0.3,
            extraction_strategy="summary",
            model_name="gpt-4o",
            context_window_max=128000,
        )
        assert config.api_base_url == "http://memory-server:9000"
        assert config.timeout == 60.0
        assert config.default_namespace == "my_app"
        assert config.search_top_k == 20
        assert config.distance_threshold == 0.5
        assert config.recency_boost is False
        assert config.semantic_weight == 0.7
        assert config.recency_weight == 0.3
        assert config.extraction_strategy == "summary"
        assert config.model_name == "gpt-4o"
        assert config.context_window_max == 128000


class TestRedisAgentMemoryService:
    """Tests for RedisAgentMemoryService."""

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
        """Create RedisAgentMemoryService with mocked client."""
        service = RedisAgentMemoryService()
        service._client = mock_memory_client
        service._client_initialized = True
        return service

    @pytest.fixture
    def memory_service_with_config(self, mock_memory_client):
        """Create RedisAgentMemoryService with custom config."""
        config = RedisAgentMemoryServiceConfig(
            default_namespace="custom_namespace",
            search_top_k=5,
            recency_boost=True,
            extraction_strategy="preferences",
        )
        service = RedisAgentMemoryService(config=config)
        service._client = mock_memory_client
        service._client_initialized = True
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
        assert working_memory.messages[1].content == "Python is a great programming language."

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
    async def test_search_memory_success(self, memory_service, mock_memory_client):
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
        assert result.memories[0].content.parts[0].text == "Python is a great language"

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
        config = RedisAgentMemoryServiceConfig(recency_boost=False)
        service = RedisAgentMemoryService(config=config)
        service._client = mock_memory_client
        service._client_initialized = True

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
        mock_memory_client.search_long_term_memory.side_effect = Exception("API Error")

        result = await memory_service.search_memory(
            app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query="test query"
        )

        assert len(result.memories) == 0

    @pytest.mark.asyncio
    async def test_close(self, memory_service, mock_memory_client):
        """Test closing the service."""
        await memory_service.close()

        mock_memory_client.close.assert_called_once()
        assert memory_service._client is None
        assert memory_service._client_initialized is False

    def test_import_error_handling(self):
        """Test that ImportError is raised when agent-memory-client is not installed."""
        service = RedisAgentMemoryService()

        with patch.dict("sys.modules", {"agent_memory_client": None}):
            with patch(
                "google.adk_community.memory.redis_agent_memory_service.RedisAgentMemoryService._get_client"
            ) as mock_get_client:
                mock_get_client.side_effect = ImportError(
                    "agent-memory-client package is required"
                )
                with pytest.raises(ImportError, match="agent-memory-client"):
                    import asyncio
                    asyncio.get_event_loop().run_until_complete(service._get_client())

