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

"""Unit tests for RedisWorkingMemorySessionService."""

import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.events.event import Event
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.session import Session
from google.genai import types
import pytest
import pytest_asyncio


class TestRedisWorkingMemorySessionServiceConfig:
  """Test cases for RedisWorkingMemorySessionServiceConfig."""

  def test_default_config(self):
    """Test default configuration values."""
    from google.adk_community.sessions import RedisWorkingMemorySessionServiceConfig

    config = RedisWorkingMemorySessionServiceConfig()

    assert config.api_base_url == "http://localhost:8000"
    assert config.timeout == 30.0
    assert config.default_namespace is None
    assert config.model_name is None
    assert config.context_window_max is None
    assert config.extraction_strategy == "discrete"
    assert config.extraction_strategy_config == {}

  def test_custom_config(self):
    """Test custom configuration values."""
    from google.adk_community.sessions import RedisWorkingMemorySessionServiceConfig

    config = RedisWorkingMemorySessionServiceConfig(
        api_base_url="http://custom:9000",
        timeout=60.0,
        default_namespace="my_namespace",
        model_name="gpt-4",
        context_window_max=8000,
        extraction_strategy="summary",
        extraction_strategy_config={"key": "value"},
    )

    assert config.api_base_url == "http://custom:9000"
    assert config.timeout == 60.0
    assert config.default_namespace == "my_namespace"
    assert config.model_name == "gpt-4"
    assert config.context_window_max == 8000
    assert config.extraction_strategy == "summary"
    assert config.extraction_strategy_config == {"key": "value"}


class TestRedisWorkingMemorySessionService:
  """Test cases for RedisWorkingMemorySessionService."""

  @pytest_asyncio.fixture
  async def mock_client(self):
    """Create a mock MemoryAPIClient."""
    mock = AsyncMock()
    mock.close = AsyncMock()
    return mock

  @pytest_asyncio.fixture
  async def service(self, mock_client):
    """Create a RedisWorkingMemorySessionService with mocked client."""
    from google.adk_community.sessions import RedisWorkingMemorySessionService
    from google.adk_community.sessions import RedisWorkingMemorySessionServiceConfig

    config = RedisWorkingMemorySessionServiceConfig(
        api_base_url="http://localhost:8000",
        default_namespace="test_namespace",
    )
    svc = RedisWorkingMemorySessionService(config=config)
    # Inject mock client
    svc.__dict__["_client"] = mock_client
    return svc

  @pytest.mark.asyncio
  async def test_create_session(self, service, mock_client):
    """Test session creation."""
    mock_wm = MagicMock()
    mock_wm.session_id = "generated_id"
    mock_wm.messages = []
    mock_wm.data = {}
    mock_client.get_or_create_working_memory = AsyncMock(
        return_value=(True, mock_wm)
    )
    mock_client.put_working_memory = AsyncMock()

    session = await service.create_session(
        app_name="test_app",
        user_id="test_user",
        state={"key": "value"},
    )

    assert session.app_name == "test_app"
    assert session.user_id == "test_user"
    assert session.state == {"key": "value"}
    assert session.events == []
    assert session.id is not None
    mock_client.get_or_create_working_memory.assert_called_once()
    # put_working_memory called to update state
    mock_client.put_working_memory.assert_called_once()

  @pytest.mark.asyncio
  async def test_create_session_with_custom_id(self, service, mock_client):
    """Test session creation with custom session ID."""
    mock_wm = MagicMock()
    mock_wm.session_id = "custom_session_id"
    mock_wm.messages = []
    mock_wm.data = {}
    mock_client.get_or_create_working_memory = AsyncMock(
        return_value=(True, mock_wm)
    )

    session = await service.create_session(
        app_name="test_app",
        user_id="test_user",
        session_id="custom_session_id",
    )

    assert session.id == "custom_session_id"
    mock_client.get_or_create_working_memory.assert_called_once()

  @pytest.mark.asyncio
  async def test_get_session(self, service, mock_client):
    """Test session retrieval."""
    mock_response = MagicMock()
    mock_response.session_id = "test_session"
    mock_response.messages = []
    mock_response.data = {"key": "value"}
    # Return (created=False, response) to indicate existing session
    mock_client.get_or_create_working_memory = AsyncMock(
        return_value=(False, mock_response)
    )

    session = await service.get_session(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
    )

    assert session is not None
    assert session.id == "test_session"
    assert session.state == {"key": "value"}
    mock_client.get_or_create_working_memory.assert_called_once()

  @pytest.mark.asyncio
  async def test_get_session_not_found(self, service, mock_client):
    """Test session retrieval when session doesn't exist."""
    # Return (created=True, response) to indicate new session was created
    mock_response = MagicMock()
    mock_response.session_id = "nonexistent"
    mock_response.messages = []
    mock_response.data = {}
    mock_client.get_or_create_working_memory = AsyncMock(
        return_value=(True, mock_response)
    )
    mock_client.delete_working_memory = AsyncMock()

    session = await service.get_session(
        app_name="test_app",
        user_id="test_user",
        session_id="nonexistent",
    )

    assert session is None

  @pytest.mark.asyncio
  async def test_list_sessions(self, service, mock_client):
    """Test listing sessions."""
    mock_response = MagicMock()
    mock_response.sessions = ["session1", "session2", "session3"]
    mock_client.list_sessions = AsyncMock(return_value=mock_response)

    result = await service.list_sessions(
        app_name="test_app",
        user_id="test_user",
    )

    assert len(result.sessions) == 3
    assert result.sessions[0].id == "session1"
    assert result.sessions[1].id == "session2"
    assert result.sessions[2].id == "session3"

  @pytest.mark.asyncio
  async def test_delete_session(self, service, mock_client):
    """Test session deletion."""
    mock_client.delete_working_memory = AsyncMock()

    await service.delete_session(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
    )

    mock_client.delete_working_memory.assert_called_once()

  @pytest.mark.asyncio
  async def test_append_event(self, service, mock_client):
    """Test appending an event to a session."""
    mock_client.append_messages_to_working_memory = AsyncMock()

    session = Session(
        id="test_session",
        app_name="test_app",
        user_id="test_user",
        state={},
        events=[],
        last_update_time=time.time(),
    )

    event = Event(
        author="user",
        content=types.Content(parts=[types.Part(text="Hello")]),
        timestamp=time.time(),
    )

    result = await service.append_event(session=session, event=event)

    assert result == event
    mock_client.append_messages_to_working_memory.assert_called_once()

  @pytest.mark.asyncio
  async def test_create_session_existing_returns_existing(
      self, service, mock_client
  ):
    """Test that creating a session with existing ID returns existing session."""
    mock_wm = MagicMock()
    mock_wm.session_id = "existing_session"
    mock_wm.messages = []
    mock_wm.data = {"existing": "data"}
    # created=False means session already exists
    mock_client.get_or_create_working_memory = AsyncMock(
        return_value=(False, mock_wm)
    )

    session = await service.create_session(
        app_name="test_app",
        user_id="test_user",
        session_id="existing_session",
    )

    assert session.id == "existing_session"
    assert session.state == {"existing": "data"}

  @pytest.mark.asyncio
  async def test_close(self, service, mock_client):
    """Test closing the service."""
    await service.close()

    mock_client.close.assert_called_once()
