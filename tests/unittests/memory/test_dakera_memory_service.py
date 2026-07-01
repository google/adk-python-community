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
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types
import pytest

from google.adk_community.memory.dakera_memory_service import DakeraMemoryService
from google.adk_community.memory.dakera_memory_service import DakeraMemoryServiceConfig

MOCK_APP_NAME = 'test-app'
MOCK_USER_ID = 'test-user'
MOCK_SESSION_ID = 'session-1'
MOCK_NAMESPACE = f'{MOCK_APP_NAME}:{MOCK_USER_ID}'

MOCK_SESSION = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id=MOCK_SESSION_ID,
    last_update_time=1000,
    events=[
        Event(
            id='event-1',
            invocation_id='inv-1',
            author='user',
            timestamp=12345,
            content=types.Content(
                parts=[types.Part(text='Hello, I like Python.')]
            ),
        ),
        Event(
            id='event-2',
            invocation_id='inv-2',
            author='model',
            timestamp=12346,
            content=types.Content(
                parts=[
                    types.Part(text='Python is a great programming language.')
                ]
            ),
        ),
        # Empty event, should be ignored
        Event(
            id='event-3',
            invocation_id='inv-3',
            author='user',
            timestamp=12347,
        ),
        # Function call event, should be ignored
        Event(
            id='event-4',
            invocation_id='inv-4',
            author='agent',
            timestamp=12348,
            content=types.Content(
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name='test_function')
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
def mock_httpx_client():
  """Mock httpx.AsyncClient for testing."""
  with patch(
      'google.adk_community.memory.dakera_memory_service.httpx.AsyncClient'
  ) as mock_client_class:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {'memories': []}
    mock_response.raise_for_status = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_class.return_value = mock_client
    yield mock_client


@pytest.fixture
def memory_service(mock_httpx_client):
  """Create DakeraMemoryService instance for testing."""
  return DakeraMemoryService(
      base_url='http://localhost:3000', api_key='dk-test'
  )


@pytest.fixture
def memory_service_with_config(mock_httpx_client):
  """Create DakeraMemoryService with custom config."""
  config = DakeraMemoryServiceConfig(
      search_top_k=5,
      user_content_importance=0.9,
      model_content_importance=0.6,
  )
  return DakeraMemoryService(
      base_url='http://localhost:3000', api_key='dk-test', config=config
  )


class TestDakeraMemoryServiceConfig:
  """Tests for DakeraMemoryServiceConfig."""

  def test_default_config(self):
    """Test default configuration values."""
    config = DakeraMemoryServiceConfig()
    assert config.search_top_k == 10
    assert config.timeout == 30.0
    assert config.user_content_importance == 0.8
    assert config.model_content_importance == 0.7
    assert config.default_importance == 0.6
    assert config.min_importance is None
    assert config.memory_type == 'episodic'
    assert config.enable_metadata_tags is True

  def test_custom_config(self):
    """Test custom configuration values."""
    config = DakeraMemoryServiceConfig(
        search_top_k=20,
        timeout=10.0,
        user_content_importance=0.9,
        model_content_importance=0.75,
        default_importance=0.5,
        min_importance=0.3,
        memory_type='semantic',
        enable_metadata_tags=False,
    )
    assert config.search_top_k == 20
    assert config.timeout == 10.0
    assert config.user_content_importance == 0.9
    assert config.model_content_importance == 0.75
    assert config.default_importance == 0.5
    assert config.min_importance == 0.3
    assert config.memory_type == 'semantic'
    assert config.enable_metadata_tags is False

  def test_config_validation_search_top_k(self):
    """Test search_top_k validation."""
    with pytest.raises(Exception):  # Pydantic validation error
      DakeraMemoryServiceConfig(search_top_k=0)

    with pytest.raises(Exception):
      DakeraMemoryServiceConfig(search_top_k=101)

  def test_config_validation_importance_bounds(self):
    """Test importance values are clamped to [0.0, 1.0]."""
    with pytest.raises(Exception):
      DakeraMemoryServiceConfig(default_importance=1.5)

    with pytest.raises(Exception):
      DakeraMemoryServiceConfig(min_importance=-0.1)


class TestDakeraMemoryServiceInit:
  """Tests for DakeraMemoryService initialization."""

  def test_api_key_required(self, monkeypatch):
    """Test that an API key is required."""
    monkeypatch.delenv('DAKERA_API_KEY', raising=False)
    with pytest.raises(ValueError, match='api_key is required'):
      DakeraMemoryService(base_url='http://localhost:3000', api_key='')

  def test_api_key_required_no_env(self, monkeypatch):
    """Test that a missing API key with no env var raises."""
    monkeypatch.delenv('DAKERA_API_KEY', raising=False)
    with pytest.raises(ValueError, match='api_key is required'):
      DakeraMemoryService(base_url='http://localhost:3000')

  def test_env_var_fallback(self, monkeypatch):
    """Test base_url and api_key fall back to environment variables."""
    monkeypatch.setenv('DAKERA_API_URL', 'http://dakera.internal:3000/')
    monkeypatch.setenv('DAKERA_API_KEY', 'dk-env')
    service = DakeraMemoryService()
    assert (
        service._base_url == 'http://dakera.internal:3000'
    )  # trailing / stripped
    assert service._api_key == 'dk-env'

  def test_default_base_url(self, monkeypatch):
    """Test the default base URL is the Dakera port 3000."""
    monkeypatch.delenv('DAKERA_API_URL', raising=False)
    service = DakeraMemoryService(api_key='dk-test')
    assert service._base_url == 'http://localhost:3000'


class TestDakeraMemoryService:
  """Tests for DakeraMemoryService."""

  @pytest.mark.asyncio
  async def test_add_session_to_memory_success(
      self, memory_service, mock_httpx_client
  ):
    """Test successful addition of session memories."""
    await memory_service.add_session_to_memory(MOCK_SESSION)

    # Should make 2 POST calls (one per valid event)
    assert mock_httpx_client.post.call_count == 2

    # First call (user event) hits the store endpoint with correct fields.
    call_args = mock_httpx_client.post.call_args_list[0]
    assert call_args.args[0].endswith('/v1/memory/store')
    request_data = call_args.kwargs['json']
    assert '[Author: user' in request_data['content']
    assert 'Hello, I like Python.' in request_data['content']
    assert request_data['agent_id'] == MOCK_NAMESPACE
    assert request_data['session_id'] == MOCK_SESSION_ID
    assert request_data['memory_type'] == 'episodic'
    assert 'session:session-1' in request_data['tags']
    assert request_data['metadata']['author'] == 'user'
    assert request_data['importance'] == 0.8  # User content importance

    # Second call (model event).
    call_args = mock_httpx_client.post.call_args_list[1]
    request_data = call_args.kwargs['json']
    assert '[Author: model' in request_data['content']
    assert 'Python is a great programming language.' in request_data['content']
    assert request_data['metadata']['author'] == 'model'
    assert request_data['importance'] == 0.7  # Model content importance

  @pytest.mark.asyncio
  async def test_add_session_filters_empty_events(
      self, memory_service, mock_httpx_client
  ):
    """Test that events without content are filtered out."""
    await memory_service.add_session_to_memory(MOCK_SESSION_WITH_EMPTY_EVENTS)
    assert mock_httpx_client.post.call_count == 0

  @pytest.mark.asyncio
  async def test_add_session_uses_config_importance(
      self, memory_service_with_config, mock_httpx_client
  ):
    """Test that importance values from config are used."""
    await memory_service_with_config.add_session_to_memory(MOCK_SESSION)

    call_args = mock_httpx_client.post.call_args_list[0]
    assert call_args.kwargs['json']['importance'] == 0.9  # Custom user value

    call_args = mock_httpx_client.post.call_args_list[1]
    assert call_args.kwargs['json']['importance'] == 0.6  # Custom model value

  @pytest.mark.asyncio
  async def test_add_session_without_metadata_tags(self, mock_httpx_client):
    """Test adding memories without metadata tags."""
    config = DakeraMemoryServiceConfig(enable_metadata_tags=False)
    memory_service = DakeraMemoryService(
        base_url='http://localhost:3000', api_key='dk-test', config=config
    )

    await memory_service.add_session_to_memory(MOCK_SESSION)

    call_args = mock_httpx_client.post.call_args_list[0]
    request_data = call_args.kwargs['json']
    assert request_data.get('tags', []) == []

  @pytest.mark.asyncio
  async def test_add_session_error_handling(
      self, memory_service, mock_httpx_client
  ):
    """Test error handling during memory addition."""
    mock_httpx_client.post.side_effect = Exception('API Error')

    # Should not raise, just log the error.
    await memory_service.add_session_to_memory(MOCK_SESSION)
    assert mock_httpx_client.post.call_count == 2

  @pytest.mark.asyncio
  async def test_search_memory_success(self, memory_service, mock_httpx_client):
    """Test successful memory recall."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'memories': [
            {
                'memory': {
                    'id': 'mem-1',
                    'content': (
                        '[Author: user, Time: 2025-01-01T00:00:00] Python is'
                        ' great'
                    ),
                },
                'score': 0.9,
            },
            {
                'memory': {
                    'id': 'mem-2',
                    'content': (
                        '[Author: model, Time: 2025-01-01T00:01:00] I like'
                        ' programming'
                    ),
                },
                'score': 0.8,
            },
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='Python programming'
    )

    # Verify the recall API call.
    call_args = mock_httpx_client.post.call_args
    assert call_args.args[0].endswith('/v1/memory/recall')
    request_data = call_args.kwargs['json']
    assert request_data['query'] == 'Python programming'
    assert request_data['top_k'] == 10
    assert request_data['agent_id'] == MOCK_NAMESPACE

    # Verify results (content cleaned of the metadata prefix).
    assert len(result.memories) == 2
    assert result.memories[0].content.parts[0].text == 'Python is great'
    assert result.memories[0].author == 'user'
    assert result.memories[1].content.parts[0].text == 'I like programming'
    assert result.memories[1].author == 'model'

  @pytest.mark.asyncio
  async def test_search_memory_scopes_to_namespace(
      self, memory_service, mock_httpx_client
  ):
    """Test that recall is scoped to the app/user namespace."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'memories': [{
            'memory': {'content': 'plain content without prefix'},
            'score': 1.0,
        }]
    }
    mock_response.raise_for_status = MagicMock()
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='test query'
    )

    request_data = mock_httpx_client.post.call_args.kwargs['json']
    assert request_data['agent_id'] == MOCK_NAMESPACE
    # Content with no enriched prefix passes through unchanged.
    assert len(result.memories) == 1
    assert (
        result.memories[0].content.parts[0].text
        == 'plain content without prefix'
    )
    assert result.memories[0].author is None

  @pytest.mark.asyncio
  async def test_search_memory_applies_min_importance(self, mock_httpx_client):
    """Test that min_importance is forwarded to recall when configured."""
    config = DakeraMemoryServiceConfig(min_importance=0.4)
    memory_service = DakeraMemoryService(
        base_url='http://localhost:3000', api_key='dk-test', config=config
    )

    await memory_service.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='q'
    )

    request_data = mock_httpx_client.post.call_args.kwargs['json']
    assert request_data['min_importance'] == 0.4

  @pytest.mark.asyncio
  async def test_search_memory_error_returns_empty(
      self, memory_service, mock_httpx_client
  ):
    """Test that recall errors return an empty response rather than raising."""
    mock_httpx_client.post.side_effect = Exception('API Error')

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='q'
    )
    assert result.memories == []
