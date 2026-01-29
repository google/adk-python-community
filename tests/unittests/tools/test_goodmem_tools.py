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

from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import call
from unittest.mock import patch

import pytest
import requests

from google.adk_community.tools.goodmem import goodmem_tools
from google.adk_community.tools.goodmem.goodmem_client import GoodmemClient
from google.adk_community.tools.goodmem.goodmem_tools import _format_debug_table
from google.adk_community.tools.goodmem.goodmem_tools import _format_timestamp_for_table
from google.adk_community.tools.goodmem.goodmem_tools import _wrap_content
from google.adk_community.tools.goodmem.goodmem_tools import goodmem_fetch
from google.adk_community.tools.goodmem.goodmem_tools import goodmem_save


class TestGoodmemSave:
  """Test cases for goodmem_save function."""

  @pytest.fixture(autouse=True)
  def clear_client_cache(self):
    """Clear the client cache before each test."""
    goodmem_tools._client_cache.clear()
    yield
    goodmem_tools._client_cache.clear()

  @pytest.fixture
  def mock_config(self):
    """Set up mock configuration."""
    return {
        'base_url': 'http://localhost:8080',
        'api_key': 'test-api-key',
    }

  @pytest.fixture
  def mock_tool_context(self):
    """Create a mock tool context."""
    context = MagicMock()
    context.user_id = 'test-user'
    context.session = MagicMock()
    context.session.id = 'test-session'
    # Mock state as a dict
    context.state = {}
    return context

  @pytest.mark.asyncio
  async def test_save_success(self, mock_config, mock_tool_context):
    """Test successful memory write."""
    with patch(
        'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
    ) as MockClient:
      mock_client = MockClient.return_value
      mock_client.insert_memory.return_value = {'memoryId': 'memory-123'}
      # Mock space already exists
      mock_client.list_spaces.return_value = [
          {'spaceId': 'existing-space-123', 'name': 'adk_tool_test-user'}
      ]

      response = await goodmem_save(
          content='Test content',
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
      )

      assert response.success is True
      assert response.memory_id == 'memory-123'
      assert 'Successfully wrote' in response.message

      mock_client.list_spaces.assert_called_once_with(name='adk_tool_test-user')
      mock_client.insert_memory.assert_called_once_with(
          space_id='existing-space-123',
          content='Test content',
          content_type='text/plain',
          metadata={'user_id': 'test-user', 'session_id': 'test-session'},
      )
      # Verify space_id was cached
      assert (
          mock_tool_context.state['_goodmem_space_id'] == 'existing-space-123'
      )

  @pytest.mark.asyncio
  async def test_save_missing_base_url(self, mock_tool_context):
    """Test error when base_url is not provided."""
    response = await goodmem_save(
        content='Test content',
        tool_context=mock_tool_context,
        base_url=None,
        api_key='test-api-key',
    )

    assert response.success is False
    assert 'base_url' in response.message.lower()

  @pytest.mark.asyncio
  async def test_save_missing_api_key(self, mock_tool_context):
    """Test error when api_key is not provided."""
    response = await goodmem_save(
        content='Test content',
        tool_context=mock_tool_context,
        base_url='http://localhost:8080',
        api_key=None,
    )

    assert response.success is False
    assert 'api_key' in response.message.lower()

  @pytest.mark.asyncio
  async def test_save_connection_error(self, mock_config, mock_tool_context):
    """Test handling of connection error."""
    with patch(
        'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
    ) as MockClient:
      mock_client = MockClient.return_value
      mock_client.list_spaces.return_value = [
          {'spaceId': 'existing-space-123', 'name': 'adk_tool_test-user'}
      ]
      mock_client.insert_memory.side_effect = (
          requests.exceptions.ConnectionError('Connection failed')
      )

      response = await goodmem_save(
          content='Test content',
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
      )

      assert response.success is False
      assert 'Connection error' in response.message
      mock_client.list_spaces.assert_called_once_with(name='adk_tool_test-user')

  @pytest.mark.asyncio
  async def test_save_http_error_401(self, mock_config, mock_tool_context):
    """Test handling of authentication error."""
    with patch(
        'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
    ) as MockClient:
      mock_client = MockClient.return_value
      mock_client.list_spaces.return_value = [
          {'spaceId': 'existing-space-123', 'name': 'adk_tool_test-user'}
      ]
      mock_error = requests.exceptions.HTTPError()
      mock_error.response = MagicMock()
      mock_error.response.status_code = 401
      mock_client.insert_memory.side_effect = mock_error

      response = await goodmem_save(
          content='Test content',
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
      )

      assert response.success is False
      assert 'Authentication error' in response.message
      mock_client.list_spaces.assert_called_once_with(name='adk_tool_test-user')

  @pytest.mark.asyncio
  async def test_save_http_error_404(self, mock_config, mock_tool_context):
    """Test handling of not found error."""
    with patch(
        'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
    ) as MockClient:
      mock_client = MockClient.return_value
      mock_client.list_spaces.return_value = [
          {'spaceId': 'existing-space-123', 'name': 'adk_tool_test-user'}
      ]
      mock_error = requests.exceptions.HTTPError()
      mock_error.response = MagicMock()
      mock_error.response.status_code = 404
      mock_client.insert_memory.side_effect = mock_error

      response = await goodmem_save(
          content='Test content',
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
      )

      assert response.success is False
      assert 'Not found error' in response.message
      mock_client.list_spaces.assert_called_once_with(name='adk_tool_test-user')

  @pytest.mark.asyncio
  async def test_save_without_tool_context(self, mock_config):
    """Test save without tool context returns error."""
    response = await goodmem_save(
        content='Test content',
        base_url=mock_config['base_url'],
        api_key=mock_config['api_key'],
    )

    assert response.success is False
    assert 'tool_context is required' in response.message

  @pytest.mark.asyncio
  async def test_save_creates_space_if_not_exists(
      self, mock_config, mock_tool_context
  ):
    """Test that a new space is created if it doesn't exist."""
    with patch(
        'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
    ) as MockClient:
      mock_client = MockClient.return_value
      # No existing spaces
      mock_client.list_spaces.return_value = []
      # Mock embedders
      mock_client.list_embedders.return_value = [
          {'embedderId': 'embedder-1', 'name': 'Test Embedder'}
      ]
      # Mock space creation
      mock_client.create_space.return_value = {'spaceId': 'new-space-123'}
      mock_client.insert_memory.return_value = {'memoryId': 'memory-123'}

      response = await goodmem_save(
          content='Test content',
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
      )

      assert response.success is True
      mock_client.list_spaces.assert_called_once_with(name='adk_tool_test-user')
      mock_client.create_space.assert_called_once_with(
          'adk_tool_test-user', 'embedder-1'
      )
      assert mock_tool_context.state['_goodmem_space_id'] == 'new-space-123'

  @pytest.mark.asyncio
  async def test_save_space_create_conflict_reuses_existing(
      self, mock_config, mock_tool_context
  ):
    """Test handling 409 conflict by reusing existing space."""
    with patch(
        'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
    ) as MockClient:
      mock_client = MockClient.return_value
      mock_client.list_spaces.side_effect = [
          [],
          [{'spaceId': 'existing-space-123', 'name': 'adk_tool_test-user'}],
      ]
      mock_client.list_embedders.return_value = [
          {'embedderId': 'embedder-1', 'name': 'Test Embedder'}
      ]
      conflict_error = requests.exceptions.HTTPError()
      conflict_error.response = MagicMock()
      conflict_error.response.status_code = 409
      mock_client.create_space.side_effect = conflict_error
      mock_client.insert_memory.return_value = {'memoryId': 'memory-123'}

      response = await goodmem_save(
          content='Test content',
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
      )

      assert response.success is True
      mock_client.list_spaces.assert_has_calls([
          call(name='adk_tool_test-user'),
          call(name='adk_tool_test-user'),
      ])
      assert mock_client.list_spaces.call_count == 2
      mock_client.insert_memory.assert_called_once_with(
          space_id='existing-space-123',
          content='Test content',
          content_type='text/plain',
          metadata={'user_id': 'test-user', 'session_id': 'test-session'},
      )
      assert (
          mock_tool_context.state['_goodmem_space_id'] == 'existing-space-123'
      )

  @pytest.mark.asyncio
  async def test_save_uses_cached_space_id(
      self, mock_config, mock_tool_context
  ):
    """Test that cached space_id is used on subsequent calls."""
    # Pre-populate cache
    mock_tool_context.state['_goodmem_space_id'] = 'cached-space-123'

    with patch(
        'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
    ) as MockClient:
      mock_client = MockClient.return_value
      mock_client.insert_memory.return_value = {'memoryId': 'memory-123'}

      response = await goodmem_save(
          content='Test content',
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
      )

      assert response.success is True
      # list_spaces should NOT be called since we have cache
      mock_client.list_spaces.assert_not_called()
      mock_client.insert_memory.assert_called_once_with(
          space_id='cached-space-123',
          content='Test content',
          content_type='text/plain',
          metadata={'user_id': 'test-user', 'session_id': 'test-session'},
      )

  @pytest.mark.asyncio
  async def test_save_prefers_exact_space_name(
      self, mock_config, mock_tool_context
  ):
    """Test that exact name match is preferred over case mismatch."""
    with patch(
        'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
    ) as MockClient:
      mock_client = MockClient.return_value
      mock_client.list_spaces.return_value = [
          {'spaceId': 'case-mismatch', 'name': 'ADK_TOOL_TEST-USER'},
          {'spaceId': 'exact-match', 'name': 'adk_tool_test-user'},
      ]
      mock_client.insert_memory.return_value = {'memoryId': 'memory-123'}

      response = await goodmem_save(
          content='Test content',
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
      )

      assert response.success is True
      mock_client.list_spaces.assert_called_once_with(
          name='adk_tool_test-user'
      )
      mock_client.create_space.assert_not_called()
      mock_client.insert_memory.assert_called_once_with(
          space_id='exact-match',
          content='Test content',
          content_type='text/plain',
          metadata={'user_id': 'test-user', 'session_id': 'test-session'},
      )

  @pytest.mark.asyncio
  async def test_save_with_custom_embedder_id(
      self, mock_config, mock_tool_context
  ):
    """Test using custom embedder_id."""
    with patch(
        'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
    ) as MockClient:
      mock_client = MockClient.return_value
      mock_client.list_spaces.return_value = []
      mock_client.list_embedders.return_value = [
          {'embedderId': 'custom-embedder', 'name': 'Custom Embedder'}
      ]
      mock_client.create_space.return_value = {'spaceId': 'new-space-123'}
      mock_client.insert_memory.return_value = {'memoryId': 'memory-123'}

      response = await goodmem_save(
          content='Test content',
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
          embedder_id='custom-embedder',
      )

      assert response.success is True
      mock_client.list_spaces.assert_called_once_with(name='adk_tool_test-user')
      mock_client.create_space.assert_called_once_with(
          'adk_tool_test-user', 'custom-embedder'
      )

  @pytest.mark.asyncio
  async def test_save_invalid_embedder_id(self, mock_config, mock_tool_context):
    """Test error when embedder_id is invalid."""
    with patch(
        'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
    ) as MockClient:
      mock_client = MockClient.return_value
      mock_client.list_spaces.return_value = []
      mock_client.list_embedders.return_value = [
          {'embedderId': 'valid-embedder', 'name': 'Valid Embedder'}
      ]

      response = await goodmem_save(
          content='Test content',
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
          embedder_id='invalid-embedder',
      )

      assert response.success is False
      assert 'invalid-embedder' in response.message
      assert 'not found' in response.message
      mock_client.list_spaces.assert_called_once_with(name='adk_tool_test-user')


class TestGoodmemFetch:
  """Test cases for goodmem_fetch function."""

  @pytest.fixture(autouse=True)
  def clear_client_cache(self):
    """Clear the client cache before each test."""
    goodmem_tools._client_cache.clear()
    yield
    goodmem_tools._client_cache.clear()

  @pytest.fixture
  def mock_config(self):
    """Set up mock configuration."""
    return {
        'base_url': 'http://localhost:8080',
        'api_key': 'test-api-key',
    }

  @pytest.fixture
  def mock_tool_context(self):
    """Create a mock tool context."""
    context = MagicMock()
    context.user_id = 'test-user'
    context.session = MagicMock()
    context.session.id = 'test-session'
    context.state = {}
    return context

  @pytest.mark.asyncio
  async def test_fetch_success(self, mock_config, mock_tool_context):
    """Test successful memory retrieval."""
    with patch(
        'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
    ) as MockClient:
      mock_client = MockClient.return_value
      mock_client.list_spaces.return_value = [
          {'spaceId': 'existing-space-123', 'name': 'adk_tool_test-user'}
      ]
      mock_client.retrieve_memories.return_value = [{
          'retrievedItem': {
              'chunk': {
                  'chunk': {
                      'memoryId': 'memory-123',
                      'chunkText': 'Test memory content',
                      'updatedAt': 1234567890,
                  }
              }
          }
      }]
      mock_client.get_memory_by_id.return_value = {
          'metadata': {'user_id': 'test-user'}
      }

      response = await goodmem_fetch(
          query='test query',
          top_k=5,
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
      )

      assert response.success is True
      assert response.count == 1
      assert len(response.memories) == 1
      assert response.memories[0].memory_id == 'memory-123'
      assert response.memories[0].content == 'Test memory content'
      assert response.memories[0].metadata == {'user_id': 'test-user'}
      mock_client.list_spaces.assert_called_once_with(name='adk_tool_test-user')

  @pytest.mark.asyncio
  async def test_fetch_no_results(self, mock_config, mock_tool_context):
    """Test fetch with no matching memories."""
    with patch(
        'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
    ) as MockClient:
      mock_client = MockClient.return_value
      mock_client.list_spaces.return_value = [
          {'spaceId': 'existing-space-123', 'name': 'adk_tool_test-user'}
      ]
      mock_client.retrieve_memories.return_value = []

      response = await goodmem_fetch(
          query='test query',
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
      )

      assert response.success is True
      assert response.count == 0
      assert len(response.memories) == 0
      assert 'No memories found' in response.message
      mock_client.list_spaces.assert_called_once_with(name='adk_tool_test-user')

  @pytest.mark.asyncio
  async def test_fetch_top_k_validation(self, mock_config, mock_tool_context):
    """Test top_k parameter validation."""
    with patch(
        'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
    ) as MockClient:
      mock_client = MockClient.return_value
      mock_client.list_spaces.return_value = [
          {'spaceId': 'existing-space-123', 'name': 'adk_tool_test-user'}
      ]
      mock_client.retrieve_memories.return_value = []

      # Test max top_k
      await goodmem_fetch(
          query='test',
          top_k=25,
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
      )
      mock_client.retrieve_memories.assert_called_with(
          query='test',
          space_ids=['existing-space-123'],
          request_size=20,  # Should be capped at 20
      )
      mock_client.list_spaces.assert_called_once_with(name='adk_tool_test-user')

      # Reset mock
      mock_client.reset_mock()
      mock_tool_context.state = {}
      mock_client.list_spaces.return_value = [
          {'spaceId': 'existing-space-123', 'name': 'adk_tool_test-user'}
      ]
      mock_client.retrieve_memories.return_value = []

      # Test min top_k
      await goodmem_fetch(
          query='test',
          top_k=0,
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
      )
      mock_client.retrieve_memories.assert_called_with(
          query='test',
          space_ids=['existing-space-123'],
          request_size=1,  # Should be at least 1
      )
      mock_client.list_spaces.assert_called_once_with(name='adk_tool_test-user')

  @pytest.mark.asyncio
  async def test_fetch_cleans_content_prefix(
      self, mock_config, mock_tool_context
  ):
    """Test that User: and LLM: prefixes are removed from content."""
    with patch(
        'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
    ) as MockClient:
      mock_client = MockClient.return_value
      mock_client.list_spaces.return_value = [
          {'spaceId': 'existing-space-123', 'name': 'adk_tool_test-user'}
      ]
      mock_client.retrieve_memories.return_value = [
          {
              'retrievedItem': {
                  'chunk': {
                      'chunk': {
                          'memoryId': 'memory-1',
                          'chunkText': 'User: Hello there',
                          'updatedAt': 1234567890,
                      }
                  }
              }
          },
          {
              'retrievedItem': {
                  'chunk': {
                      'chunk': {
                          'memoryId': 'memory-2',
                          'chunkText': 'LLM: Hi! How can I help?',
                          'updatedAt': 1234567891,
                      }
                  }
              }
          },
      ]
      mock_client.get_memory_by_id.return_value = {'metadata': {}}

      response = await goodmem_fetch(
          query='test',
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
      )

      assert response.memories[0].content == 'Hello there'
      assert response.memories[1].content == 'Hi! How can I help?'
      mock_client.list_spaces.assert_called_once_with(name='adk_tool_test-user')

  @pytest.mark.asyncio
  async def test_fetch_connection_error(self, mock_config, mock_tool_context):
    """Test handling of connection error."""
    with patch(
        'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
    ) as MockClient:
      mock_client = MockClient.return_value
      mock_client.list_spaces.side_effect = requests.exceptions.ConnectionError(
          'Connection failed'
      )

      response = await goodmem_fetch(
          query='test',
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
      )

      assert response.success is False
      assert (
          'Connection error' in response.message
          or 'Error getting or creating space' in response.message
      )
      mock_client.list_spaces.assert_called_once_with(name='adk_tool_test-user')

  @pytest.mark.asyncio
  async def test_fetch_missing_config(self, mock_tool_context):
    """Test error when configuration is missing."""
    response = await goodmem_fetch(
        query='test',
        tool_context=mock_tool_context,
        base_url=None,
        api_key=None,
    )

    assert response.success is False
    assert 'base_url' in response.message.lower()

  @pytest.mark.asyncio
  async def test_fetch_deduplicates_memories(
      self, mock_config, mock_tool_context
  ):
    """Test that duplicate memory IDs are filtered."""
    with patch(
        'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
    ) as MockClient:
      mock_client = MockClient.return_value
      mock_client.list_spaces.return_value = [
          {'spaceId': 'existing-space-123', 'name': 'adk_tool_test-user'}
      ]
      # Return same memory ID twice
      mock_client.retrieve_memories.return_value = [
          {
              'retrievedItem': {
                  'chunk': {
                      'chunk': {
                          'memoryId': 'memory-123',
                          'chunkText': 'First chunk',
                          'updatedAt': 1234567890,
                      }
                  }
              }
          },
          {
              'retrievedItem': {
                  'chunk': {
                      'chunk': {
                          'memoryId': 'memory-123',
                          'chunkText': 'Second chunk',
                          'updatedAt': 1234567891,
                      }
                  }
              }
          },
      ]
      mock_client.get_memory_by_id.return_value = {'metadata': {}}

      response = await goodmem_fetch(
          query='test',
          tool_context=mock_tool_context,
          base_url=mock_config['base_url'],
          api_key=mock_config['api_key'],
      )

      # Should only return one memory despite two chunks
      assert response.count == 1
      assert len(response.memories) == 1
      mock_client.list_spaces.assert_called_once_with(name='adk_tool_test-user')

  @pytest.mark.asyncio
  async def test_fetch_debug_table_output(self, mock_config, mock_tool_context):
    """Test that debug table is printed when debug mode is enabled."""
    with (
        patch(
            'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
        ) as MockClient,
        patch('builtins.print') as mock_print,
    ):
      mock_client = MockClient.return_value
      mock_client.list_spaces.return_value = [
          {'spaceId': 'existing-space-123', 'name': 'adk_tool_test-user'}
      ]
      # Set debug mode
      goodmem_tools._tool_debug = True
      try:
        mock_client.retrieve_memories.return_value = [{
            'retrievedItem': {
                'chunk': {
                    'chunk': {
                        'memoryId': 'memory-123',
                        'chunkText': 'User: Test content',
                        'updatedAt': 1234567890000,  # 2009-02-13 23:31:30 UTC
                    }
                }
            }
        }]
        mock_client.get_memory_by_id.return_value = {
            'metadata': {'user_id': 'test-user'}
        }

        response = await goodmem_fetch(
            query='test query',
            tool_context=mock_tool_context,
            base_url=mock_config['base_url'],
            api_key=mock_config['api_key'],
        )

        assert response.success is True
        # Verify debug table was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        debug_table_printed = any(
            '[DEBUG] Retrieved memories:' in str(call) for call in print_calls
        )
        assert (
            debug_table_printed
        ), 'Debug table should be printed when debug is enabled'
        mock_client.list_spaces.assert_called_once_with(
            name='adk_tool_test-user'
        )
      finally:
        goodmem_tools._tool_debug = False

  @pytest.mark.asyncio
  async def test_fetch_role_detection_from_prefix(
      self, mock_config, mock_tool_context
  ):
    """Test that role is correctly detected from content prefix."""
    with (
        patch(
            'google.adk_community.tools.goodmem.goodmem_tools.GoodmemClient'
        ) as MockClient,
        patch('builtins.print') as mock_print,
    ):
      mock_client = MockClient.return_value
      mock_client.list_spaces.return_value = [
          {'spaceId': 'existing-space-123', 'name': 'adk_tool_test-user'}
      ]
      goodmem_tools._tool_debug = True
      try:
        mock_client.retrieve_memories.return_value = [
            {
                'retrievedItem': {
                    'chunk': {
                        'chunk': {
                            'memoryId': 'memory-user',
                            'chunkText': 'User: This is from user',
                            'updatedAt': 1234567890000,
                        }
                    }
                }
            },
            {
                'retrievedItem': {
                    'chunk': {
                        'chunk': {
                            'memoryId': 'memory-llm',
                            'chunkText': 'LLM: This is from llm',
                            'updatedAt': 1234567891000,
                        }
                    }
                }
            },
        ]
        mock_client.get_memory_by_id.return_value = {'metadata': {}}

        response = await goodmem_fetch(
            query='test',
            tool_context=mock_tool_context,
            base_url=mock_config['base_url'],
            api_key=mock_config['api_key'],
        )

        assert response.success is True
        assert len(response.memories) == 2
        # Content should have prefix removed
        assert response.memories[0].content == 'This is from user'
        assert response.memories[1].content == 'This is from llm'

        # Verify debug table contains correct roles
        print_calls = str(mock_print.call_args_list)
        assert 'user' in print_calls.lower() or 'role' in print_calls.lower()
        mock_client.list_spaces.assert_called_once_with(
            name='adk_tool_test-user'
        )
      finally:
        goodmem_tools._tool_debug = False


class TestDebugTableFormatting:
  """Test cases for debug table formatting functions."""

  def test_format_timestamp_for_table(self):
    """Test timestamp formatting for table display."""
    # Test valid timestamp
    timestamp_ms = 1234567890000  # 2009-02-13 23:31:30 UTC
    result = _format_timestamp_for_table(timestamp_ms)
    assert result == '2009-02-13 23:31'

    # Test None
    result = _format_timestamp_for_table(None)
    assert result == ''

    # Test invalid timestamp (should return string representation)
    result = _format_timestamp_for_table('invalid')
    assert isinstance(result, str)

  def test_wrap_content(self):
    """Test content wrapping."""
    # Short content should not wrap
    content = 'Short content'
    result = _wrap_content(content, max_width=55)
    assert result == ['Short content']

    # Long content should wrap
    long_content = (
        'This is a very long content that should definitely wrap because it'
        ' exceeds the maximum width of 55 characters'
    )
    result = _wrap_content(long_content, max_width=55)
    assert len(result) > 1
    assert all(len(line) <= 55 for line in result)

    # Empty content
    result = _wrap_content('', max_width=55)
    assert result == ['']

  def test_format_debug_table(self):
    """Test debug table formatting."""
    records = [
        {
            'memory_id': '019c01e4-385a-7784-a2aa-4b2a3d0b7167',
            'timestamp_ms': 1738029420000,  # 2026-01-27 23:57:00 UTC
            'role': 'user',
            'content': "what's my name",
        },
        {
            'memory_id': '019c01e7-a4d1-7400-ad8b-6782f4277343',
            'timestamp_ms': 1738032060000,  # 2026-01-28 00:01:00 UTC
            'role': 'llm',
            'content': (
                "As an AI, I don't know your name unless you've told me during"
                ' our current conversation.'
            ),
        },
    ]

    result = _format_debug_table(records)

    # Verify table structure
    assert 'memory ID' in result
    assert 'datetime' in result
    assert 'role' in result
    assert 'content' in result
    assert '019c01e4-385a-7784-a2aa-4b2a3d0b7167' in result
    assert 'user' in result
    assert 'llm' in result
    assert "what's my name" in result
    assert '|' in result  # Table separators

    # Test empty records
    result = _format_debug_table([])
    assert result == ''

  def test_format_debug_table_with_wrapped_content(self):
    """Test debug table with content that needs wrapping."""
    records = [
        {
            'memory_id': 'test-id-123',
            'timestamp_ms': 1234567890000,
            'role': 'user',
            'content': (
                'This is a very long content that should wrap because it'
                ' exceeds the maximum width of 55 characters and needs to be'
                ' displayed across multiple lines'
            ),
        },
    ]

    result = _format_debug_table(records)

    # Should contain the memory ID and role
    assert 'test-id-123' in result
    assert 'user' in result
    # Content should be wrapped (multiple lines)
    lines = result.split('\n')
    # Should have header, separator, and at least 2 content lines
    assert len(lines) >= 4


class TestGoodmemClientNDJSON:
  """Test cases for NDJSON parsing edge cases in GoodmemClient."""

  def test_ndjson_with_blank_lines(self):
    """Test NDJSON parsing with blank lines interspersed."""
    client = GoodmemClient(base_url='http://localhost:8080', api_key='test-key')

    # Mock response with blank lines between valid JSON
    ndjson_response = (
        '\n{"retrievedItem": {"chunk": {"chunk": {"memoryId": "1", "chunkText":'
        ' "First"}}}}\n\n{"retrievedItem": {"chunk": {"chunk": {"memoryId":'
        ' "2", "chunkText": "Second"}}}}\n'
    )

    with patch('requests.post') as mock_post:
      mock_response = Mock()
      mock_response.text = ndjson_response
      mock_response.raise_for_status = Mock()
      mock_post.return_value = mock_response

      result = client.retrieve_memories(query='test', space_ids=['space-1'])

      assert len(result) == 2
      assert result[0]['retrievedItem']['chunk']['chunk']['memoryId'] == '1'
      assert result[1]['retrievedItem']['chunk']['chunk']['memoryId'] == '2'

  def test_ndjson_with_multiple_consecutive_blank_lines(self):
    """Test NDJSON parsing with multiple consecutive blank lines."""
    client = GoodmemClient(base_url='http://localhost:8080', api_key='test-key')

    ndjson_response = (
        '{"retrievedItem": {"chunk": {"chunk": {"memoryId": "1", "chunkText":'
        ' "First"}}}}\n\n\n\n{"retrievedItem": {"chunk": {"chunk": {"memoryId":'
        ' "2", "chunkText": "Second"}}}}'
    )

    with patch('requests.post') as mock_post:
      mock_response = Mock()
      mock_response.text = ndjson_response
      mock_response.raise_for_status = Mock()
      mock_post.return_value = mock_response

      result = client.retrieve_memories(query='test', space_ids=['space-1'])

      assert len(result) == 2

  def test_ndjson_with_whitespace_only_lines(self):
    """Test NDJSON parsing with lines containing only whitespace."""
    client = GoodmemClient(base_url='http://localhost:8080', api_key='test-key')

    ndjson_response = (
        '{"retrievedItem": {"chunk": {"chunk": {"memoryId": "1", "chunkText":'
        ' "First"}}}}\n   \n\t\n{"retrievedItem": {"chunk": {"chunk":'
        ' {"memoryId": "2", "chunkText": "Second"}}}}'
    )

    with patch('requests.post') as mock_post:
      mock_response = Mock()
      mock_response.text = ndjson_response
      mock_response.raise_for_status = Mock()
      mock_post.return_value = mock_response

      result = client.retrieve_memories(query='test', space_ids=['space-1'])

      assert len(result) == 2

  def test_ndjson_with_trailing_newlines(self):
    """Test NDJSON parsing with trailing newlines."""
    client = GoodmemClient(base_url='http://localhost:8080', api_key='test-key')

    ndjson_response = (
        '{"retrievedItem": {"chunk": {"chunk": {"memoryId": "1", "chunkText":'
        ' "First"}}}}\n{"retrievedItem": {"chunk": {"chunk": {"memoryId": "2",'
        ' "chunkText": "Second"}}}}\n\n\n'
    )

    with patch('requests.post') as mock_post:
      mock_response = Mock()
      mock_response.text = ndjson_response
      mock_response.raise_for_status = Mock()
      mock_post.return_value = mock_response

      result = client.retrieve_memories(query='test', space_ids=['space-1'])

      assert len(result) == 2

  def test_ndjson_empty_response(self):
    """Test NDJSON parsing with empty response."""
    client = GoodmemClient(base_url='http://localhost:8080', api_key='test-key')

    ndjson_response = ''

    with patch('requests.post') as mock_post:
      mock_response = Mock()
      mock_response.text = ndjson_response
      mock_response.raise_for_status = Mock()
      mock_post.return_value = mock_response

      result = client.retrieve_memories(query='test', space_ids=['space-1'])

      assert len(result) == 0

  def test_ndjson_only_blank_lines(self):
    """Test NDJSON parsing with only blank lines."""
    client = GoodmemClient(base_url='http://localhost:8080', api_key='test-key')

    ndjson_response = '\n\n\n   \n\t\n'

    with patch('requests.post') as mock_post:
      mock_response = Mock()
      mock_response.text = ndjson_response
      mock_response.raise_for_status = Mock()
      mock_post.return_value = mock_response

      result = client.retrieve_memories(query='test', space_ids=['space-1'])

      assert len(result) == 0

  def test_ndjson_filters_non_retrieved_items(self):
    """Test that lines without 'retrievedItem' key are filtered out."""
    client = GoodmemClient(base_url='http://localhost:8080', api_key='test-key')

    # Mix of valid retrievedItem and other JSON objects
    ndjson_response = (
        '{"retrievedItem": {"chunk": {"chunk": {"memoryId": "1", "chunkText":'
        ' "First"}}}}\n{"status": "processing"}\n{"retrievedItem": {"chunk":'
        ' {"chunk": {"memoryId": "2", "chunkText": "Second"}}}}'
    )

    with patch('requests.post') as mock_post:
      mock_response = Mock()
      mock_response.text = ndjson_response
      mock_response.raise_for_status = Mock()
      mock_post.return_value = mock_response

      result = client.retrieve_memories(query='test', space_ids=['space-1'])

      # Should only return the 2 items with retrievedItem key
      assert len(result) == 2
      assert all('retrievedItem' in item for item in result)


class TestGoodmemClientListSpaces:
  """Test cases for GoodmemClient list_spaces."""

  def test_list_spaces_with_name_filter(self):
    """Test list_spaces includes nameFilter and maxResults."""
    client = GoodmemClient(base_url='http://localhost:8080', api_key='test-key')

    with patch('requests.get') as mock_get:
      mock_response = Mock()
      mock_response.json.return_value = {
          'spaces': [{'spaceId': 'space-1', 'name': 'adk_tool_test-user'}]
      }
      mock_response.raise_for_status = Mock()
      mock_get.return_value = mock_response

      result = client.list_spaces(name='adk_tool_test-user')

      assert len(result) == 1
      assert result[0]['name'] == 'adk_tool_test-user'
      mock_get.assert_called_once_with(
          'http://localhost:8080/v1/spaces',
          headers=client._headers,
          params={'maxResults': 1000, 'nameFilter': 'adk_tool_test-user'},
          timeout=30,
      )
