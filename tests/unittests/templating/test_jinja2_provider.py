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

from unittest.mock import MagicMock

import pytest

from google.adk_community.templating import Jinja2InstructionProvider


@pytest.fixture
def mock_readonly_context():
  """Create a mock ReadonlyContext for testing."""
  context = MagicMock()
  session = MagicMock()
  session.id = 'test-session-id'
  session.user_id = 'test-user'
  session.app_name = 'test-app'

  invocation_context = MagicMock()
  invocation_context.session = session

  context._invocation_context = invocation_context
  context.state = {}

  return context


class TestJinja2InstructionProvider:
  """Test suite for Jinja2InstructionProvider."""

  async def test_basic_variable_substitution(self, mock_readonly_context):
    """Test basic variable substitution."""
    mock_readonly_context.state = {'name': 'Alice', 'role': 'Engineer'}

    provider = Jinja2InstructionProvider(
        'Hello {{ name }}, you are a {{ role }}.'
    )
    result = await provider(mock_readonly_context)

    assert result == 'Hello Alice, you are a Engineer.'

  async def test_nested_object_access(self, mock_readonly_context):
    """Test accessing nested objects."""
    mock_readonly_context.state = {
        'user': {'name': 'Bob', 'profile': {'age': 30, 'role': 'Developer'}}
    }

    provider = Jinja2InstructionProvider(
        'User: {{ user.name }}, Role: {{ user.profile.role }}'
    )
    result = await provider(mock_readonly_context)

    assert result == 'User: Bob, Role: Developer'

  async def test_control_structures(self, mock_readonly_context):
    """Test if/else control structures."""
    mock_readonly_context.state = {'logged_in': True, 'username': 'Charlie'}

    provider = Jinja2InstructionProvider(
        '{% if logged_in %}Welcome {{ username }}!{% else %}Please log'
        ' in.{% endif %}'
    )
    result = await provider(mock_readonly_context)

    assert result == 'Welcome Charlie!'

  async def test_for_loop(self, mock_readonly_context):
    """Test for loops."""
    mock_readonly_context.state = {
        'servers': [
            {'name': 'srv-01', 'cpu': 25},
            {'name': 'srv-02', 'cpu': 90},
        ]
    }

    provider = Jinja2InstructionProvider(
        '{% for server in servers %}{{ server.name }}: {{ server.cpu }}%\n{%'
        ' endfor %}'
    )
    result = await provider(mock_readonly_context)

    assert 'srv-01: 25%' in result
    assert 'srv-02: 90%' in result

  async def test_filters(self, mock_readonly_context):
    """Test Jinja2 filters."""
    mock_readonly_context.state = {'name': 'alice'}

    provider = Jinja2InstructionProvider('Hello {{ name|upper }}!')
    result = await provider(mock_readonly_context)

    assert result == 'Hello ALICE!'

  async def test_custom_filters(self, mock_readonly_context):
    """Test custom filters."""
    mock_readonly_context.state = {'number': 42}

    def double(value):
      return value * 2

    provider = Jinja2InstructionProvider(
        'Result: {{ number|double }}', custom_filters={'double': double}
    )
    result = await provider(mock_readonly_context)

    assert result == 'Result: 84'

  async def test_custom_tests(self, mock_readonly_context):
    """Test custom tests."""
    mock_readonly_context.state = {'value': 10}

    def is_large(value):
      return value > 50

    provider = Jinja2InstructionProvider(
        '{% if value is large %}Large{% else %}Small{% endif %}',
        custom_tests={'large': is_large},
    )
    result = await provider(mock_readonly_context)

    assert result == 'Small'

  async def test_custom_globals(self, mock_readonly_context):
    """Test custom global variables."""
    mock_readonly_context.state = {}

    provider = Jinja2InstructionProvider(
        'Version: {{ version }}', custom_globals={'version': '1.0.0'}
    )
    result = await provider(mock_readonly_context)

    assert result == 'Version: 1.0.0'

  async def test_session_metadata_access(self, mock_readonly_context):
    """Test access to session metadata."""
    provider = Jinja2InstructionProvider(
        'Session: {{ adk_session_id }}, User: {{ adk_user_id }}, App: {{'
        ' adk_app_name }}'
    )
    result = await provider(mock_readonly_context)

    assert result == 'Session: test-session-id, User: test-user, App: test-app'

  async def test_missing_variable_default(self, mock_readonly_context):
    """Test default value for missing variables."""
    mock_readonly_context.state = {}

    provider = Jinja2InstructionProvider('Hello {{ name|default("Guest") }}!')
    result = await provider(mock_readonly_context)

    assert result == 'Hello Guest!'

  async def test_empty_template(self, mock_readonly_context):
    """Test empty template."""
    provider = Jinja2InstructionProvider('')
    result = await provider(mock_readonly_context)

    assert result == ''

  async def test_complex_devops_scenario(self, mock_readonly_context):
    """Test complex DevOps scenario similar to the example."""
    mock_readonly_context.state = {
        'environment': 'PRODUCTION',
        'user_query': 'Check system status',
        'servers': [
            {'id': 'srv-01', 'role': 'LoadBalancer', 'cpu': 15},
            {'id': 'srv-02', 'role': 'AppServer', 'cpu': 95},
        ],
    }

    template = """
Environment: {{ environment }}
{% if servers %}
Active Servers:
{% for server in servers %}
  - [{{ server.id }}] {{ server.role }}: CPU {{ server.cpu }}%
    {%- if server.cpu > 80 %} (CRITICAL!){% endif %}
{% endfor %}
{% else %}
No servers active.
{% endif %}
Query: {{ user_query }}
""".strip()

    provider = Jinja2InstructionProvider(template)
    result = await provider(mock_readonly_context)

    assert 'Environment: PRODUCTION' in result
    assert 'srv-01' in result
    assert 'srv-02' in result
    assert 'CRITICAL!' in result
    assert 'Query: Check system status' in result

  def test_import_error_when_jinja2_not_installed(self, monkeypatch):
    """Test that ImportError is raised when jinja2 is not installed."""
    # Mock the import to simulate jinja2 not being installed
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
      if name == 'jinja2':
        raise ImportError('No module named jinja2')
      return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)

    with pytest.raises(ImportError) as exc_info:
      Jinja2InstructionProvider('test template')

    assert 'jinja2' in str(exc_info.value).lower()
    assert 'google-adk-community[templating]' in str(exc_info.value)
