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

from google.adk_community.templating import DjangoInstructionProvider


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


class TestDjangoInstructionProvider:
  """Test suite for DjangoInstructionProvider."""

  async def test_basic_variable_substitution(self, mock_readonly_context):
    """Test basic variable substitution."""
    mock_readonly_context.state = {'name': 'Alice', 'role': 'Engineer'}

    provider = DjangoInstructionProvider(
        'Hello {{ name }}, you are a {{ role }}.'
    )
    result = await provider(mock_readonly_context)

    assert result == 'Hello Alice, you are a Engineer.'

  async def test_nested_object_access(self, mock_readonly_context):
    """Test accessing nested objects."""
    mock_readonly_context.state = {
        'user': {'name': 'Bob', 'profile': {'age': 30, 'role': 'Developer'}}
    }

    provider = DjangoInstructionProvider(
        'User: {{ user.name }}, Role: {{ user.profile.role }}'
    )
    result = await provider(mock_readonly_context)

    assert result == 'User: Bob, Role: Developer'

  async def test_control_structures(self, mock_readonly_context):
    """Test if/else control structures."""
    mock_readonly_context.state = {'logged_in': True, 'username': 'Charlie'}

    provider = DjangoInstructionProvider(
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

    provider = DjangoInstructionProvider(
        '{% for server in servers %}{{ server.name }}: {{ server.cpu }}%\n{%'
        ' endfor %}'
    )
    result = await provider(mock_readonly_context)

    assert 'srv-01: 25%' in result
    assert 'srv-02: 90%' in result

  async def test_filters(self, mock_readonly_context):
    """Test Django filters."""
    mock_readonly_context.state = {'name': 'alice'}

    provider = DjangoInstructionProvider('Hello {{ name|upper }}!')
    result = await provider(mock_readonly_context)

    assert result == 'Hello ALICE!'

  async def test_default_filter(self, mock_readonly_context):
    """Test default filter for missing variables."""
    mock_readonly_context.state = {}

    provider = DjangoInstructionProvider('Hello {{ name|default:"Guest" }}!')
    result = await provider(mock_readonly_context)

    assert result == 'Hello Guest!'

  async def test_custom_filters(self, mock_readonly_context):
    """Test custom filters."""
    mock_readonly_context.state = {'number': 42}

    def double(value):
      return value * 2

    provider = DjangoInstructionProvider(
        '{% load custom %}Result: {{ number|double }}',
        custom_filters={'double': double},
    )
    result = await provider(mock_readonly_context)

    assert result == 'Result: 84'

  async def test_session_metadata_access(self, mock_readonly_context):
    """Test access to session metadata."""
    provider = DjangoInstructionProvider(
        'Session: {{ adk_session_id }}, User: {{ adk_user_id }}, App: {{'
        ' adk_app_name }}'
    )
    result = await provider(mock_readonly_context)

    assert result == 'Session: test-session-id, User: test-user, App: test-app'

  async def test_empty_template(self, mock_readonly_context):
    """Test empty template."""
    provider = DjangoInstructionProvider('')
    result = await provider(mock_readonly_context)

    assert result == ''

  async def test_for_empty_tag(self, mock_readonly_context):
    """Test for...empty tag."""
    mock_readonly_context.state = {'items': []}

    provider = DjangoInstructionProvider(
        '{% for item in items %}{{ item }}{% empty %}No items{% endfor %}'
    )
    result = await provider(mock_readonly_context)

    assert result == 'No items'

  async def test_comparison_operators(self, mock_readonly_context):
    """Test comparison operators in if statements."""
    mock_readonly_context.state = {'cpu': 95}

    provider = DjangoInstructionProvider(
        '{% if cpu > 80 %}CRITICAL{% else %}NORMAL{% endif %}'
    )
    result = await provider(mock_readonly_context)

    assert result == 'CRITICAL'

  async def test_complex_devops_scenario(self, mock_readonly_context):
    """Test complex DevOps scenario."""
    mock_readonly_context.state = {
        'environment': 'PRODUCTION',
        'user_query': 'Check system status',
        'servers': [
            {'id': 'srv-01', 'role': 'LoadBalancer', 'cpu': 15},
            {'id': 'srv-02', 'role': 'AppServer', 'cpu': 95},
        ],
    }

    template = """Environment: {{ environment }}
{% if servers %}
Active Servers:
{% for server in servers %}
  - [{{ server.id }}] {{ server.role }}: CPU {{ server.cpu }}%{% if server.cpu > 80 %} (CRITICAL!){% endif %}
{% endfor %}
{% else %}
No servers active.
{% endif %}
Query: {{ user_query }}"""

    provider = DjangoInstructionProvider(template)
    result = await provider(mock_readonly_context)

    assert 'Environment: PRODUCTION' in result
    assert 'srv-01' in result
    assert 'srv-02' in result
    assert 'CRITICAL!' in result
    assert 'Query: Check system status' in result

  async def test_length_filter(self, mock_readonly_context):
    """Test length filter."""
    mock_readonly_context.state = {'items': [1, 2, 3, 4, 5]}

    provider = DjangoInstructionProvider('Count: {{ items|length }}')
    result = await provider(mock_readonly_context)

    assert result == 'Count: 5'

  async def test_autoescape_disabled(self, mock_readonly_context):
    """Test that autoescape is disabled by default."""
    mock_readonly_context.state = {'html': '<b>bold</b>'}

    provider = DjangoInstructionProvider('{{ html }}', autoescape=False)
    result = await provider(mock_readonly_context)

    assert result == '<b>bold</b>'

  def test_import_error_when_django_not_installed(self, monkeypatch):
    """Test that ImportError is raised when Django is not installed."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
      if name == 'django.template' or name.startswith('django.'):
        raise ImportError('No module named django')
      return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)

    with pytest.raises(ImportError) as exc_info:
      DjangoInstructionProvider('test template')

    assert 'django' in str(exc_info.value).lower()
    assert 'google-adk-community[templating]' in str(exc_info.value)
