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

from google.adk_community.templating import MustacheInstructionProvider


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


class TestMustacheInstructionProvider:
  """Test suite for MustacheInstructionProvider."""

  async def test_basic_variable_substitution(self, mock_readonly_context):
    """Test basic variable substitution."""
    mock_readonly_context.state = {'name': 'Alice', 'role': 'Engineer'}

    provider = MustacheInstructionProvider(
        'Hello {{name}}, you are a {{role}}.'
    )
    result = await provider(mock_readonly_context)

    assert result == 'Hello Alice, you are a Engineer.'

  async def test_nested_object_access(self, mock_readonly_context):
    """Test accessing nested objects."""
    mock_readonly_context.state = {
        'user': {'name': 'Bob', 'profile': {'role': 'Developer'}}
    }

    provider = MustacheInstructionProvider(
        'User: {{user.name}}, Role: {{user.profile.role}}'
    )
    result = await provider(mock_readonly_context)

    assert result == 'User: Bob, Role: Developer'

  async def test_section_with_truthy_value(self, mock_readonly_context):
    """Test section rendering with truthy value."""
    mock_readonly_context.state = {'logged_in': True, 'username': 'Charlie'}

    provider = MustacheInstructionProvider(
        '{{#logged_in}}Welcome {{username}}!{{/logged_in}}'
    )
    result = await provider(mock_readonly_context)

    assert result == 'Welcome Charlie!'

  async def test_inverted_section(self, mock_readonly_context):
    """Test inverted section (renders when value is falsy)."""
    mock_readonly_context.state = {'logged_in': False}

    provider = MustacheInstructionProvider(
        '{{^logged_in}}Please log in.{{/logged_in}}'
    )
    result = await provider(mock_readonly_context)

    assert result == 'Please log in.'

  async def test_list_iteration(self, mock_readonly_context):
    """Test iterating over a list."""
    mock_readonly_context.state = {
        'servers': [
            {'name': 'srv-01', 'cpu': 25},
            {'name': 'srv-02', 'cpu': 90},
        ]
    }

    provider = MustacheInstructionProvider(
        '{{#servers}}{{name}}: {{cpu}}%\n{{/servers}}'
    )
    result = await provider(mock_readonly_context)

    assert 'srv-01: 25%' in result
    assert 'srv-02: 90%' in result

  async def test_session_metadata_access(self, mock_readonly_context):
    """Test access to session metadata."""
    provider = MustacheInstructionProvider(
        'Session: {{adk_session_id}}, User: {{adk_user_id}}, App:'
        ' {{adk_app_name}}'
    )
    result = await provider(mock_readonly_context)

    assert result == 'Session: test-session-id, User: test-user, App: test-app'

  async def test_missing_variable_removal(self, mock_readonly_context):
    """Test that missing variables are removed by default."""
    mock_readonly_context.state = {'name': 'Dave'}

    provider = MustacheInstructionProvider(
        'Hello {{name}}, your role is {{role}}.'
    )
    result = await provider(mock_readonly_context)

    # Missing variable {{role}} should be removed
    assert result == 'Hello Dave, your role is .'

  async def test_empty_template(self, mock_readonly_context):
    """Test empty template."""
    provider = MustacheInstructionProvider('')
    result = await provider(mock_readonly_context)

    assert result == ''

  async def test_empty_list(self, mock_readonly_context):
    """Test section with empty list."""
    mock_readonly_context.state = {'servers': []}

    provider = MustacheInstructionProvider(
        '{{#servers}}Active{{/servers}}{{^servers}}No servers{{/servers}}'
    )
    result = await provider(mock_readonly_context)

    assert result == 'No servers'

  async def test_comment_ignored(self, mock_readonly_context):
    """Test that comments are ignored."""
    mock_readonly_context.state = {'name': 'Eve'}

    provider = MustacheInstructionProvider(
        'Hello {{name}}! {{! This is a comment }}'
    )
    result = await provider(mock_readonly_context)

    assert result == 'Hello Eve! '
    assert 'comment' not in result

  async def test_complex_devops_scenario(self, mock_readonly_context):
    """Test complex DevOps scenario."""
    mock_readonly_context.state = {
        'environment': 'PRODUCTION',
        'servers': [
            {
                'id': 'srv-01',
                'role': 'LoadBalancer',
                'cpu': 15,
                'critical': False,
            },
            {'id': 'srv-02', 'role': 'AppServer', 'cpu': 95, 'critical': True},
        ],
    }

    template = """Environment: {{environment}}
{{#servers}}
Active Servers:
{{#servers}}
  - [{{id}}] {{role}}: CPU {{cpu}}%{{#critical}} (CRITICAL!){{/critical}}
{{/servers}}
{{/servers}}
{{^servers}}
No servers active.
{{/servers}}
"""

    provider = MustacheInstructionProvider(template)
    result = await provider(mock_readonly_context)

    assert 'Environment: PRODUCTION' in result
    assert 'srv-01' in result
    assert 'srv-02' in result
    assert 'CRITICAL!' in result

  def test_import_error_when_chevron_not_installed(self, monkeypatch):
    """Test that ImportError is raised when chevron is not installed."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
      if name == 'chevron':
        raise ImportError('No module named chevron')
      return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)

    with pytest.raises(ImportError) as exc_info:
      MustacheInstructionProvider('test template')

    assert 'chevron' in str(exc_info.value).lower()
    assert 'google-adk-community[templating]' in str(exc_info.value)
