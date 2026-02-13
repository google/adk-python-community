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

from google.adk_community.templating import MakoInstructionProvider


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


class TestMakoInstructionProvider:
  """Test suite for MakoInstructionProvider."""

  async def test_basic_variable_substitution(self, mock_readonly_context):
    """Test basic variable substitution."""
    mock_readonly_context.state = {'name': 'Alice', 'role': 'Engineer'}

    provider = MakoInstructionProvider('Hello ${name}, you are a ${role}.')
    result = await provider(mock_readonly_context)

    assert result == 'Hello Alice, you are a Engineer.'

  async def test_nested_object_access(self, mock_readonly_context):
    """Test accessing nested objects using dictionary access."""
    mock_readonly_context.state = {
        'user': {'name': 'Bob', 'profile': {'age': 30, 'role': 'Developer'}}
    }

    provider = MakoInstructionProvider(
        "User: ${user['name']}, Role: ${user['profile']['role']}"
    )
    result = await provider(mock_readonly_context)

    assert result == 'User: Bob, Role: Developer'

  async def test_control_structures(self, mock_readonly_context):
    """Test if/else control structures."""
    mock_readonly_context.state = {'logged_in': True, 'username': 'Charlie'}

    provider = MakoInstructionProvider(
        '% if logged_in:\nWelcome ${username}!\n% else:\nPlease log in.\n%'
        ' endif'
    )
    result = await provider(mock_readonly_context)

    assert 'Welcome Charlie!' in result

  async def test_for_loop(self, mock_readonly_context):
    """Test for loops."""
    mock_readonly_context.state = {
        'servers': [
            {'name': 'srv-01', 'cpu': 25},
            {'name': 'srv-02', 'cpu': 90},
        ]
    }

    provider = MakoInstructionProvider(
        "% for server in servers:\n${server['name']}: ${server['cpu']}%\n%"
        ' endfor'
    )
    result = await provider(mock_readonly_context)

    assert 'srv-01: 25%' in result
    assert 'srv-02: 90%' in result

  async def test_python_expressions(self, mock_readonly_context):
    """Test Python expressions in templates."""
    mock_readonly_context.state = {'x': 10, 'y': 5}

    provider = MakoInstructionProvider('Result: ${x + y}')
    result = await provider(mock_readonly_context)

    assert result == 'Result: 15'

  async def test_session_metadata_access(self, mock_readonly_context):
    """Test access to session metadata."""
    provider = MakoInstructionProvider(
        'Session: ${adk_session_id}, User: ${adk_user_id}, App: ${adk_app_name}'
    )
    result = await provider(mock_readonly_context)

    assert result == 'Session: test-session-id, User: test-user, App: test-app'

  async def test_get_method_with_default(self, mock_readonly_context):
    """Test using .get() method for safe access."""
    mock_readonly_context.state = {'user': {'name': 'Dave'}}

    provider = MakoInstructionProvider(
        "Name: ${user.get('name', 'Unknown')}, Role: ${user.get('role', 'N/A')}"
    )
    result = await provider(mock_readonly_context)

    assert result == 'Name: Dave, Role: N/A'

  async def test_empty_template(self, mock_readonly_context):
    """Test empty template."""
    provider = MakoInstructionProvider('')
    result = await provider(mock_readonly_context)

    assert result == ''

  async def test_conditional_with_comparison(self, mock_readonly_context):
    """Test conditional with comparison operators."""
    mock_readonly_context.state = {'cpu': 95}

    provider = MakoInstructionProvider(
        '% if cpu > 80:\nCRITICAL: CPU at ${cpu}%\n% else:\nNormal: CPU at'
        ' ${cpu}%\n% endif'
    )
    result = await provider(mock_readonly_context)

    assert 'CRITICAL: CPU at 95%' in result

  async def test_complex_devops_scenario(self, mock_readonly_context):
    """Test complex DevOps scenario."""
    mock_readonly_context.state = {
        'environment': 'PRODUCTION',
        'servers': [
            {'id': 'srv-01', 'role': 'LoadBalancer', 'cpu': 15},
            {'id': 'srv-02', 'role': 'AppServer', 'cpu': 95},
        ],
    }

    template = """Environment: ${environment}
% if servers:
Active Servers:
% for server in servers:
  - [${server['id']}] ${server['role']}: CPU ${server['cpu']}%\\
% if server['cpu'] > 80:
 (CRITICAL!)\\
% endif

% endfor
% else:
No servers active.
% endif
"""

    provider = MakoInstructionProvider(template)
    result = await provider(mock_readonly_context)

    assert 'Environment: PRODUCTION' in result
    assert 'srv-01' in result
    assert 'srv-02' in result
    assert 'CRITICAL!' in result

  def test_import_error_when_mako_not_installed(self, monkeypatch):
    """Test that ImportError is raised when Mako is not installed."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
      if name == 'mako.template' or name == 'mako':
        raise ImportError('No module named mako')
      return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)

    with pytest.raises(ImportError) as exc_info:
      MakoInstructionProvider('test template')

    assert 'mako' in str(exc_info.value).lower()
    assert 'google-adk-community[templating]' in str(exc_info.value)
