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

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

from typing_extensions import override

from .base import BaseTemplateProvider


class Jinja2InstructionProvider(BaseTemplateProvider):
  """Instruction provider using Jinja2 template engine.

  Jinja2 is a modern and designer-friendly templating language for Python.
  It provides powerful features including:
  - Variables: {{ variable }}
  - Control structures: {% if %}, {% for %}
  - Filters: {{ variable|filter }}
  - Template inheritance and macros

  Example:
      ```python
      from google.adk.agents import Agent
      from google.adk_community.templating import Jinja2InstructionProvider

      provider = Jinja2InstructionProvider('''
          You are a {{ role }} assistant.
          Current user: {{ user.name }}

          {% if servers %}
          Active Servers:
          {% for server in servers %}
            - {{ server.name }}: CPU {{ server.cpu }}%
              {% if server.cpu > 80 %}(CRITICAL!){% endif %}
          {% endfor %}
          {% endif %}
      ''')

      agent = Agent(
          name="my_agent",
          model="gemini-2.0-flash",
          instruction=provider
      )
      ```

  See https://jinja.palletsprojects.com/ for full documentation.
  """

  def __init__(
      self,
      template: str,
      custom_filters: Optional[Dict[str, Callable]] = None,
      custom_tests: Optional[Dict[str, Callable]] = None,
      custom_globals: Optional[Dict[str, Any]] = None,
      **jinja_env_kwargs,
  ):
    """Initialize the Jinja2 instruction provider.

    Args:
        template: The Jinja2 template string.
        custom_filters: Optional dictionary of custom Jinja2 filters.
            Example: {'uppercase': str.upper}
        custom_tests: Optional dictionary of custom Jinja2 tests.
            Example: {'even': lambda x: x % 2 == 0}
        custom_globals: Optional dictionary of global variables/functions
            available in templates.
        **jinja_env_kwargs: Additional keyword arguments passed to
            jinja2.Environment constructor (e.g., autoescape, trim_blocks).

    Raises:
        ImportError: If jinja2 is not installed.
    """
    super().__init__(template)

    try:
      import jinja2
    except ImportError as e:
      raise ImportError(
          'Jinja2InstructionProvider requires jinja2. '
          'Install it with: pip install google-adk-community[templating] '
          'or: pip install jinja2'
      ) from e

    # Create Jinja2 environment with user-provided settings
    self.env = jinja2.Environment(**jinja_env_kwargs)

    # Add custom filters
    if custom_filters:
      self.env.filters.update(custom_filters)

    # Add custom tests
    if custom_tests:
      self.env.tests.update(custom_tests)

    # Add custom globals
    if custom_globals:
      self.env.globals.update(custom_globals)

    # Compile the template once during initialization
    self.compiled_template = self.env.from_string(template)

  @override
  async def _render_template(self, context: Dict[str, Any]) -> str:
    """Render the Jinja2 template with the provided context.

    Args:
        context: Dictionary of variables for template rendering.

    Returns:
        The rendered template string.

    Raises:
        jinja2.TemplateError: If template rendering fails.
    """
    return self.compiled_template.render(context)
