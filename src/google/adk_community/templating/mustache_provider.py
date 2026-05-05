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
from typing import Dict

from typing_extensions import override

from .base import BaseTemplateProvider


class MustacheInstructionProvider(BaseTemplateProvider):
  """Instruction provider using Mustache template engine.

  Mustache is a logic-less templating language that works across many
  programming languages. It provides:
  - Variables: {{variable}}
  - Sections: {{#section}}...{{/section}}
  - Inverted sections: {{^section}}...{{/section}}
  - Comments: {{! comment }}
  - Partials: {{> partial}}

  This implementation uses the 'chevron' library, a Python implementation
  of Mustache.

  Example:
      ```python
      from google.adk.agents import Agent
      from google.adk_community.templating import MustacheInstructionProvider

      provider = MustacheInstructionProvider('''
          You are a {{role}} assistant.
          Current user: {{user.name}}

          {{#servers}}
          Active Servers:
          {{#servers}}
            - {{name}}: CPU {{cpu}}%
              {{#high_cpu}}(CRITICAL!){{/high_cpu}}
          {{/servers}}
          {{/servers}}
          {{^servers}}
          No servers currently active.
          {{/servers}}
      ''')

      agent = Agent(
          name="my_agent",
          model="gemini-2.0-flash",
          instruction=provider
      )
      ```

  See https://mustache.github.io/ and https://github.com/noahmorrison/chevron
  for full documentation.
  """

  def __init__(
      self,
      template: str,
      warn: bool = False,
  ):
    """Initialize the Mustache instruction provider.

    Args:
        template: The Mustache template string.
        warn: If True, prints a warning to stderr when a tag is not found
            in the data. Default is False.

    Raises:
        ImportError: If chevron is not installed.
    """
    super().__init__(template)

    try:
      import chevron
    except ImportError as e:
      raise ImportError(
          'MustacheInstructionProvider requires chevron. '
          'Install it with: pip install google-adk-community[templating] '
          'or: pip install chevron'
      ) from e

    self.warn = warn
    self._chevron = chevron

  @override
  async def _render_template(self, context: Dict[str, Any]) -> str:
    """Render the Mustache template with the provided context.

    Args:
        context: Dictionary of variables for template rendering.

    Returns:
        The rendered template string.
    """
    return self._chevron.render(self.template, context, warn=self.warn)
