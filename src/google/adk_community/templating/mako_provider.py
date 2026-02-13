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


class MakoInstructionProvider(BaseTemplateProvider):
  """Instruction provider using Mako template engine.

  Mako is a fast, Python-centric templating library that allows you to
  write Python expressions directly in your templates. It provides:
  - Python expressions: ${variable}
  - Control structures: % if, % for
  - Def blocks for reusable components
  - Template inheritance
  - Fast compilation and execution

  Example:
      ```python
      from google.adk.agents import Agent
      from google.adk_community.templating import MakoInstructionProvider

      provider = MakoInstructionProvider('''
          You are a ${role} assistant.
          Current user: ${user.get('name', 'Unknown')}

          % if servers:
          Active Servers:
          % for server in servers:
            - ${server['name']}: CPU ${server['cpu']}%
              % if server['cpu'] > 80:
              (CRITICAL!)
              % endif
          % endfor
          % endif
      ''')

      agent = Agent(
          name="my_agent",
          model="gemini-2.0-flash",
          instruction=provider
      )
      ```

  See https://www.makotemplates.org/ for full documentation.
  """

  def __init__(
      self,
      template: str,
      strict_undefined: bool = False,
      **mako_template_kwargs,
  ):
    """Initialize the Mako instruction provider.

    Args:
        template: The Mako template string.
        strict_undefined: If True, raises an exception when accessing
            undefined variables. If False (default), undefined variables
            evaluate to None or empty string.
        **mako_template_kwargs: Additional keyword arguments passed to
            mako.template.Template constructor (e.g., input_encoding,
            output_encoding, error_handler).

    Raises:
        ImportError: If Mako is not installed.
    """
    super().__init__(template)

    try:
      from mako.template import Template
    except ImportError as e:
      raise ImportError(
          'MakoInstructionProvider requires Mako. '
          'Install it with: pip install google-adk-community[templating] '
          'or: pip install Mako'
      ) from e

    # Set strict_undefined as a Mako-specific setting
    if strict_undefined:
      mako_template_kwargs['strict_undefined'] = True

    # Compile the template once during initialization
    self.compiled_template = Template(text=template, **mako_template_kwargs)

  @override
  async def _render_template(self, context: Dict[str, Any]) -> str:
    """Render the Mako template with the provided context.

    Args:
        context: Dictionary of variables for template rendering.

    Returns:
        The rendered template string.

    Raises:
        mako.exceptions.MakoException: If template rendering fails.
    """
    return self.compiled_template.render(**context)
