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
from typing import Optional

from typing_extensions import override

from .base import BaseTemplateProvider


class DjangoInstructionProvider(BaseTemplateProvider):
  """Instruction provider using Django template engine.

  Django's template language is designed to strike a balance between power
  and ease. It provides:
  - Variables: {{ variable }}
  - Tags: {% if %}, {% for %}, {% block %}
  - Filters: {{ variable|filter }}
  - Template inheritance
  - Automatic HTML escaping (configurable)

  Example:
      ```python
      from google.adk.agents import Agent
      from google.adk_community.templating import DjangoInstructionProvider

      provider = DjangoInstructionProvider('''
          You are a {{ role }} assistant.
          Current user: {{ user.name|default:"Unknown" }}

          {% if servers %}
          Active Servers:
          {% for server in servers %}
            - {{ server.name }}: CPU {{ server.cpu }}%
              {% if server.cpu > 80 %}(CRITICAL!){% endif %}
          {% endfor %}
          {% else %}
          No servers currently active.
          {% endif %}
      ''')

      agent = Agent(
          name="my_agent",
          model="gemini-2.0-flash",
          instruction=provider
      )
      ```

  See https://docs.djangoproject.com/en/stable/ref/templates/ for full
  documentation.
  """

  def __init__(
      self,
      template: str,
      autoescape: bool = False,
      custom_filters: Optional[Dict[str, Any]] = None,
      custom_tags: Optional[Dict[str, Any]] = None,
  ):
    """Initialize the Django instruction provider.

    Args:
        template: The Django template string.
        autoescape: If True, enables HTML autoescaping. Default is False
            since instructions are typically plain text.
        custom_filters: Optional dictionary of custom template filters.
        custom_tags: Optional dictionary of custom template tags.

    Raises:
        ImportError: If Django is not installed.
    """
    super().__init__(template)

    try:
      from django import conf
      from django.template import Context
      from django.template import Engine
      from django.template.library import Library
    except ImportError as e:
      raise ImportError(
          'DjangoInstructionProvider requires Django. '
          'Install it with: pip install google-adk-community[templating] '
          'or: pip install Django'
      ) from e

    # Configure Django settings if not already configured
    # This is needed for features like localization and number formatting
    if not conf.settings.configured:
      conf.settings.configure(
          USE_I18N=False,  # Disable internationalization for simple use case
          USE_L10N=False,  # Disable localization
          USE_TZ=False,  # Disable timezone support
      )

    # Create a Django template engine
    # We don't need a full Django setup, just the template engine
    self.engine = Engine(autoescape=autoescape)

    # Register custom filters and tags if provided
    if custom_filters or custom_tags:
      library = Library()
      if custom_filters:
        for name, func in custom_filters.items():
          library.filter(name, func)
      if custom_tags:
        for name, func in custom_tags.items():
          library.tag(name, func)
      # Add library to engine's built-in libraries
      self.engine.template_libraries['custom'] = library
      # Load the custom library in templates by default
      self.engine.builtins.append('custom')

    # Store Django classes and settings for use in render
    self._Context = Context
    self._autoescape = autoescape

    # Compile the template once during initialization
    self.compiled_template = self.engine.from_string(template)

  @override
  async def _render_template(self, context: Dict[str, Any]) -> str:
    """Render the Django template with the provided context.

    Args:
        context: Dictionary of variables for template rendering.

    Returns:
        The rendered template string.

    Raises:
        django.template.TemplateSyntaxError: If template has syntax errors.
        django.template.TemplateDoesNotExist: If a referenced template
            (e.g., via {% extends %}) does not exist.
    """
    django_context = self._Context(context, autoescape=self._autoescape)
    return self.compiled_template.render(django_context)
