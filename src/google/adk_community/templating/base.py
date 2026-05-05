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

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from google.adk.agents.readonly_context import ReadonlyContext


class BaseTemplateProvider(ABC):
  """Base class for template-based instruction providers.

  This class provides common functionality for all templating engines,
  including state extraction and error handling patterns.

  Subclasses must implement the `_render_template` method to integrate
  with their specific templating engine.
  """

  def __init__(self, template: str):
    """Initialize the template provider.

    Args:
        template: The template string to render.
    """
    self.template = template

  async def __call__(self, context: ReadonlyContext) -> str:
    """Render the template with session state from context.

    This method is called by the ADK framework to generate dynamic
    instructions for agents.

    Args:
        context: The readonly context containing session state.

    Returns:
        The rendered instruction string.
    """
    render_context = self._extract_context(context)
    return await self._render_template(render_context)

  def _extract_context(self, context: ReadonlyContext) -> Dict[str, Any]:
    """Extract rendering context from ADK's ReadonlyContext.

    This method provides access to:
    - Session state variables
    - User ID
    - Session ID
    - App name

    Args:
        context: The readonly context from ADK.

    Returns:
        Dictionary of variables available for template rendering.
    """
    # Access the invocation context to get session information
    invocation_context = context._invocation_context
    session = invocation_context.session

    # Build the render context with session state and metadata
    render_context = dict(context.state)

    # Add session metadata as special variables
    # Note: Using 'adk_' prefix instead of '__' to avoid conflicts with
    # templating engines that don't allow variables starting with underscores
    render_context['adk_session_id'] = session.id
    render_context['adk_user_id'] = session.user_id
    render_context['adk_app_name'] = session.app_name

    return render_context

  @abstractmethod
  async def _render_template(self, context: Dict[str, Any]) -> str:
    """Render the template using the specific templating engine.

    Subclasses must implement this method to integrate with their
    templating engine (Jinja2, Mako, Mustache, Django, etc.).

    Args:
        context: Dictionary of variables for template rendering.

    Returns:
        The rendered template string.

    Raises:
        Any templating engine-specific exceptions.
    """
    pass
