# Copyright 2026 Google LLC
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

"""ADK plugin for VAID agent-identity and delegation verification.

Verifies each agent's signed identity (VAID) and delegation chain before the
agent runs, before it is delegated to as a sub-agent, and before every tool
call, denying fail-closed whenever any of that cannot be positively
confirmed. Built on vaid-adk (https://github.com/solara-associates/vaid).

Requires: ``pip install google-adk-community[vaid]``
"""

from __future__ import annotations

from typing import Any
from typing import Optional

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as gtypes


class _VaidIdentityUnavailableError(ImportError):
  """Raised when vaid-adk is not installed."""


try:
  from vaid_adk import ToolAuthorization
  from vaid_adk import TrustedIssuers
  from vaid_adk import VaidAuthPlugin
  from vaid_adk import VaidPresentation
  from vaid_mint.revocation import RevocationCheck
except ImportError as exc:
  raise _VaidIdentityUnavailableError(
      "vaid-adk is required for VaidIdentityPlugin. Install with: "
      "pip install google-adk-community[vaid]"
  ) from exc


class VaidIdentityPlugin(BasePlugin):
  """ADK plugin that verifies a VAID before an agent runs, before it is
  delegated to as a sub-agent, and before every tool call.

  A thin wrapper over :class:`vaid_adk.VaidAuthPlugin`: this class delegates
  every verification decision to it and adds no verification logic of its
  own. See that class's docstring for the fail-closed semantics this plugin
  inherits (an agent with no registered presentation is denied, never
  treated as exempt; a tool with no entry in ``tool_authorizations`` is
  denied, never allowed by omission).

  Args:
      keys: Trusted kernel keys, each bound to the one trust domain it is
          trusted for. Required; there is no default trust configuration.
      revocation: The revocation check to consult for every verified
          lineage. Required; there is no fail-open default.
      tool_authorizations: Maps a tool name to the capability (and,
          optionally, resource) a VAID must carry to call it.
      name: The plugin's registered name on the ``Runner``/``App``.

  Example::

      from google.adk_community.plugins import VaidIdentityPlugin
      from vaid_adk import ToolAuthorization, TrustedIssuers, VaidPresentation

      plugin = VaidIdentityPlugin(
          keys=TrustedIssuers.from_raw_keys([("vaid.example", issuer_public_key)]),
          revocation=my_revocation_check,
          tool_authorizations={"lookup_order": ToolAuthorization(capability="read")},
      )
      plugin.register("worker", VaidPresentation(leaf=child_vaid, chain=(root_vaid,)))
      runner = Runner(agent=my_agent, plugins=[plugin], ...)
  """

  def __init__(
      self,
      *,
      keys: "TrustedIssuers",
      revocation: "RevocationCheck",
      tool_authorizations: Optional[dict[str, "ToolAuthorization"]] = None,
      name: str = "vaid_identity",
  ) -> None:
    super().__init__(name=name)
    self._delegate = VaidAuthPlugin(
        keys=keys,
        revocation=revocation,
        tool_authorizations=tool_authorizations,
        name=name,
    )

  def register(self, agent_name: str, presentation: "VaidPresentation") -> None:
    """Register the VAID presentation for one agent, by its ADK agent
    ``name``. See :meth:`vaid_adk.VaidAuthPlugin.register`."""
    self._delegate.register(agent_name, presentation)

  def unregister(self, agent_name: str) -> None:
    """Drop a registered presentation. See
    :meth:`vaid_adk.VaidAuthPlugin.unregister`."""
    self._delegate.unregister(agent_name)

  async def before_agent_callback(
      self, *, agent: Any, callback_context: Any
  ) -> Optional[gtypes.Content]:
    """Delegates to :meth:`vaid_adk.VaidAuthPlugin.before_agent_callback`."""
    return await self._delegate.before_agent_callback(
        agent=agent, callback_context=callback_context
    )

  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[dict]:
    """Delegates to :meth:`vaid_adk.VaidAuthPlugin.before_tool_callback`."""
    return await self._delegate.before_tool_callback(
        tool=tool, tool_args=tool_args, tool_context=tool_context
    )
