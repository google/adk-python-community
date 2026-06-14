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

"""ARDHF toolset — wraps HuggingFace Agent Finder (ARD) as BaseToolset.

Supports two modes:

* **remote** (default) — HTTP POST to any ARD-compatible registry
  endpoint (e.g. the hosted ``hf-agentfinder``).
* **local** — uses the ``agentfinder`` Python package in-process for
  zero-latency, offline-capable search.

The toolset exposes two tools to the agent:

* ``search_agents`` — search ARD registries for agents, skills, MCP
  servers, and other agentic resources.
* ``get_agent_card`` — fetch a specific artifact (agent card, skill
  markdown, MCP server descriptor) by URL.

Prepared for rename to ARD (Agentic Resource Discovery).
Reference: https://github.com/huggingface/hf-agentfinder
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset, ToolPredicate
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)

# Default hosted HuggingFace Agent Finder registry.
_DEFAULT_REGISTRY_URL = "https://huggingface.co/api/agentfinder"

# HTTP timeout for remote requests (seconds).
_HTTP_TIMEOUT = 30


def _registry_search_url(registry_url: str) -> str:
  """Normalise a registry base URL to its ``/search`` endpoint."""
  normalised = registry_url.rstrip("/")
  if normalised.endswith("/search"):
    return normalised
  return urljoin(f"{normalised}/", "search")


def _artifact_type_for_kind(kind: str) -> Optional[str]:
  """Map a human-friendly kind label to its ARD media type."""
  types = {
      "skill": "application/ai-skill",
      "mcp": "application/mcp-server+json",
      "space": "application/vnd.huggingface.space+json",
      "a2a": "application/a2a-agent-card+json",
  }
  return types.get(kind)


def _remote_search(
    registry_url: str,
    query: str,
    *,
    artifact_type: Optional[str] = None,
    limit: int = 10,
    token: Optional[str] = None,
) -> dict[str, Any]:
  """POST a SearchRequest to a remote ARD registry and return raw JSON."""
  search_query: dict[str, Any] = {"text": query}
  if artifact_type is not None:
    search_query["filter"] = {"type": [artifact_type]}

  request_body = {
      "query": search_query,
      "pageSize": limit,
  }

  headers = {
      "Accept": "application/json",
      "Content-Type": "application/json",
      "User-Agent": "adk-ardhf/0.1",
  }
  if token is not None:
    headers["Authorization"] = f"Bearer {token}"

  url = _registry_search_url(registry_url)
  req = UrlRequest(
      url,
      data=json.dumps(request_body).encode("utf-8"),
      headers=headers,
      method="POST",
  )
  with urlopen(req, timeout=_HTTP_TIMEOUT) as response:  # noqa: S310
    return json.loads(response.read().decode("utf-8"))


def _remote_fetch(
    url: str, *, token: Optional[str] = None
) -> str:
  """GET an artifact URL and return its text content."""
  headers = {"User-Agent": "adk-ardhf/0.1"}
  if token is not None:
    headers["Authorization"] = f"Bearer {token}"

  req = UrlRequest(url, headers=headers)
  with urlopen(req, timeout=_HTTP_TIMEOUT) as response:  # noqa: S310
    return response.read().decode("utf-8")


def _local_search(
    query: str,
    *,
    artifact_type: Optional[str] = None,
    limit: int = 10,
    token: Optional[str] = None,
) -> dict[str, Any]:
  """Search using the in-process ``agentfinder`` package."""
  try:
    from agentfinder.models import SearchQuery, SearchRequest
    from agentfinder.server import search_agent_finder
  except ImportError as exc:
    raise ImportError(
        "Local mode requires the 'hf-agentfinder' package. "
        "Install it with: pip install hf-agentfinder"
    ) from exc

  search_filter: dict[str, Any] = {}
  if artifact_type is not None:
    search_filter["type"] = [artifact_type]

  request = SearchRequest(
      query=SearchQuery(text=query, filter=search_filter),
      pageSize=limit,
  )
  response = search_agent_finder(request, token=token)
  return json.loads(
      response.model_dump_json(
          exclude_none=True, exclude_defaults=True
      )
  )


class AgentFinderToolset(BaseToolset):
  """ADK BaseToolset wrapping HuggingFace Agent Finder (ARD).

  Provides ``search_agents`` and ``get_agent_card`` tools to any ADK
  agent.

  Args:
    registry_url: ARD registry URL for remote mode.  Ignored when
        ``local=True``.
    token: Optional Bearer token for authenticated registry access.
    local: When ``True``, use the ``agentfinder`` Python package
        in-process instead of making HTTP requests.
    tool_filter: Optional filter to select which tools are exposed.
    tool_name_prefix: Optional prefix for tool names.
  """

  def __init__(
      self,
      *,
      registry_url: str = _DEFAULT_REGISTRY_URL,
      token: Optional[str] = None,
      local: bool = False,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
      tool_name_prefix: Optional[str] = None,
  ) -> None:
    super().__init__(
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
    )
    self._registry_url = registry_url
    self._token = token
    self._local = local

  # -- Tool implementations -----------------------------------------------

  async def _search_agents(
      self,
      tool_context: ToolContext,
      query: str,
      artifact_type: Optional[str] = None,
      limit: int = 10,
  ) -> dict[str, Any]:
    """Search ARD registries for agents, skills, and MCP servers.

    Args:
      query: Natural-language search query describing what you need,
          e.g. "remove image background" or "code review agent".
      artifact_type: Optional filter by artifact kind.  Supported
          values: ``skill``, ``mcp``, ``space``, ``a2a``, or a raw
          media type like ``application/mcp-server+json``.  When
          omitted, all artifact types are returned.
      limit: Maximum number of results to return (1-100, default 10).

    Returns:
      A dictionary with ``results`` (list of matching entries) and
      optionally ``referrals`` (list of additional registries).
    """
    # Resolve human-friendly kind to media type.
    resolved_type = (
        _artifact_type_for_kind(artifact_type) if artifact_type else None
    )
    if resolved_type is None and artifact_type is not None:
      # Assume it is already a raw media type string.
      resolved_type = artifact_type

    try:
      if self._local:
        return _local_search(
            query,
            artifact_type=resolved_type,
            limit=limit,
            token=self._token,
        )
      return _remote_search(
          self._registry_url,
          query,
          artifact_type=resolved_type,
          limit=limit,
          token=self._token,
      )
    except (HTTPError, URLError, TimeoutError) as exc:
      logger.warning("ARD search failed: %s", exc)
      return {"error": f"Search request failed: {exc}"}
    except ImportError as exc:
      return {"error": str(exc)}

  async def _get_agent_card(
      self,
      tool_context: ToolContext,
      url: str,
  ) -> dict[str, Any]:
    """Fetch a specific agent card or artifact by URL.

    Args:
      url: The full URL of the artifact to fetch.  This is typically
          the ``url`` field from a search result entry.

    Returns:
      A dictionary with the artifact content.  For markdown artifacts
      (skills), the content is returned under a ``content`` key.
      For JSON artifacts, the parsed object is returned directly.
    """
    try:
      raw = _remote_fetch(url, token=self._token)
    except (HTTPError, URLError, TimeoutError) as exc:
      logger.warning("ARD fetch failed for %s: %s", url, exc)
      return {"error": f"Failed to fetch {url}: {exc}"}

    # Try to parse as JSON; fall back to returning raw text.
    try:
      return json.loads(raw)
    except json.JSONDecodeError:
      return {
          "content": raw,
          "url": url,
          "content_type": "text/markdown",
      }

  # -- BaseToolset interface ----------------------------------------------

  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> list[BaseTool]:
    """Return the search_agents and get_agent_card tools."""
    tools: list[BaseTool] = [
        FunctionTool(self._search_agents),
        FunctionTool(self._get_agent_card),
    ]

    # Rename the tools to use cleaner names (strip leading underscore).
    for tool in tools:
      if tool.name.startswith("_"):
        tool.name = tool.name[1:]

    if readonly_context is not None:
      tools = [
          tool
          for tool in tools
          if self._is_tool_selected(tool, readonly_context)
      ]

    return tools
