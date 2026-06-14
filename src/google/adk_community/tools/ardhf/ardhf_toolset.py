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

The toolset exposes the following tools to the agent:

* ``search_ards`` — search ARD registries across all artifact types
  (agents, skills, MCP servers, spaces).
* ``search_agents`` — convenience alias: search filtered to A2A agents.
* ``search_skills`` — convenience alias: search filtered to skills.
* ``search_tools`` — convenience alias: search filtered to MCP servers.
* ``search_spaces`` — convenience alias: search filtered to HF Spaces.
* ``get_agent_card`` — fetch a specific artifact (agent card, skill
  markdown, MCP server descriptor) by URL.
* ``connect_agent`` — send a message to a remote A2A agent and return
  the response, enabling the full discover → connect → use flow.

Reference: https://github.com/huggingface/hf-agentfinder
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, List, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
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


def _extract_text_from_a2a_response(a2a_response: Any) -> list[str]:
  """Extract text strings from an A2A client response event.

  The response from ``A2AClient.send_message`` can be either a tuple
  of ``(Task, update)`` or an ``A2AMessage``.  This helper walks the
  parts and collects all text content.
  """
  texts: list[str] = []

  try:
    from a2a.types import Message as A2AMessage
    from a2a.types import TextPart as A2ATextPart
  except ImportError:
    return texts

  def _extract_from_parts(
      parts: Optional[list[Any]],
  ) -> None:
    if not parts:
      return
    for part in parts:
      root = getattr(part, "root", None)
      if isinstance(root, A2ATextPart):
        texts.append(root.text)

  if isinstance(a2a_response, tuple):
    task = a2a_response[0]
    if task is not None:
      # Extract from task artifacts.
      for artifact in getattr(task, "artifacts", None) or []:
        _extract_from_parts(getattr(artifact, "parts", None))
      # Extract from task status message.
      status = getattr(task, "status", None)
      if status is not None:
        status_msg = getattr(status, "message", None)
        if status_msg is not None:
          _extract_from_parts(
              getattr(status_msg, "parts", None)
          )
  elif isinstance(a2a_response, A2AMessage):
    _extract_from_parts(getattr(a2a_response, "parts", None))

  return texts


class AgentFinderToolset(BaseToolset):
  """ADK BaseToolset wrapping HuggingFace Agent Finder (ARD).

  Provides ``search_ards``, ``search_agents``, ``search_skills``,
  ``search_tools``, ``search_spaces``, ``get_agent_card``, and
  ``connect_agent`` tools to any ADK agent, enabling the full
  *discover → inspect → connect* workflow.

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

  # -- Internal search logic -----------------------------------------------

  async def _do_search(
      self,
      query: str,
      artifact_type: Optional[str] = None,
      limit: int = 10,
  ) -> dict[str, Any]:
    """Core search logic shared by all search tools."""
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

  # -- Tool implementations -----------------------------------------------

  async def _search_ards(
      self,
      tool_context: ToolContext,
      query: str,
      artifact_type: Optional[str] = None,
      limit: int = 10,
  ) -> dict[str, Any]:
    """Search ARD registries across all artifact types.

    Args:
      query: Natural-language search query describing what you need,
          e.g. "remove image background" or "code review".
      artifact_type: Optional filter by artifact kind.  Supported
          values: ``skill``, ``mcp``, ``space``, ``a2a``, or a raw
          media type like ``application/mcp-server+json``.  When
          omitted, all artifact types are returned.
      limit: Maximum number of results to return (1-100, default 10).

    Returns:
      A dictionary with ``results`` (list of matching entries) and
      optionally ``referrals`` (list of additional registries).
    """
    return await self._do_search(
        query, artifact_type=artifact_type, limit=limit
    )

  async def _search_agents(
      self,
      tool_context: ToolContext,
      query: str,
      limit: int = 10,
  ) -> dict[str, Any]:
    """Search ARD registries for A2A agents only.

    Args:
      query: Natural-language search query describing the agent
          capability you need, e.g. "code review" or "translation".
      limit: Maximum number of results to return (1-100, default 10).

    Returns:
      A dictionary with ``results`` filtered to A2A agents and
      optionally ``referrals``.
    """
    return await self._do_search(query, artifact_type="a2a", limit=limit)

  async def _search_skills(
      self,
      tool_context: ToolContext,
      query: str,
      limit: int = 10,
  ) -> dict[str, Any]:
    """Search ARD registries for skills only.

    Args:
      query: Natural-language search query describing the skill you
          need, e.g. "code review" or "triage issues".
      limit: Maximum number of results to return (1-100, default 10).

    Returns:
      A dictionary with ``results`` filtered to skills and optionally
      ``referrals``.
    """
    return await self._do_search(
        query, artifact_type="skill", limit=limit
    )

  async def _search_tools(
      self,
      tool_context: ToolContext,
      query: str,
      limit: int = 10,
  ) -> dict[str, Any]:
    """Search ARD registries for MCP servers only.

    Args:
      query: Natural-language search query describing the tool you
          need, e.g. "database query" or "image processing".
      limit: Maximum number of results to return (1-100, default 10).

    Returns:
      A dictionary with ``results`` filtered to MCP servers and
      optionally ``referrals``.
    """
    return await self._do_search(query, artifact_type="mcp", limit=limit)

  async def _search_spaces(
      self,
      tool_context: ToolContext,
      query: str,
      limit: int = 10,
  ) -> dict[str, Any]:
    """Search ARD registries for HuggingFace Spaces only.

    Args:
      query: Natural-language search query describing the Space you
          need, e.g. "text to speech" or "image generation".
      limit: Maximum number of results to return (1-100, default 10).

    Returns:
      A dictionary with ``results`` filtered to HuggingFace Spaces and
      optionally ``referrals``.
    """
    return await self._do_search(
        query, artifact_type="space", limit=limit
    )

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

  async def _connect_agent(
      self,
      tool_context: ToolContext,
      agent_card_url: str,
      message: str,
  ) -> dict[str, Any]:
    """Send a message to a remote A2A agent and return the response.

    Use this after ``search_agents`` and ``get_agent_card`` to
    interact with a discovered A2A agent.  The agent card URL should
    be for an artifact with media type
    ``application/a2a-agent-card+json``.

    Args:
      agent_card_url: Full URL to the remote agent's A2A agent card
          (typically the ``url`` field from a search result whose
          ``type`` is ``application/a2a-agent-card+json``).
      message: The message to send to the remote agent.

    Returns:
      A dictionary with ``response`` (the agent's reply text),
      ``agent_name``, and ``agent_url``.  On failure, returns a
      dictionary with an ``error`` key.
    """
    try:
      from a2a.client.card_resolver import (
          A2ACardResolver,
      )
      from a2a.client.client import (
          ClientConfig as A2AClientConfig,
      )
      from a2a.client.client_factory import (
          ClientFactory as A2AClientFactory,
      )
      from a2a.types import Message as A2AMessage
      from a2a.types import Part as A2APart
      from a2a.types import TextPart as A2ATextPart
      import httpx
    except ImportError:
      return {
          "error": (
              "A2A dependencies are not installed.  Install them"
              " with:  pip install 'google-adk[a2a]'"
          )
      }

    try:
      parsed = urlparse(agent_card_url)
      if not parsed.scheme or not parsed.netloc:
        return {"error": f"Invalid agent card URL: {agent_card_url}"}

      base_url = f"{parsed.scheme}://{parsed.netloc}"
      relative_path = parsed.path

      async with httpx.AsyncClient(
          timeout=httpx.Timeout(timeout=float(_HTTP_TIMEOUT))
      ) as http_client:
        # Resolve the agent card.
        resolver = A2ACardResolver(
            httpx_client=http_client,
            base_url=base_url,
        )
        agent_card = await resolver.get_agent_card(
            relative_card_path=relative_path,
        )

        # Create an A2A client for this agent.
        factory = A2AClientFactory(
            config=A2AClientConfig(httpx_client=http_client),
        )
        a2a_client = factory.create(agent_card)

        # Build and send the message.
        request = A2AMessage(
            message_id=str(uuid.uuid4()),
            parts=[A2APart(root=A2ATextPart(text=message))],
            role="user",
        )

        response_texts: list[str] = []
        async for a2a_response in a2a_client.send_message(
            request=request,
        ):
          response_texts.extend(
              _extract_text_from_a2a_response(a2a_response)
          )

        agent_name = getattr(agent_card, "name", "unknown")
        response_text = "\n".join(response_texts) if response_texts else ""

        return {
            "response": response_text,
            "agent_name": agent_name,
            "agent_url": agent_card_url,
        }

    except Exception as exc:
      logger.warning(
          "A2A connect failed for %s: %s", agent_card_url, exc
      )
      return {
          "error": (
              f"Failed to communicate with agent at"
              f" {agent_card_url}: {exc}"
          ),
      }

  # -- BaseToolset interface ----------------------------------------------

  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> list[BaseTool]:
    """Return all search, inspect, and connect tools."""
    tools: list[BaseTool] = [
        FunctionTool(self._search_ards),
        FunctionTool(self._search_agents),
        FunctionTool(self._search_skills),
        FunctionTool(self._search_tools),
        FunctionTool(self._search_spaces),
        FunctionTool(self._get_agent_card),
        FunctionTool(self._connect_agent),
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
