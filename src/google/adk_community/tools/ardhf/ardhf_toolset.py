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

"""ARDHF toolset — wraps HuggingFace Discover (ARD) as BaseToolset.

Supports two modes:

* **remote** (default) — HTTP POST to any ARD-compatible registry
  endpoint (e.g. the hosted ``hf-discover``).
* **local** — uses the ``discover`` Python package in-process for
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

Reference: https://github.com/huggingface/hf-discover
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any
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

# Default hosted HuggingFace Discover registry.
_DEFAULT_REGISTRY_URL = "https://huggingface-hf-discover.hf.space"

# HTTP timeout for remote requests (seconds).
_HTTP_TIMEOUT = 30

# Default allowed URL schemes (secure by default).
_DEFAULT_ALLOWED_SCHEMES = frozenset(("http", "https"))


def _registry_search_url(registry_url: str) -> str:
  """Normalise a registry base URL to its ``/search`` endpoint."""
  normalised = registry_url.rstrip("/")
  if normalised.endswith("/search"):
    return normalised
  return urljoin(f"{normalised}/", "search")


_KIND_TO_MEDIA_TYPES: dict[str, list[str]] = {
    "skill": ["application/ai-skill"],
    "mcp": [
        "application/mcp-server-card+json",
        "application/mcp-server+json",
    ],
    "space": ["application/vnd.huggingface.space+json"],
    "a2a": ["application/a2a-agent-card+json"],
}


def _artifact_types_for_kind(kind: str) -> list[str] | None:
  """Map a human-friendly kind label to its ARD media type(s)."""
  return _KIND_TO_MEDIA_TYPES.get(kind)


def _remote_search(
    registry_url: str,
    query: str,
    *,
    artifact_types: list[str] | None = None,
    limit: int = 10,
    token: str | None = None,
) -> dict[str, Any]:
  """POST a SearchRequest to a remote ARD registry and return raw JSON."""
  search_query: dict[str, Any] = {"text": query}
  if artifact_types is not None:
    search_query["filter"] = {"type": artifact_types}

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
    url: str, *, token: str | None = None
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
    artifact_type: str | None = None,
    limit: int = 10,
    token: str | None = None,
) -> dict[str, Any]:
  """Search using the in-process ``discover`` package."""
  try:
    from discover.models import SearchQuery, SearchRequest
    from discover.server import search_discover
  except ImportError as exc:
    raise ImportError(
        "Local mode requires the 'hf-discover' package. "
        "Install it with: pip install hf-discover"
    ) from exc

  search_filter: dict[str, Any] = {}
  if artifact_type is not None:
    search_filter["type"] = [artifact_type]

  request = SearchRequest(
      query=SearchQuery(text=query, filter=search_filter),
      pageSize=limit,
  )
  response = search_discover(request, token=token)
  return response.model_dump(
      exclude_none=True, exclude_defaults=True
  )


def _extract_text_from_a2a_response(a2a_response: Any) -> list[str]:
  """Extract text strings from an A2A client StreamResponse.

  The a2a-sdk 1.x ``send_message`` yields ``StreamResponse`` protobuf
  messages.  Each may contain a ``task`` (with artifacts and status) or
  a ``message`` (with parts).  This helper walks through and collects
  all text content.
  """
  texts: list[str] = []

  def _extract_from_parts(
      parts: Any,
  ) -> None:
    if not parts:
      return
    for part in parts:
      # Protobuf Part: check if the 'text' oneof is set.
      if hasattr(part, "HasField") and part.HasField("text"):
        texts.append(part.text)
      # Pydantic Part (older SDKs): check for root.text.
      elif hasattr(part, "root") and hasattr(part.root, "text"):
        texts.append(part.root.text)

  # StreamResponse has oneof: task, message, status_update, artifact_update.
  if hasattr(a2a_response, "HasField"):
    # Protobuf StreamResponse
    if a2a_response.HasField("task"):
      task = a2a_response.task
      for artifact in task.artifacts:
        _extract_from_parts(artifact.parts)
      if task.HasField("status") and task.status.HasField("message"):
        _extract_from_parts(task.status.message.parts)
    elif a2a_response.HasField("message"):
      _extract_from_parts(a2a_response.message.parts)
  elif isinstance(a2a_response, tuple):
    # Legacy tuple format (Task, update)
    task = a2a_response[0]
    if task is not None:
      for artifact in getattr(task, "artifacts", None) or []:
        _extract_from_parts(getattr(artifact, "parts", None))
      status = getattr(task, "status", None)
      if status is not None:
        status_msg = getattr(status, "message", None)
        if status_msg is not None:
          _extract_from_parts(getattr(status_msg, "parts", None))

  return texts


class AgentFinderToolset(BaseToolset):
  """ADK BaseToolset wrapping HuggingFace Discover (ARD).

  Provides ``search_ards``, ``search_agents``, ``search_skills``,
  ``search_tools``, ``search_spaces``, ``get_agent_card``, and
  ``connect_agent`` tools to any ADK agent, enabling the full
  *discover → inspect → connect* workflow.

  Args:
    registry_url: ARD registry URL for remote mode.  Ignored when
        ``local=True``.
    token: Optional Bearer token for authenticated registry access.
    local: When ``True``, use the ``discover`` Python package
        in-process instead of making HTTP requests.
    allowed_schemes: URL schemes permitted for ``get_agent_card``
        and ``connect_agent``.  Defaults to ``("http", "https")``
        for security (prevents SSRF via ``file://`` etc.).  Set to
        ``("http", "https", "file")`` for local development or
        ``("http", "https", "grpc", "grpcs")`` for gRPC support.
    tool_filter: Optional filter to select which tools are exposed.
    tool_name_prefix: Optional prefix for tool names.
  """

  def __init__(
      self,
      *,
      registry_url: str = _DEFAULT_REGISTRY_URL,
      token: str | None = None,
      local: bool = False,
      allowed_schemes: tuple[str, ...] | list[str] | None = None,
      tool_filter: ToolPredicate | list[str] | None = None,
      tool_name_prefix: str | None = None,
  ) -> None:
    super().__init__(
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
    )
    self._registry_url = registry_url
    self._token = token
    self._local = local
    self._allowed_schemes = frozenset(
        allowed_schemes if allowed_schemes is not None
        else _DEFAULT_ALLOWED_SCHEMES
    )

  # -- Internal search logic -----------------------------------------------

  async def _do_search(
      self,
      query: str,
      artifact_type: str | None = None,
      limit: int = 10,
  ) -> dict[str, Any]:
    """Core search logic shared by all search tools."""
    # Clamp limit to the valid range.
    limit = max(1, min(limit, 100))

    # Resolve human-friendly kind to media type(s).
    resolved_types = (
        _artifact_types_for_kind(artifact_type) if artifact_type else None
    )
    if resolved_types is None and artifact_type is not None:
      # Assume it is already a raw media type string.
      resolved_types = [artifact_type]

    try:
      if self._local:
        return await asyncio.to_thread(
            _local_search,
            query,
            artifact_type=resolved_types[0] if resolved_types else None,
            limit=limit,
            token=self._token,
        )
      return await asyncio.to_thread(
          _remote_search,
          self._registry_url,
          query,
          artifact_types=resolved_types,
          limit=limit,
          token=self._token,
      )
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
      logger.warning("ARD search failed: %s", exc)
      return {"error": f"Search request failed: {exc}"}
    except ImportError as exc:
      return {"error": str(exc)}

  # -- Tool implementations -----------------------------------------------

  async def search_ards(
      self,
      tool_context: ToolContext,
      query: str,
      artifact_type: str | None = None,
      limit: int = 10,
  ) -> dict[str, Any]:
    """Search ARD registries across all artifact types.

    Args:
      query: Natural-language search query describing what you need,
          e.g. "remove image background" or "code review".
      artifact_type: Optional filter by artifact kind.  Supported
          values: ``skill``, ``mcp``, ``space``, ``a2a``, or a raw
          media type like ``application/mcp-server-card+json``.  When
          omitted, all artifact types are returned.
      limit: Maximum number of results to return (1-100, default 10).

    Returns:
      A dictionary with ``results`` (list of matching entries) and
      optionally ``referrals`` (list of additional registries).
    """
    return await self._do_search(
        query, artifact_type=artifact_type, limit=limit
    )

  async def search_agents(
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

  async def search_skills(
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

  async def search_tools(
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

  async def search_spaces(
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

  async def get_agent_card(
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
    parsed = urlparse(url)
    if parsed.scheme not in self._allowed_schemes:
      return {"error": f"URL scheme '{parsed.scheme}' not allowed: {url}"}

    try:
      raw = await asyncio.to_thread(
          _remote_fetch, url, token=self._token
      )
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

  async def connect_agent(
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

    Note: The remote agent may return an immediate response or start
    a long-running task (submitted → working → completed).  This
    call collects available response text, but the exchange is the
    beginning of an A2A conversation — for multi-turn interactions,
    use ``RemoteA2aAgent`` directly with the discovered agent card URL.

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
      import httpx
      from a2a.client.card_resolver import (
          A2ACardResolver,
      )
      from a2a.client.client import (
          ClientConfig as A2AClientConfig,
      )
      from a2a.client.client_factory import (
          ClientFactory as A2AClientFactory,
      )
      from a2a.types import (
          Message as A2AMessage,
          Part as A2APart,
          Role as A2ARole,
          SendMessageConfiguration,
          SendMessageRequest,
      )
    except ImportError:
      return {
          "error": (
              "A2A dependencies are not installed.  Install them"
              " with:  pip install 'google-adk[a2a]'"
          )
      }

    try:
      parsed = urlparse(agent_card_url)
      if parsed.scheme not in self._allowed_schemes or not parsed.netloc:
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
        agent_name = getattr(agent_card, "name", "unknown")

        # Try the SDK-based A2A client first.
        try:
          factory = A2AClientFactory(
              config=A2AClientConfig(httpx_client=http_client),
          )
          a2a_client = factory.create(agent_card)

          # a2a-sdk >=1.x uses protobuf: Part(text=...) and
          # SendMessageRequest wrapping a Message.
          a2a_msg = A2AMessage(
              message_id=str(uuid.uuid4()),
              parts=[A2APart(text=message)],
              role=A2ARole.ROLE_USER,
          )
          request = SendMessageRequest(
              message=a2a_msg,
              configuration=SendMessageConfiguration(),
          )

          response_texts: list[str] = []
          async for a2a_response in a2a_client.send_message(
              request=request,
          ):
            response_texts.extend(
                _extract_text_from_a2a_response(a2a_response)
            )

          response_text = (
              "\n".join(response_texts) if response_texts else ""
          )
          return {
              "response": response_text,
              "agent_name": agent_name,
              "agent_url": agent_card_url,
          }

        except Exception as sdk_exc:
          # SDK call failed (e.g. interface URL differs from card
          # URL and has SSL / connectivity issues).  Fall back to
          # raw JSON-RPC POST at the card URL's base.
          logger.info(
              "SDK A2A call failed (%s), trying JSON-RPC "
              "fallback to %s",
              sdk_exc,
              base_url,
          )
          return await self._connect_agent_jsonrpc_fallback(
              http_client=http_client,
              base_url=base_url,
              agent_name=agent_name,
              agent_card_url=agent_card_url,
              message=message,
          )

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

  async def _connect_agent_jsonrpc_fallback(
      self,
      *,
      http_client: Any,
      base_url: str,
      agent_name: str,
      agent_card_url: str,
      message: str,
  ) -> dict[str, Any]:
    """Send an A2A message via raw JSON-RPC POST.

    This fallback is used when the SDK-based client fails (e.g. due to
    the agent card's interface URL differing from the card fetch URL
    with SSL or connectivity issues).  We POST a ``message/send``
    JSON-RPC request directly to the card URL's base.
    """
    jsonrpc_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "user",
                "parts": [{"kind": "text", "text": message}],
            },
        },
    }
    try:
      resp = await http_client.post(
          f"{base_url}/",
          json=jsonrpc_payload,
          headers={"Content-Type": "application/json"},
      )
      resp.raise_for_status()
      data = resp.json()

      # Extract text from the JSON-RPC result.
      result = data.get("result", {})
      texts: list[str] = []
      for artifact in result.get("artifacts", []):
        for part in artifact.get("parts", []):
          if part.get("kind") == "text" and "text" in part:
            texts.append(part["text"])

      # Also check status message.
      status = result.get("status", {})
      status_msg = status.get("message", {})
      if isinstance(status_msg, dict):
        for part in status_msg.get("parts", []):
          if part.get("kind") == "text" and "text" in part:
            texts.append(part["text"])

      return {
          "response": "\n".join(texts) if texts else "",
          "agent_name": agent_name,
          "agent_url": agent_card_url,
          "method": "jsonrpc_fallback",
      }
    except Exception as exc:
      return {
          "error": (
              f"Failed to communicate with agent at"
              f" {agent_card_url}: {exc}"
          ),
      }

  # -- BaseToolset interface ----------------------------------------------

  async def get_tools(
      self,
      readonly_context: ReadonlyContext | None = None,
  ) -> list[BaseTool]:
    """Return all search, inspect, and connect tools."""
    tool_funcs = [
        self.search_ards,
        self.search_agents,
        self.search_skills,
        self.search_tools,
        self.search_spaces,
        self.get_agent_card,
        self.connect_agent,
    ]
    tools: list[BaseTool] = [FunctionTool(func) for func in tool_funcs]

    if readonly_context is not None:
      tools = [
          tool
          for tool in tools
          if self._is_tool_selected(tool, readonly_context)
      ]

    return tools
