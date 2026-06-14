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

"""Tests for the ARDHF toolset.

Unit tests run without any external services.  Integration tests
require the hf-agentfinder challenge server (deterministic fixtures,
no API keys needed)::

    pip install hf-agentfinder
    hf-agentfinder challenge serve --port 8090
    pytest tests/unittests/tools/ardhf/ -v
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from google.adk_community.tools.ardhf.ardhf_toolset import (
    AgentFinderToolset,
    _artifact_type_for_kind,
    _extract_text_from_a2a_response,
    _registry_search_url,
    _remote_fetch,
    _remote_search,
)

# ------------------------------------------------------------------ #
# Configuration                                                       #
# ------------------------------------------------------------------ #

CHALLENGE_URL = os.environ.get(
    "ARDHF_TEST_REGISTRY_URL", "http://127.0.0.1:8090"
)


# ------------------------------------------------------------------ #
# Unit tests (no server needed)                                        #
# ------------------------------------------------------------------ #


class TestRegistrySearchUrl:
  """Tests for URL normalisation helper."""

  def test_appends_search_to_base_url(self):
    """A bare base URL gets /search appended."""
    assert (
        _registry_search_url("http://localhost:8090")
        == "http://localhost:8090/search"
    )

  def test_preserves_existing_search_path(self):
    """A URL already ending in /search is returned unchanged."""
    url = "http://localhost:8090/registries/tools/search"
    assert _registry_search_url(url) == url

  def test_strips_trailing_slash(self):
    """Trailing slashes are normalised before appending."""
    assert (
        _registry_search_url("http://localhost:8090/")
        == "http://localhost:8090/search"
    )


class TestArtifactTypeForKind:
  """Tests for kind-to-media-type mapping."""

  def test_skill_kind(self):
    assert _artifact_type_for_kind("skill") == "application/ai-skill"

  def test_mcp_kind(self):
    assert (
        _artifact_type_for_kind("mcp")
        == "application/mcp-server+json"
    )

  def test_space_kind(self):
    assert (
        _artifact_type_for_kind("space")
        == "application/vnd.huggingface.space+json"
    )

  def test_a2a_kind(self):
    assert (
        _artifact_type_for_kind("a2a")
        == "application/a2a-agent-card+json"
    )

  def test_unknown_kind_returns_none(self):
    assert _artifact_type_for_kind("unknown") is None


class TestToolsetGetTools:
  """Tests for AgentFinderToolset.get_tools without a running server."""

  @pytest.mark.asyncio
  async def test_returns_three_tools(self):
    """The toolset exposes search_agents, get_agent_card, and connect_agent."""
    toolset = AgentFinderToolset()

    tools = await toolset.get_tools()

    assert len(tools) == 3
    names = {tool.name for tool in tools}
    assert names == {"search_agents", "get_agent_card", "connect_agent"}

  @pytest.mark.asyncio
  async def test_tools_have_descriptions(self):
    """Each tool has a non-empty description."""
    toolset = AgentFinderToolset()
    tools = await toolset.get_tools()

    for tool in tools:
      assert tool.description, f"Tool {tool.name} has no description"

  @pytest.mark.asyncio
  async def test_tool_name_prefix(self):
    """tool_name_prefix is applied to all tool names."""
    toolset = AgentFinderToolset(tool_name_prefix="ard")

    tools = await toolset.get_tools_with_prefix()

    names = {tool.name for tool in tools}
    assert "ard_search_agents" in names
    assert "ard_get_agent_card" in names
    assert "ard_connect_agent" in names

  @pytest.mark.asyncio
  async def test_search_handles_connection_error(self):
    """search_agents returns an error dict for unreachable servers."""
    toolset = AgentFinderToolset(
        registry_url="http://127.0.0.1:19999"
    )
    tools = await toolset.get_tools()
    search_tool = next(t for t in tools if t.name == "search_agents")

    mock_context = AsyncMock()
    result = await search_tool.run_async(
        args={"query": "test", "limit": 5},
        tool_context=mock_context,
    )

    assert "error" in result


class TestExtractTextFromA2aResponse:
  """Tests for the _extract_text_from_a2a_response helper."""

  def test_extracts_text_from_message(self):
    """Text is extracted from an A2AMessage with TextPart."""
    try:
      from a2a.types import Message as A2AMessage
      from a2a.types import Part as A2APart
      from a2a.types import TextPart as A2ATextPart
    except ImportError:
      pytest.skip("a2a SDK not installed")

    msg = A2AMessage(
        message_id="test-1",
        parts=[A2APart(root=A2ATextPart(text="Hello from agent"))],
        role="agent",
    )

    texts = _extract_text_from_a2a_response(msg)

    assert texts == ["Hello from agent"]

  def test_extracts_text_from_task_tuple(self):
    """Text is extracted from a (Task, None) tuple response."""
    try:
      from a2a.types import Artifact
      from a2a.types import Part as A2APart
      from a2a.types import Task as A2ATask
      from a2a.types import TaskState
      from a2a.types import TaskStatus
      from a2a.types import TextPart as A2ATextPart
    except ImportError:
      pytest.skip("a2a SDK not installed")

    task = A2ATask(
        id="task-1",
        contextId="ctx-1",
        status=TaskStatus(state=TaskState.completed),
        artifacts=[
            Artifact(
                artifactId="art-1",
                parts=[
                    A2APart(root=A2ATextPart(text="Task result"))
                ],
            )
        ],
    )

    texts = _extract_text_from_a2a_response((task, None))

    assert "Task result" in texts

  def test_returns_empty_for_unknown_type(self):
    """An unknown response type returns an empty list."""
    texts = _extract_text_from_a2a_response("unexpected")
    assert texts == []

  def test_returns_empty_when_a2a_not_installed(self):
    """Returns empty list when a2a SDK is not available."""
    with patch.dict(
        "sys.modules", {"a2a": None, "a2a.types": None}
    ):
      # Re-import to pick up the patched modules.
      texts = _extract_text_from_a2a_response("anything")
      assert texts == []


class TestConnectAgent:
  """Tests for the connect_agent tool."""

  @pytest.mark.asyncio
  async def test_connect_agent_tool_exists(self):
    """The toolset exposes a connect_agent tool."""
    toolset = AgentFinderToolset()
    tools = await toolset.get_tools()

    names = {tool.name for tool in tools}
    assert "connect_agent" in names

  @pytest.mark.asyncio
  async def test_connect_agent_has_description(self):
    """The connect_agent tool has a non-empty description."""
    toolset = AgentFinderToolset()
    tools = await toolset.get_tools()
    connect_tool = next(
        t for t in tools if t.name == "connect_agent"
    )

    assert connect_tool.description

  @pytest.mark.asyncio
  async def test_connect_agent_invalid_url(self):
    """connect_agent returns error for invalid URLs."""
    toolset = AgentFinderToolset()
    tools = await toolset.get_tools()
    connect_tool = next(
        t for t in tools if t.name == "connect_agent"
    )

    mock_context = AsyncMock()
    result = await connect_tool.run_async(
        args={
            "agent_card_url": "not-a-valid-url",
            "message": "hello",
        },
        tool_context=mock_context,
    )

    assert "error" in result

  @pytest.mark.asyncio
  async def test_connect_agent_unreachable_host(self):
    """connect_agent returns error for unreachable agents."""
    toolset = AgentFinderToolset()
    tools = await toolset.get_tools()
    connect_tool = next(
        t for t in tools if t.name == "connect_agent"
    )

    mock_context = AsyncMock()
    result = await connect_tool.run_async(
        args={
            "agent_card_url": (
                "http://127.0.0.1:19999/.well-known/agent.json"
            ),
            "message": "hello",
        },
        tool_context=mock_context,
    )

    assert "error" in result

  @pytest.mark.asyncio
  async def test_connect_agent_success_with_mocked_a2a(self):
    """connect_agent returns response text from a mocked A2A agent."""
    try:
      from a2a.types import AgentCard
      from a2a.types import Message as A2AMessage
      from a2a.types import Part as A2APart
      from a2a.types import TextPart as A2ATextPart
    except ImportError:
      pytest.skip("a2a SDK not installed")

    mock_agent_card = MagicMock(spec=AgentCard)
    mock_agent_card.name = "test-agent"
    mock_agent_card.url = "http://localhost:9999/a2a"

    mock_response = A2AMessage(
        message_id="resp-1",
        parts=[A2APart(root=A2ATextPart(text="I can help!"))],
        role="agent",
    )

    async def mock_send_message(**kwargs):
      yield mock_response

    mock_client = MagicMock()
    mock_client.send_message = mock_send_message

    mock_factory = MagicMock()
    mock_factory.create.return_value = mock_client

    mock_resolver = AsyncMock()
    mock_resolver.get_agent_card.return_value = mock_agent_card

    toolset = AgentFinderToolset()
    tools = await toolset.get_tools()
    connect_tool = next(
        t for t in tools if t.name == "connect_agent"
    )

    mock_context = AsyncMock()

    with (
        patch(
            "a2a.client.card_resolver.A2ACardResolver",
            return_value=mock_resolver,
        ) as _,
        patch(
            "a2a.client.client_factory.ClientFactory",
            return_value=mock_factory,
        ) as _,
    ):
      result = await connect_tool.run_async(
          args={
              "agent_card_url": (
                  "http://localhost:9999/.well-known/agent.json"
              ),
              "message": "Can you help?",
          },
          tool_context=mock_context,
      )

    assert result["response"] == "I can help!"
    assert result["agent_name"] == "test-agent"


# ------------------------------------------------------------------ #
# Integration tests (require challenge server)                         #
# ------------------------------------------------------------------ #


def _challenge_server_available() -> bool:
  """Check if the challenge server is reachable."""
  try:
    from urllib.request import urlopen

    with urlopen(  # noqa: S310
        f"{CHALLENGE_URL}/health", timeout=2
    ) as resp:
      data = json.loads(resp.read())
      return data.get("status") == "ok"
  except Exception:
    return False


challenge_server = pytest.mark.skipif(
    not _challenge_server_available(),
    reason=f"Challenge server not available at {CHALLENGE_URL}",
)


@challenge_server
class TestRemoteSearchAgainstChallenge:
  """Integration tests against the challenge server fixtures."""

  def test_search_returns_results(self):
    """A search query returns a non-empty results list."""
    response = _remote_search(
        CHALLENGE_URL, "find tools", limit=5
    )

    assert "results" in response
    assert len(response["results"]) > 0

  def test_search_results_have_required_fields(self):
    """Each result contains ARD-required fields."""
    response = _remote_search(
        CHALLENGE_URL, "find tools", limit=5
    )

    for result in response["results"]:
      assert "identifier" in result
      assert "displayName" in result
      assert "type" in result
      assert "score" in result

  def test_search_with_mcp_filter(self):
    """Filtering by MCP type returns only MCP server results."""
    response = _remote_search(
        CHALLENGE_URL,
        "find tools",
        artifact_type="application/mcp-server+json",
        limit=10,
    )

    for result in response["results"]:
      assert result["type"] == "application/mcp-server+json"

  def test_search_with_skill_filter(self):
    """Filtering by skill type returns only skill results."""
    response = _remote_search(
        CHALLENGE_URL,
        "find tools",
        artifact_type="application/ai-skill",
        limit=10,
    )

    for result in response["results"]:
      assert result["type"] == "application/ai-skill"

  def test_search_returns_referrals(self):
    """The challenge server returns referrals to sub-registries."""
    response = _remote_search(
        CHALLENGE_URL, "find tools", limit=5
    )

    assert "referrals" in response
    assert len(response["referrals"]) > 0

  def test_search_respects_limit(self):
    """The pageSize parameter limits the number of results."""
    response = _remote_search(
        CHALLENGE_URL, "find tools", limit=2
    )

    assert len(response["results"]) <= 2

  def test_sub_registry_search(self):
    """Searching a sub-registry returns its specific results."""
    response = _remote_search(
        f"{CHALLENGE_URL}/registries/tools",
        "find tools",
        limit=10,
    )

    assert "results" in response
    assert len(response["results"]) > 0

  def test_empty_registry_returns_no_results(self):
    """The empty sub-registry returns an empty results list."""
    response = _remote_search(
        f"{CHALLENGE_URL}/registries/empty",
        "anything",
        limit=10,
    )

    assert response["results"] == []


@challenge_server
class TestRemoteFetchAgainstChallenge:
  """Integration tests for fetching artifacts from challenge server."""

  def test_fetch_skill_artifact(self):
    """Fetching a skill URL returns markdown content."""
    content = _remote_fetch(
        f"{CHALLENGE_URL}/artifacts/skills/triage-skill/SKILL.md"
    )

    assert "triage-skill" in content
    assert "Challenge fixture skill" in content

  def test_fetch_mcp_artifact(self):
    """Fetching an MCP artifact URL returns a JSON descriptor."""
    content = _remote_fetch(
        f"{CHALLENGE_URL}/artifacts/mcp/echo-tools"
    )

    data = json.loads(content)
    assert data["name"] == "echo-tools"
    assert "tools" in data


@challenge_server
class TestToolsetAgainstChallenge:
  """Integration tests for AgentFinderToolset against challenge."""

  @pytest.mark.asyncio
  async def test_search_agents_tool(self):
    """search_agents returns results from the challenge server."""
    toolset = AgentFinderToolset(registry_url=CHALLENGE_URL)
    tools = await toolset.get_tools()
    search_tool = next(
        t for t in tools if t.name == "search_agents"
    )

    mock_context = AsyncMock()
    result = await search_tool.run_async(
        args={"query": "find tools", "limit": 5},
        tool_context=mock_context,
    )

    assert "results" in result
    assert len(result["results"]) > 0

  @pytest.mark.asyncio
  async def test_get_agent_card_json(self):
    """get_agent_card returns parsed JSON for MCP artifacts."""
    toolset = AgentFinderToolset(registry_url=CHALLENGE_URL)
    tools = await toolset.get_tools()
    fetch_tool = next(
        t for t in tools if t.name == "get_agent_card"
    )

    mock_context = AsyncMock()
    result = await fetch_tool.run_async(
        args={
            "url": f"{CHALLENGE_URL}/artifacts/mcp/echo-tools"
        },
        tool_context=mock_context,
    )

    assert result["name"] == "echo-tools"

  @pytest.mark.asyncio
  async def test_get_agent_card_markdown(self):
    """get_agent_card returns markdown content for skill artifacts."""
    toolset = AgentFinderToolset(registry_url=CHALLENGE_URL)
    tools = await toolset.get_tools()
    fetch_tool = next(
        t for t in tools if t.name == "get_agent_card"
    )

    mock_context = AsyncMock()
    result = await fetch_tool.run_async(
        args={
            "url": (
                f"{CHALLENGE_URL}/artifacts/skills/"
                "triage-skill/SKILL.md"
            )
        },
        tool_context=mock_context,
    )

    assert "content" in result
    assert "triage-skill" in result["content"]

  @pytest.mark.asyncio
  async def test_search_with_kind_resolution(self):
    """search_agents resolves human-friendly kind names."""
    toolset = AgentFinderToolset(registry_url=CHALLENGE_URL)
    tools = await toolset.get_tools()
    search_tool = next(
        t for t in tools if t.name == "search_agents"
    )

    mock_context = AsyncMock()
    result = await search_tool.run_async(
        args={
            "query": "find tools",
            "artifact_type": "mcp",
            "limit": 10,
        },
        tool_context=mock_context,
    )

    assert "results" in result
    for entry in result["results"]:
      assert entry["type"] == "application/mcp-server+json"
