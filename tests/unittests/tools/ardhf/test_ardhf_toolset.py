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
require the hf-discover challenge server (deterministic fixtures,
no API keys needed)::

    pip install hf-discover
    hf-discover challenge serve --port 8090
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
        == "application/mcp-server-card+json"
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
  async def test_returns_all_tools(self):
    """All seven tools are exposed by default."""
    toolset = AgentFinderToolset()

    tools = await toolset.get_tools()

    assert len(tools) == 7
    names = {tool.name for tool in tools}
    assert names == {
        "search_ards",
        "search_agents",
        "search_skills",
        "search_tools",
        "search_spaces",
        "get_agent_card",
        "connect_agent",
    }

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
    assert "ard_search_ards" in names
    assert "ard_search_agents" in names
    assert "ard_search_skills" in names
    assert "ard_search_tools" in names
    assert "ard_search_spaces" in names
    assert "ard_get_agent_card" in names
    assert "ard_connect_agent" in names

  @pytest.mark.asyncio
  async def test_search_ards_handles_connection_error(self):
    """search_ards returns an error dict for unreachable servers."""
    toolset = AgentFinderToolset(
        registry_url="http://127.0.0.1:19999"
    )
    tools = await toolset.get_tools()
    search_tool = next(t for t in tools if t.name == "search_ards")

    mock_context = AsyncMock()
    result = await search_tool.run_async(
        args={"query": "test", "limit": 5},
        tool_context=mock_context,
    )

    assert "error" in result

  @pytest.mark.asyncio
  async def test_search_agents_delegates_with_a2a_type(self):
    """search_agents passes artifact_type='a2a' to the search logic."""
    toolset = AgentFinderToolset()

    with patch.object(
        toolset, "_do_search", new_callable=AsyncMock
    ) as mock_search:
      mock_search.return_value = {"results": []}
      mock_context = AsyncMock()
      await toolset._search_agents(
          mock_context, query="test", limit=5
      )

    mock_search.assert_called_once_with(
        "test", artifact_type="a2a", limit=5
    )

  @pytest.mark.asyncio
  async def test_search_skills_delegates_with_skill_type(self):
    """search_skills passes artifact_type='skill' to the search logic."""
    toolset = AgentFinderToolset()

    with patch.object(
        toolset, "_do_search", new_callable=AsyncMock
    ) as mock_search:
      mock_search.return_value = {"results": []}
      mock_context = AsyncMock()
      await toolset._search_skills(
          mock_context, query="test", limit=5
      )

    mock_search.assert_called_once_with(
        "test", artifact_type="skill", limit=5
    )

  @pytest.mark.asyncio
  async def test_search_tools_delegates_with_mcp_type(self):
    """search_tools passes artifact_type='mcp' to the search logic."""
    toolset = AgentFinderToolset()

    with patch.object(
        toolset, "_do_search", new_callable=AsyncMock
    ) as mock_search:
      mock_search.return_value = {"results": []}
      mock_context = AsyncMock()
      await toolset._search_tools(
          mock_context, query="test", limit=5
      )

    mock_search.assert_called_once_with(
        "test", artifact_type="mcp", limit=5
    )

  @pytest.mark.asyncio
  async def test_search_spaces_delegates_with_space_type(self):
    """search_spaces passes artifact_type='space' to the search logic."""
    toolset = AgentFinderToolset()

    with patch.object(
        toolset, "_do_search", new_callable=AsyncMock
    ) as mock_search:
      mock_search.return_value = {"results": []}
      mock_context = AsyncMock()
      await toolset._search_spaces(
          mock_context, query="test", limit=5
      )

    mock_search.assert_called_once_with(
        "test", artifact_type="space", limit=5
    )


class TestDoSearch:
  """Tests for the _do_search core logic."""

  @pytest.mark.asyncio
  async def test_limit_clamped_to_valid_range(self):
    """Limit values outside 1-100 are clamped."""
    toolset = AgentFinderToolset()

    with patch(
        "google.adk_community.tools.ardhf.ardhf_toolset"
        "._remote_search",
        return_value={"results": []},
    ) as mock_search:
      await toolset._do_search("test", limit=0)
      _, kwargs = mock_search.call_args
      assert kwargs["limit"] == 1

      await toolset._do_search("test", limit=200)
      _, kwargs = mock_search.call_args
      assert kwargs["limit"] == 100

  @pytest.mark.asyncio
  async def test_raw_media_type_passed_through(self):
    """A raw media type string is used as-is when kind lookup fails."""
    toolset = AgentFinderToolset()

    with patch(
        "google.adk_community.tools.ardhf.ardhf_toolset"
        "._remote_search",
        return_value={"results": []},
    ) as mock_search:
      await toolset._do_search(
          "test",
          artifact_type="application/custom+json",
      )
      _, kwargs = mock_search.call_args
      assert (
          kwargs["artifact_type"] == "application/custom+json"
      )

  @pytest.mark.asyncio
  async def test_get_agent_card_returns_error_for_unreachable(self):
    """get_agent_card returns error dict for unreachable URLs."""
    toolset = AgentFinderToolset()
    tools = await toolset.get_tools()
    fetch_tool = next(
        t for t in tools if t.name == "get_agent_card"
    )

    mock_context = AsyncMock()
    result = await fetch_tool.run_async(
        args={"url": "http://127.0.0.1:19999/nonexistent"},
        tool_context=mock_context,
    )

    assert "error" in result

  @pytest.mark.asyncio
  async def test_get_agent_card_rejects_file_url(self):
    """get_agent_card rejects file:// URLs to prevent SSRF."""
    toolset = AgentFinderToolset()
    mock_context = AsyncMock()

    result = await toolset._get_agent_card(
        mock_context, url="file:///etc/passwd"
    )

    assert "error" in result
    assert "not allowed" in result["error"]

  @pytest.mark.asyncio
  async def test_connect_agent_rejects_file_url(self):
    """connect_agent rejects non-HTTP URLs."""
    toolset = AgentFinderToolset()
    mock_context = AsyncMock()

    result = await toolset._connect_agent(
        mock_context,
        agent_card_url="ftp://example.com/agent.json",
        message="hello",
    )

    assert "error" in result


class TestGetAgentCard:
  """Tests for the get_agent_card tool's content handling."""

  @pytest.mark.asyncio
  async def test_returns_parsed_json_for_json_content(self):
    """JSON content is parsed and returned as a dict."""
    toolset = AgentFinderToolset()
    json_content = '{"name": "test-tool", "version": "1.0"}'

    with patch(
        "google.adk_community.tools.ardhf.ardhf_toolset"
        "._remote_fetch",
        return_value=json_content,
    ):
      mock_context = AsyncMock()
      result = await toolset._get_agent_card(
          mock_context, url="https://example.com/tool.json"
      )

    assert result["name"] == "test-tool"
    assert result["version"] == "1.0"

  @pytest.mark.asyncio
  async def test_returns_markdown_for_non_json_content(self):
    """Non-JSON content is returned as raw text under 'content'."""
    toolset = AgentFinderToolset()
    md_content = "# Skill\n\nThis is a skill."

    with patch(
        "google.adk_community.tools.ardhf.ardhf_toolset"
        "._remote_fetch",
        return_value=md_content,
    ):
      mock_context = AsyncMock()
      result = await toolset._get_agent_card(
          mock_context,
          url="https://example.com/SKILL.md",
      )

    assert result["content"] == md_content
    assert result["content_type"] == "text/markdown"

  @pytest.mark.asyncio
  async def test_local_mode_delegates_to_local_search(self):
    """Local mode calls _local_search instead of _remote_search."""
    toolset = AgentFinderToolset(local=True)

    with patch(
        "google.adk_community.tools.ardhf.ardhf_toolset"
        "._local_search",
        return_value={"results": []},
    ) as mock_local:
      await toolset._do_search("test query", limit=5)

    mock_local.assert_called_once_with(
        "test query",
        artifact_type=None,
        limit=5,
        token=None,
    )

  @pytest.mark.asyncio
  async def test_local_mode_import_error_returns_error(self):
    """Local mode returns error when hf-discover is not installed."""
    toolset = AgentFinderToolset(local=True)

    with patch(
        "google.adk_community.tools.ardhf.ardhf_toolset"
        "._local_search",
        side_effect=ImportError("hf-discover not installed"),
    ):
      result = await toolset._do_search("test query")

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
        ),
        patch(
            "a2a.client.client_factory.ClientFactory",
            return_value=mock_factory,
        ),
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
        artifact_type="application/mcp-server-card+json",
        limit=10,
    )

    for result in response["results"]:
      assert result["type"] == "application/mcp-server-card+json"

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
  async def test_search_ards_tool(self):
    """search_ards returns results from the challenge server."""
    toolset = AgentFinderToolset(registry_url=CHALLENGE_URL)
    tools = await toolset.get_tools()
    search_tool = next(
        t for t in tools if t.name == "search_ards"
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
  async def test_search_ards_with_kind_resolution(self):
    """search_ards resolves human-friendly kind names."""
    toolset = AgentFinderToolset(registry_url=CHALLENGE_URL)
    tools = await toolset.get_tools()
    search_tool = next(
        t for t in tools if t.name == "search_ards"
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
      assert entry["type"] == "application/mcp-server-card+json"
