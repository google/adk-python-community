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
from unittest.mock import AsyncMock

import pytest

from google.adk_community.tools.ardhf.ardhf_toolset import (
    AgentFinderToolset,
    _artifact_type_for_kind,
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
  async def test_returns_two_tools(self):
    """The toolset exposes search_agents and get_agent_card."""
    toolset = AgentFinderToolset()

    tools = await toolset.get_tools()

    assert len(tools) == 2
    names = {tool.name for tool in tools}
    assert names == {"search_agents", "get_agent_card"}

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
