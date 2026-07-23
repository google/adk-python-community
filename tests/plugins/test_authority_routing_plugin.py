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

"""Tests for AuthorityRoutingPlugin.

These tests are fully deterministic — no model calls. The optional posture
router is exercised with plain Python callables, so the suite runs with no
credentials and no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from google.adk_community.plugins.authority_routing_plugin import AuthorityRequest
from google.adk_community.plugins.authority_routing_plugin import AuthorityRoutingPlugin
from google.adk_community.plugins.authority_routing_plugin import Posture
from google.adk_community.plugins.authority_routing_plugin import PostureVerdict
from google.adk_community.plugins.authority_routing_plugin import stricter

# ---------------------------------------------------------------------------
# Fake ADK types for testing
# ---------------------------------------------------------------------------


class _FakeBaseTool:

  def __init__(self, name: str):
    self.name = name


class _FakeToolContext:

  def __init__(self, state: dict[str, Any] | None = None):
    self.state = state or {}


def _make_router(*, posture: Posture, scope: str = "", rationale: str = ""):
  def router(_req: AuthorityRequest) -> PostureVerdict:
    return PostureVerdict(posture=posture, scope=scope, rationale=rationale)

  return router


async def _decide(
    plugin: AuthorityRoutingPlugin, tool_name, args=None, state=None
):
  return await plugin.before_tool_callback(
      tool=_FakeBaseTool(tool_name),
      tool_args=args or {},
      tool_context=_FakeToolContext(state),
  )


# ---------------------------------------------------------------------------
# Posture ladder
# ---------------------------------------------------------------------------


class TestStrictness:

  def test_stricter_picks_higher_rung(self):
    assert stricter(Posture.EXECUTE, Posture.DEFER) is Posture.DEFER
    assert stricter(Posture.STOP, Posture.EXECUTE) is Posture.STOP
    assert stricter(Posture.ADVISE, Posture.DEFER) is Posture.DEFER

  def test_stricter_is_symmetric(self):
    assert stricter(Posture.DEFER, Posture.ADVISE) is stricter(
        Posture.ADVISE, Posture.DEFER
    )

  def test_full_ladder_order(self):
    ladder = [Posture.EXECUTE, Posture.ADVISE, Posture.DEFER, Posture.STOP]
    for i in range(len(ladder)):
      for j in range(len(ladder)):
        expected = ladder[max(i, j)]
        assert stricter(ladder[i], ladder[j]) is expected


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:

  def test_default_name(self):
    assert AuthorityRoutingPlugin().name == "authority_routing"

  def test_custom_name(self):
    assert AuthorityRoutingPlugin(name="authz").name == "authz"

  def test_no_dependencies_required(self):
    # Constructs with zero configuration and no optional deps installed.
    plugin = AuthorityRoutingPlugin()
    assert plugin is not None


# ---------------------------------------------------------------------------
# Guard-only mode (no router)
# ---------------------------------------------------------------------------


class TestGuardOnly:

  @pytest.mark.asyncio
  async def test_clean_call_executes(self):
    plugin = AuthorityRoutingPlugin()
    result = await _decide(plugin, "web_search", {"query": "weather"})
    assert result is None

  @pytest.mark.asyncio
  async def test_irreversible_tool_name_defers(self):
    plugin = AuthorityRoutingPlugin()
    result = await _decide(plugin, "delete_record", {"id": 7})
    assert result is not None
    assert result["error"] == "authority_blocked"
    assert result["posture"] == "DEFER"
    assert "tripwire" in result["reason"]

  @pytest.mark.asyncio
  async def test_irreversible_arg_value_defers(self):
    plugin = AuthorityRoutingPlugin()
    # Benign tool name, but the argument reveals a destructive command.
    result = await _decide(plugin, "run_shell", {"cmd": "rm -rf build"})
    assert result is not None
    assert result["posture"] == "DEFER"
    assert "rm -" in result["reason"]

  @pytest.mark.asyncio
  async def test_approval_marker_in_context_defers(self):
    plugin = AuthorityRoutingPlugin()
    result = await _decide(
        plugin,
        "write_file",
        {"path": "notes.txt"},
        state={"authority_context": "This task is pending approval from ops."},
    )
    assert result is not None
    assert result["posture"] == "DEFER"
    assert "approval-state marker" in result["reason"]

  @pytest.mark.asyncio
  async def test_approval_marker_takes_precedence_over_tripwire(self):
    # Both fire; the approval marker is checked first, but either way DEFER.
    plugin = AuthorityRoutingPlugin()
    result = await _decide(
        plugin,
        "delete_record",
        {"id": 1},
        state={"authority_context": "hold off — awaiting approval"},
    )
    assert result["posture"] == "DEFER"
    assert "approval-state marker" in result["reason"]


# ---------------------------------------------------------------------------
# Router mode
# ---------------------------------------------------------------------------


class TestRouter:

  @pytest.mark.asyncio
  async def test_router_execute_allows(self):
    plugin = AuthorityRoutingPlugin(
        posture_router=_make_router(posture=Posture.EXECUTE, scope="one row")
    )
    result = await _decide(plugin, "update_row", {"id": 2})
    assert result is None

  @pytest.mark.asyncio
  async def test_router_advise_blocks(self):
    plugin = AuthorityRoutingPlugin(
        posture_router=_make_router(
            posture=Posture.ADVISE, rationale="user asked for a review"
        )
    )
    result = await _decide(plugin, "update_row", {"id": 2})
    assert result is not None
    assert result["posture"] == "ADVISE"
    assert result["reason"] == "user asked for a review"

  @pytest.mark.asyncio
  async def test_router_stop_blocks(self):
    plugin = AuthorityRoutingPlugin(
        posture_router=_make_router(posture=Posture.STOP)
    )
    result = await _decide(plugin, "read_public_doc", {})
    assert result is not None
    assert result["posture"] == "STOP"

  @pytest.mark.asyncio
  async def test_router_scope_surfaced_when_no_rationale(self):
    plugin = AuthorityRoutingPlugin(
        posture_router=_make_router(
            posture=Posture.DEFER, scope="scope unclear"
        )
    )
    result = await _decide(plugin, "read_doc", {})
    assert result["scope"] == "scope unclear"


# ---------------------------------------------------------------------------
# Most-restrictive-wins composition
# ---------------------------------------------------------------------------


class TestComposition:

  @pytest.mark.asyncio
  async def test_guard_overrides_router_execute(self):
    # Router says EXECUTE, but the tool call is irreversible -> DEFER.
    plugin = AuthorityRoutingPlugin(
        posture_router=_make_router(posture=Posture.EXECUTE, scope="one file")
    )
    result = await _decide(plugin, "delete_file", {"path": "prod.db"})
    assert result is not None
    assert result["posture"] == "DEFER"
    assert "deterministic guard" in result["reason"]

  @pytest.mark.asyncio
  async def test_router_stop_survives_clean_guard(self):
    # Router says STOP; guard finds nothing. The guard never loosens -> STOP.
    plugin = AuthorityRoutingPlugin(
        posture_router=_make_router(posture=Posture.STOP)
    )
    result = await _decide(plugin, "web_search", {"query": "hi"})
    assert result["posture"] == "STOP"

  @pytest.mark.asyncio
  async def test_router_advise_survives_clean_guard(self):
    plugin = AuthorityRoutingPlugin(
        posture_router=_make_router(posture=Posture.ADVISE)
    )
    result = await _decide(plugin, "web_search", {"query": "hi"})
    assert result["posture"] == "ADVISE"


# ---------------------------------------------------------------------------
# Fail-closed behavior
# ---------------------------------------------------------------------------


class TestFailClosed:

  @pytest.mark.asyncio
  async def test_router_raises_defaults_to_defer(self):
    def boom(_req):
      raise RuntimeError("model timeout")

    plugin = AuthorityRoutingPlugin(posture_router=boom)
    result = await _decide(plugin, "web_search", {})
    assert result is not None
    assert result["posture"] == "DEFER"

  @pytest.mark.asyncio
  async def test_router_unparseable_defaults_to_defer(self):
    plugin = AuthorityRoutingPlugin(posture_router=lambda _req: 42)
    result = await _decide(plugin, "web_search", {})
    assert result["posture"] == "DEFER"

  @pytest.mark.asyncio
  async def test_router_bad_string_defaults_to_defer(self):
    plugin = AuthorityRoutingPlugin(posture_router=lambda _req: "MAYBE")
    result = await _decide(plugin, "web_search", {})
    assert result["posture"] == "DEFER"

  @pytest.mark.asyncio
  async def test_fail_open_ignores_bad_router(self):
    def boom(_req):
      raise RuntimeError("model timeout")

    plugin = AuthorityRoutingPlugin(posture_router=boom, fail_closed=False)
    # Router ignored -> EXECUTE, and a clean guard -> allow.
    result = await _decide(plugin, "web_search", {})
    assert result is None

  @pytest.mark.asyncio
  async def test_fail_open_still_respects_guard(self):
    def boom(_req):
      raise RuntimeError("model timeout")

    plugin = AuthorityRoutingPlugin(posture_router=boom, fail_closed=False)
    # Router ignored, but the guard still trips on an irreversible call.
    result = await _decide(plugin, "delete_file", {"path": "x"})
    assert result["posture"] == "DEFER"


# ---------------------------------------------------------------------------
# Verdict coercion
# ---------------------------------------------------------------------------


class TestVerdictCoercion:

  @pytest.mark.asyncio
  async def test_router_returns_bare_posture(self):
    plugin = AuthorityRoutingPlugin(posture_router=lambda _req: Posture.ADVISE)
    result = await _decide(plugin, "web_search", {})
    assert result["posture"] == "ADVISE"

  @pytest.mark.asyncio
  async def test_router_returns_posture_string(self):
    plugin = AuthorityRoutingPlugin(posture_router=lambda _req: "stop")
    result = await _decide(plugin, "web_search", {})
    assert result["posture"] == "STOP"

  @pytest.mark.asyncio
  async def test_router_returns_verdict_with_string_posture(self):
    # A hand-built verdict whose posture slipped through as a string.
    plugin = AuthorityRoutingPlugin(
        posture_router=lambda _req: PostureVerdict(
            posture="defer", scope="s"  # type: ignore[arg-type]
        )
    )
    result = await _decide(plugin, "web_search", {})
    assert result["posture"] == "DEFER"
    assert result["scope"] == "s"


# ---------------------------------------------------------------------------
# Customization
# ---------------------------------------------------------------------------


class TestCustomization:

  @pytest.mark.asyncio
  async def test_custom_irreversible_keywords(self):
    plugin = AuthorityRoutingPlugin(irreversible_keywords=["yeet"])
    # Default keyword no longer trips...
    assert await _decide(plugin, "delete_file", {"path": "x"}) is None
    # ...but the custom one does.
    result = await _decide(plugin, "yeet_file", {"path": "x"})
    assert result["posture"] == "DEFER"

  @pytest.mark.asyncio
  async def test_custom_approval_markers(self):
    plugin = AuthorityRoutingPlugin(pending_approval_markers=["frozen"])
    result = await _decide(
        plugin,
        "web_search",
        {},
        state={"authority_context": "this workflow is frozen"},
    )
    assert result["posture"] == "DEFER"

  @pytest.mark.asyncio
  async def test_custom_context_state_key(self):
    plugin = AuthorityRoutingPlugin(context_state_key="approvals")
    result = await _decide(
        plugin,
        "web_search",
        {},
        state={"approvals": "approval required"},
    )
    assert result["posture"] == "DEFER"

  @pytest.mark.asyncio
  async def test_router_receives_request_fields(self):
    seen = {}

    def router(req: AuthorityRequest):
      seen["action"] = req.action
      seen["args"] = req.tool_args
      seen["context"] = req.context
      return Posture.EXECUTE

    plugin = AuthorityRoutingPlugin(posture_router=router)
    await _decide(
        plugin, "update_row", {"id": 9}, state={"authority_context": "ctx"}
    )
    assert seen == {"action": "update_row", "args": {"id": 9}, "context": "ctx"}


# ---------------------------------------------------------------------------
# Block response shape
# ---------------------------------------------------------------------------


class TestBlockShape:

  @pytest.mark.asyncio
  async def test_block_dict_has_all_fields(self):
    plugin = AuthorityRoutingPlugin(
        posture_router=_make_router(
            posture=Posture.STOP, scope="sc", rationale="ra"
        )
    )
    result = await _decide(plugin, "web_search", {})
    assert set(result) == {"error", "posture", "reason", "scope"}
    assert result["error"] == "authority_blocked"
    assert result["posture"] == "STOP"
    assert result["reason"] == "ra"
    assert result["scope"] == "sc"

  @pytest.mark.asyncio
  async def test_missing_state_is_safe(self):
    # A tool context without a usable state must not raise.
    plugin = AuthorityRoutingPlugin()

    class _NoState:
      state = None

    result = await plugin.before_tool_callback(
        tool=_FakeBaseTool("web_search"),
        tool_args={},
        tool_context=_NoState(),
    )
    assert result is None
