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

The plugin's ADK integration surface is small (a state dict on the invocation
context and on the tool context), so these tests use light fakes instead of a
full ADK runtime. The deterministic guard is tested directly; the model router
is stubbed, so no live model key is required.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from google.adk_community.plugins.authority_routing_plugin import (
    AuthorityRoutingPlugin,
    CallableRouter,
    Posture,
    PostureRouter,
    PostureVerdict,
    stricter,
)


# ---------------------------------------------------------------------------
# Fakes for the small slice of ADK the plugin touches
# ---------------------------------------------------------------------------


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state


class _FakeInvocationContext:
    def __init__(self, state: dict[str, Any]) -> None:
        self.session = _FakeSession(state)


class _FakeToolContext:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state


class _FakePart:
    def __init__(self, text: Optional[str]) -> None:
        self.text = text


class _FakeContent:
    def __init__(self, *texts: str) -> None:
        self.parts = [_FakePart(t) for t in texts]


class _StubRouter(PostureRouter):
    """A router that always returns a fixed verdict (or raises)."""

    def __init__(
        self, verdict: Optional[PostureVerdict] = None, raises: bool = False
    ) -> None:
        self._verdict = verdict
        self._raises = raises

    async def route(self, request: str, context: str) -> PostureVerdict:
        if self._raises:
            raise RuntimeError("boom")
        assert self._verdict is not None
        return self._verdict


async def _route(
    plugin: AuthorityRoutingPlugin, state: dict, request: str
) -> None:
    await plugin.on_user_message_callback(
        invocation_context=_FakeInvocationContext(state),
        user_message=_FakeContent(request),
    )


async def _tool(plugin: AuthorityRoutingPlugin, state: dict, name: str):
    return await plugin.before_tool_callback(
        tool=_FakeTool(name),
        tool_args={},
        tool_context=_FakeToolContext(state),
    )


# ---------------------------------------------------------------------------
# Composition + guard (pure, no async)
# ---------------------------------------------------------------------------


def test_plugin_name() -> None:
    assert AuthorityRoutingPlugin().name == "authority_routing"


def test_stricter_is_most_restrictive_wins() -> None:
    assert stricter(Posture.EXECUTE, Posture.ADVISE) == Posture.ADVISE
    assert stricter(Posture.ADVISE, Posture.DEFER) == Posture.DEFER
    assert stricter(Posture.DEFER, Posture.STOP) == Posture.STOP
    assert stricter(Posture.EXECUTE, Posture.EXECUTE) == Posture.EXECUTE


def test_floor_irreversible_keyword_forces_defer() -> None:
    plugin = AuthorityRoutingPlugin()
    posture, reason = plugin.deterministic_floor("please delete the table", "")
    assert posture == Posture.DEFER
    assert "delete" in reason


def test_floor_pending_marker_in_context_forces_defer() -> None:
    plugin = AuthorityRoutingPlugin()
    posture, reason = plugin.deterministic_floor(
        "apply the change", "this is pending approval from the owner"
    )
    assert posture == Posture.DEFER
    assert "pending approval" in reason


def test_floor_clean_request_is_execute() -> None:
    plugin = AuthorityRoutingPlugin()
    posture, _ = plugin.deterministic_floor("rename this local variable", "")
    assert posture == Posture.EXECUTE


def test_extra_irreversible_keywords_extend_the_guard() -> None:
    plugin = AuthorityRoutingPlugin(extra_irreversible_keywords=["frobnicate"])
    posture, reason = plugin.deterministic_floor("frobnicate the widget", "")
    assert posture == Posture.DEFER
    assert "frobnicate" in reason


# ---------------------------------------------------------------------------
# Routing + enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guard_only_mode_allows_clean_execute() -> None:
    plugin = AuthorityRoutingPlugin()  # no router
    state: dict[str, Any] = {}
    await _route(plugin, state, "rename a local variable")
    assert state["authority:posture"]["posture"] == "EXECUTE"
    assert await _tool(plugin, state, "edit_file") is None


@pytest.mark.asyncio
async def test_advise_denies_side_effecting_tool() -> None:
    plugin = AuthorityRoutingPlugin(
        router=_StubRouter(
            PostureVerdict(Posture.ADVISE, "", "user wants advice")
        )
    )
    state: dict[str, Any] = {}
    await _route(plugin, state, "what would you change about the retry settings?")
    result = await _tool(plugin, state, "edit_file")
    assert result is not None
    assert result["error"] == "authority_denied"
    assert result["posture"] == "ADVISE"


@pytest.mark.asyncio
async def test_guard_tightens_execute_to_defer_on_irreversible() -> None:
    # Router says EXECUTE, but the request text trips the irreversibility wire.
    plugin = AuthorityRoutingPlugin(
        router=_StubRouter(PostureVerdict(Posture.EXECUTE, "the config", "ok"))
    )
    state: dict[str, Any] = {}
    await _route(plugin, state, "deploy the config to production")
    assert state["authority:posture"]["posture"] == "DEFER"
    result = await _tool(plugin, state, "shell")
    assert result is not None
    assert result["posture"] == "DEFER"


@pytest.mark.asyncio
async def test_execute_clean_allows_tool() -> None:
    plugin = AuthorityRoutingPlugin(
        router=_StubRouter(PostureVerdict(Posture.EXECUTE, "one file", "bounded"))
    )
    state: dict[str, Any] = {}
    await _route(plugin, state, "rename foo to bar in utils")
    assert await _tool(plugin, state, "edit_file") is None


@pytest.mark.asyncio
async def test_router_exception_fails_closed_to_defer() -> None:
    plugin = AuthorityRoutingPlugin(router=_StubRouter(raises=True))
    state: dict[str, Any] = {}
    await _route(plugin, state, "rename a variable")
    assert state["authority:posture"]["posture"] == "DEFER"
    assert (await _tool(plugin, state, "edit_file"))["error"] == "authority_denied"


@pytest.mark.asyncio
async def test_router_exception_fails_open_when_configured() -> None:
    plugin = AuthorityRoutingPlugin(
        router=_StubRouter(raises=True), fail_closed=False
    )
    state: dict[str, Any] = {}
    await _route(plugin, state, "rename a variable")
    assert state["authority:posture"]["posture"] == "EXECUTE"
    assert await _tool(plugin, state, "edit_file") is None


@pytest.mark.asyncio
async def test_read_only_tool_allowed_under_advise() -> None:
    plugin = AuthorityRoutingPlugin(
        router=_StubRouter(PostureVerdict(Posture.ADVISE, "", "advice")),
        read_only_tool_names={"read_file"},
    )
    state: dict[str, Any] = {}
    await _route(plugin, state, "review this module")
    assert await _tool(plugin, state, "read_file") is None
    assert (await _tool(plugin, state, "edit_file")) is not None


@pytest.mark.asyncio
async def test_read_only_tool_blocked_under_stop() -> None:
    plugin = AuthorityRoutingPlugin(
        router=_StubRouter(PostureVerdict(Posture.STOP, "", "disallowed")),
        read_only_tool_names={"read_file"},
    )
    state: dict[str, Any] = {}
    await _route(plugin, state, "do the disallowed thing")
    assert (await _tool(plugin, state, "read_file")) is not None


@pytest.mark.asyncio
async def test_tool_without_routed_decision_fails_closed() -> None:
    plugin = AuthorityRoutingPlugin()
    result = await _tool(plugin, {}, "edit_file")
    assert result is not None
    assert result["error"] == "authority_denied"


@pytest.mark.asyncio
async def test_tool_without_routed_decision_allowed_when_open() -> None:
    plugin = AuthorityRoutingPlugin(fail_closed=False)
    assert await _tool(plugin, {}, "edit_file") is None


@pytest.mark.asyncio
async def test_callable_router_wraps_sync_callable() -> None:
    def sync_router(request: str, context: str) -> PostureVerdict:
        return PostureVerdict(Posture.ADVISE, "", "sync")

    plugin = AuthorityRoutingPlugin(router=CallableRouter(sync_router))
    state: dict[str, Any] = {}
    await _route(plugin, state, "review this")
    assert state["authority:posture"]["posture"] == "ADVISE"


@pytest.mark.asyncio
async def test_audit_log_records_route_and_deny() -> None:
    plugin = AuthorityRoutingPlugin(
        router=_StubRouter(PostureVerdict(Posture.ADVISE, "", "advice"))
    )
    state: dict[str, Any] = {}
    await _route(plugin, state, "review this")
    await _tool(plugin, state, "edit_file")
    events = [e["event"] for e in plugin.audit_log]
    assert "route" in events
    assert "deny" in events
