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

"""Tests for TulipGovernancePlugin.

Unlike AgentGovernancePlugin's tests, these run against the real
``tulip-agents`` package (``pip install tulip-agents``, or
``pip install .[tulip]``/``.[test]`` in this repo) rather than a fake —
``tulip.control``/``tulip.security.policy``/``tulip.security.audit`` are
pure dataclass logic with no LLM or network dependency, so there is nothing
to mock. The "not installed" path is exercised separately by removing the
real modules from ``sys.modules`` for the duration of one test.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import patch

import pytest

tulip = pytest.importorskip("tulip", reason="tulip-agents is not installed")

from google.adk_community.plugins.tulip_governance_plugin import (  # noqa: E402
    TulipGovernancePlugin,
)
from tulip.control import Action  # noqa: E402
from tulip.security.audit import AuditTrail  # noqa: E402
from tulip.security.policy import ControlPolicy  # noqa: E402


# ---------------------------------------------------------------------------
# Fake ADK types for testing (same pattern as test_agent_governance_plugin.py)
# ---------------------------------------------------------------------------


class _FakeBaseTool:

  def __init__(self, name: str):
    self.name = name


class _FakeToolContext:
  pass


def _action_from(tool: Any, args: dict) -> Action:
  """A simple action builder: production/payment tools carry those labels,
  everything else is a plain read."""
  if tool.name in ("refund", "shell_exec", "file_delete"):
    return Action(
        name=tool.name,
        asset=str(args.get("order_id", args.get("path", ""))),
        kind="payment" if tool.name == "refund" else "exec",
        environment="production",
    )
  return Action(name=tool.name, environment="staging")


@pytest.fixture()
def policy() -> ControlPolicy:
  # require_verification_score=0.0: this plugin has no VerificationResult
  # to offer approve() (there's no fact/evidence step for an arbitrary ADK
  # tool call), and ControlPolicy's own default (0.8) would otherwise push
  # every call to at least require_human for lack of one. See the "Note on
  # ControlPolicy" section of the plugin's module docstring.
  return ControlPolicy(
      require_verification_score=0.0,
      deny_for=frozenset({"exec"}),
      require_human_for=frozenset({"production", "payment"}),
  )


class TestTulipGovernancePlugin:
  """Tests for TulipGovernancePlugin construction and behavior."""

  def test_plugin_name(self, policy: ControlPolicy):
    plugin = TulipGovernancePlugin(policy=policy, action=_action_from)
    assert plugin.name == "tulip_governance"

  def test_creates_a_trail_when_none_given(self, policy: ControlPolicy):
    plugin = TulipGovernancePlugin(policy=policy, action=_action_from)
    assert isinstance(plugin.trail, AuditTrail)
    assert len(plugin.trail) == 0

  def test_uses_the_shared_trail_when_given(self, policy: ControlPolicy):
    trail = AuditTrail()
    plugin = TulipGovernancePlugin(
        policy=policy, action=_action_from, trail=trail
    )
    assert plugin.trail is trail

  @pytest.mark.asyncio
  async def test_allows_safe_tool_call(self, policy: ControlPolicy):
    plugin = TulipGovernancePlugin(policy=policy, action=_action_from)

    result = await plugin.before_tool_callback(
        tool=_FakeBaseTool("web_search"),
        tool_args={"query": "test"},
        tool_context=_FakeToolContext(),
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_denies_a_hard_denied_tool_call(self, policy: ControlPolicy):
    plugin = TulipGovernancePlugin(policy=policy, action=_action_from)

    result = await plugin.before_tool_callback(
        tool=_FakeBaseTool("shell_exec"),
        tool_args={"cmd": "rm -rf /"},
        tool_context=_FakeToolContext(),
    )
    assert result is not None
    assert result["error"] == "policy_denied"
    assert "exec" in result["reason"]

  @pytest.mark.asyncio
  async def test_holds_a_production_call_for_a_human(
      self, policy: ControlPolicy
  ):
    """require_human is NOT collapsed into policy_denied — a distinct
    response, so a caller can route the two differently."""
    plugin = TulipGovernancePlugin(policy=policy, action=_action_from)

    result = await plugin.before_tool_callback(
        tool=_FakeBaseTool("refund"),
        tool_args={"order_id": "ord-9"},
        tool_context=_FakeToolContext(),
    )
    assert result is not None
    assert result["error"] == "held_for_approval"
    assert "production" in result["reason"] or "payment" in result["reason"]

  @pytest.mark.asyncio
  async def test_audit_trail_records_every_decision_allow_and_deny(
      self, policy: ControlPolicy
  ):
    trail = AuditTrail()
    plugin = TulipGovernancePlugin(
        policy=policy, action=_action_from, trail=trail
    )

    await plugin.before_tool_callback(
        tool=_FakeBaseTool("web_search"),
        tool_args={},
        tool_context=_FakeToolContext(),
    )
    await plugin.before_tool_callback(
        tool=_FakeBaseTool("shell_exec"),
        tool_args={"cmd": "rm -rf /"},
        tool_context=_FakeToolContext(),
    )
    await plugin.before_tool_callback(
        tool=_FakeBaseTool("refund"),
        tool_args={"order_id": "ord-9"},
        tool_context=_FakeToolContext(),
    )

    assert len(trail) == 3
    records = trail.records()
    assert records[0].payload["tool"] == "web_search"
    assert records[0].payload["outcome"] == "allow"
    assert records[1].payload["tool"] == "shell_exec"
    assert records[1].payload["outcome"] == "deny"
    assert records[2].payload["tool"] == "refund"
    assert records[2].payload["outcome"] == "require_human"
    # Tamper-evident: the chain verifies intact, and breaks if a record
    # is edited after the fact.
    assert trail.verify() is True
    object.__setattr__(records[1], "payload", {"tool": "tampered"})
    tampered_trail = AuditTrail.from_records(records)
    assert tampered_trail.verify() is False

  def test_raises_import_error_when_tulip_missing_and_not_fail_open(
      self, policy: ControlPolicy
  ):
    removed = {
        k: v
        for k, v in sys.modules.items()
        if k == "tulip" or k.startswith("tulip.")
    }
    for key in removed:
      del sys.modules[key]
    try:
      with patch.dict(sys.modules, {"tulip": None, "tulip.control": None}):
        with pytest.raises(ImportError, match="tulip-agents"):
          TulipGovernancePlugin(policy=policy, action=_action_from)
    finally:
      sys.modules.update(removed)

  @pytest.mark.asyncio
  async def test_fail_open_allows_all_when_tulip_missing(
      self, policy: ControlPolicy
  ):
    removed = {
        k: v
        for k, v in sys.modules.items()
        if k == "tulip" or k.startswith("tulip.")
    }
    for key in removed:
      del sys.modules[key]
    try:
      with patch.dict(sys.modules, {"tulip": None, "tulip.control": None}):
        plugin = TulipGovernancePlugin(
            policy=policy, action=_action_from, fail_open=True
        )
      result = await plugin.before_tool_callback(
          tool=_FakeBaseTool("shell_exec"),
          tool_args={"cmd": "rm -rf /"},
          tool_context=_FakeToolContext(),
      )
      assert result is None
    finally:
      sys.modules.update(removed)
