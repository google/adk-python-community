# Copyright 2026 SidClaw
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Unit tests for SidClaw governance service."""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from google.adk_community.governance.sidclaw_governance import (
    SidClawGovernanceConfig,
    SidClawGovernanceService,
    _safe_serialize,
)


# -------------------------------------------------------------------
# Fixtures & helpers
# -------------------------------------------------------------------


class FakeDecision:
  """Minimal stand-in for sidclaw.EvaluateResponse."""

  def __init__(
      self,
      decision: str = "allow",
      trace_id: str = "trc_test123",
      reason: str = "allowed by policy",
      approval_request_id: Optional[str] = None,
      policy_rule_id: Optional[str] = None,
  ):
    self.decision = decision
    self.trace_id = trace_id
    self.reason = reason
    self.approval_request_id = approval_request_id
    self.policy_rule_id = policy_rule_id


class FakeApprovalStatus:
  """Minimal stand-in for sidclaw.ApprovalStatusResponse."""

  def __init__(
      self,
      status: str = "approved",
      approver_name: Optional[str] = "reviewer",
      decision_note: Optional[str] = None,
  ):
    self.status = status
    self.approver_name = approver_name
    self.decision_note = decision_note


def _make_tool(name: str = "search_docs") -> MagicMock:
  tool = MagicMock()
  tool.name = name
  return tool


def _make_tool_context() -> MagicMock:
  ctx = MagicMock()
  ctx.state = {}
  return ctx


def _make_client(
    decision: FakeDecision | None = None,
    approval_status: FakeApprovalStatus | None = None,
) -> AsyncMock:
  client = AsyncMock()
  client.evaluate = AsyncMock(
      return_value=decision or FakeDecision()
  )
  client.wait_for_approval = AsyncMock(
      return_value=approval_status or FakeApprovalStatus()
  )
  client.record_outcome = AsyncMock(return_value=None)
  return client


# -------------------------------------------------------------------
# Tests: SidClawGovernanceConfig
# -------------------------------------------------------------------


class TestGovernanceConfig:
  """Tests for SidClawGovernanceConfig defaults."""

  def test_default_values(self):
    config = SidClawGovernanceConfig()
    assert config.default_classification == "internal"
    assert config.resource_scope == "google_adk"
    assert config.wait_for_approval is True
    assert config.approval_timeout_seconds == 300.0
    assert config.state_key == "_sidclaw_trace_id"

  def test_custom_values(self):
    config = SidClawGovernanceConfig(
        default_classification="confidential",
        tool_classifications={"delete_db": "restricted"},
        resource_scope="prod",
        wait_for_approval=False,
    )
    assert config.default_classification == "confidential"
    assert config.tool_classifications["delete_db"] == "restricted"
    assert config.resource_scope == "prod"
    assert config.wait_for_approval is False


# -------------------------------------------------------------------
# Tests: before_tool_callback — allow
# -------------------------------------------------------------------


class TestBeforeToolAllow:
  """Tests for the before-tool callback when policy allows."""

  @pytest.mark.asyncio
  async def test_allow_returns_none(self):
    """Allowed tool calls return None (proceed)."""
    client = _make_client(FakeDecision(decision="allow"))
    svc = SidClawGovernanceService(client=client)

    result = await svc.before_tool_callback(
        _make_tool(), {"query": "test"}, _make_tool_context()
    )

    assert result is None

  @pytest.mark.asyncio
  async def test_allow_stores_trace_id(self):
    """Trace ID is stored in tool_context.state."""
    client = _make_client(
        FakeDecision(decision="allow", trace_id="trc_abc")
    )
    svc = SidClawGovernanceService(client=client)
    ctx = _make_tool_context()

    await svc.before_tool_callback(
        _make_tool(), {}, ctx
    )

    assert ctx.state["_sidclaw_trace_id"] == "trc_abc"

  @pytest.mark.asyncio
  async def test_allow_calls_evaluate_with_correct_params(self):
    """Evaluate is called with tool name and classification."""
    client = _make_client(FakeDecision(decision="allow"))
    config = SidClawGovernanceConfig(
        default_classification="confidential",
        resource_scope="prod",
    )
    svc = SidClawGovernanceService(
        client=client, config=config
    )

    await svc.before_tool_callback(
        _make_tool("my_tool"), {"key": "val"}, _make_tool_context()
    )

    call_args = client.evaluate.call_args[0][0]
    assert call_args["operation"] == "my_tool"
    assert call_args["target_integration"] == "google_adk"
    assert call_args["resource_scope"] == "prod"
    assert call_args["data_classification"] == "confidential"

  @pytest.mark.asyncio
  async def test_per_tool_classification_override(self):
    """Per-tool classification overrides the default."""
    client = _make_client(FakeDecision(decision="allow"))
    config = SidClawGovernanceConfig(
        default_classification="internal",
        tool_classifications={"send_email": "restricted"},
    )
    svc = SidClawGovernanceService(
        client=client, config=config
    )

    await svc.before_tool_callback(
        _make_tool("send_email"), {}, _make_tool_context()
    )

    call_args = client.evaluate.call_args[0][0]
    assert call_args["data_classification"] == "restricted"


# -------------------------------------------------------------------
# Tests: before_tool_callback — deny
# -------------------------------------------------------------------


class TestBeforeToolDeny:
  """Tests for the before-tool callback when policy denies."""

  @pytest.mark.asyncio
  async def test_deny_returns_error(self):
    """Denied tool calls return an error dict."""
    client = _make_client(
        FakeDecision(
            decision="deny",
            reason="destructive operation blocked",
        )
    )
    svc = SidClawGovernanceService(client=client)

    result = await svc.before_tool_callback(
        _make_tool(), {}, _make_tool_context()
    )

    assert result is not None
    assert "error" in result
    assert "destructive operation blocked" in result["error"]


# -------------------------------------------------------------------
# Tests: before_tool_callback — approval_required
# -------------------------------------------------------------------


class TestBeforeToolApproval:
  """Tests for the before-tool callback with approval flow."""

  @pytest.mark.asyncio
  async def test_approval_wait_approved(self):
    """Approved requests return None (proceed)."""
    client = _make_client(
        decision=FakeDecision(
            decision="approval_required",
            approval_request_id="apr_123",
            reason="needs review",
        ),
        approval_status=FakeApprovalStatus(status="approved"),
    )
    svc = SidClawGovernanceService(client=client)

    result = await svc.before_tool_callback(
        _make_tool(), {}, _make_tool_context()
    )

    assert result is None
    client.wait_for_approval.assert_awaited_once()

  @pytest.mark.asyncio
  async def test_approval_wait_denied(self):
    """Denied approval returns error dict."""
    client = _make_client(
        decision=FakeDecision(
            decision="approval_required",
            approval_request_id="apr_456",
            reason="risky operation",
        ),
        approval_status=FakeApprovalStatus(
            status="denied",
            decision_note="too risky",
        ),
    )
    svc = SidClawGovernanceService(client=client)

    result = await svc.before_tool_callback(
        _make_tool(), {}, _make_tool_context()
    )

    assert result is not None
    assert "denied" in result["error"].lower()
    assert "too risky" in result["error"]

  @pytest.mark.asyncio
  async def test_approval_no_wait_returns_error(self):
    """When wait_for_approval=False, returns error immediately."""
    client = _make_client(
        FakeDecision(
            decision="approval_required",
            approval_request_id="apr_789",
            reason="needs human review",
        )
    )
    config = SidClawGovernanceConfig(wait_for_approval=False)
    svc = SidClawGovernanceService(
        client=client, config=config
    )

    result = await svc.before_tool_callback(
        _make_tool(), {}, _make_tool_context()
    )

    assert result is not None
    assert "apr_789" in result["error"]
    client.wait_for_approval.assert_not_awaited()


# -------------------------------------------------------------------
# Tests: before_tool_callback — error handling
# -------------------------------------------------------------------


class TestBeforeToolErrors:
  """Tests for error handling in before-tool callback."""

  @pytest.mark.asyncio
  async def test_evaluate_failure_fails_closed_by_default(self):
    """By default, tool is blocked when SidClaw is unreachable."""
    client = _make_client()
    client.evaluate = AsyncMock(
        side_effect=Exception("connection refused")
    )
    svc = SidClawGovernanceService(client=client)

    result = await svc.before_tool_callback(
        _make_tool(), {}, _make_tool_context()
    )

    assert result is not None
    assert "error" in result

  @pytest.mark.asyncio
  async def test_evaluate_failure_fails_open_when_configured(self):
    """With fail_open=True, tool proceeds on SidClaw failure."""
    client = _make_client()
    client.evaluate = AsyncMock(
        side_effect=Exception("connection refused")
    )
    config = SidClawGovernanceConfig(fail_open=True)
    svc = SidClawGovernanceService(
        client=client, config=config
    )

    result = await svc.before_tool_callback(
        _make_tool(), {}, _make_tool_context()
    )

    assert result is None  # Fail-open

  @pytest.mark.asyncio
  async def test_approval_request_id_none_returns_error(self):
    """approval_required with no ID returns error immediately."""
    client = _make_client(
        FakeDecision(
            decision="approval_required",
            approval_request_id=None,
            reason="flagged",
        )
    )
    svc = SidClawGovernanceService(client=client)

    result = await svc.before_tool_callback(
        _make_tool(), {}, _make_tool_context()
    )

    assert result is not None
    assert "Approval required" in result["error"]
    client.wait_for_approval.assert_not_awaited()

  @pytest.mark.asyncio
  async def test_tool_without_name_uses_fallback(self):
    """Tool with no name attribute uses 'unknown_tool'."""
    client = _make_client(FakeDecision(decision="allow"))
    svc = SidClawGovernanceService(client=client)
    tool = MagicMock(spec=[])  # No name attribute

    await svc.before_tool_callback(tool, {}, _make_tool_context())

    call_args = client.evaluate.call_args[0][0]
    assert call_args["operation"] == "unknown_tool"

  @pytest.mark.asyncio
  async def test_wait_for_approval_exception_returns_error(self):
    """Exception during approval wait returns error dict."""
    client = _make_client(
        FakeDecision(
            decision="approval_required",
            approval_request_id="apr_exc",
            reason="needs review",
        )
    )
    client.wait_for_approval = AsyncMock(
        side_effect=Exception("timeout")
    )
    svc = SidClawGovernanceService(client=client)

    result = await svc.before_tool_callback(
        _make_tool(), {}, _make_tool_context()
    )

    assert result is not None
    assert "apr_exc" in result["error"]


# -------------------------------------------------------------------
# Tests: after_tool_callback
# -------------------------------------------------------------------


class TestAfterToolCallback:
  """Tests for the after-tool callback."""

  @pytest.mark.asyncio
  async def test_records_success_outcome(self):
    """Records success when tool returns normally."""
    client = _make_client()
    svc = SidClawGovernanceService(client=client)
    ctx = _make_tool_context()
    ctx.state["_sidclaw_trace_id"] = "trc_xyz"

    await svc.after_tool_callback(
        _make_tool(), {}, ctx, {"result": "ok"}
    )

    client.record_outcome.assert_awaited_once()
    call_args = client.record_outcome.call_args[0]
    assert call_args[0] == "trc_xyz"
    assert call_args[1]["status"] == "success"

  @pytest.mark.asyncio
  async def test_records_error_outcome(self):
    """Records error when tool returns error dict."""
    client = _make_client()
    svc = SidClawGovernanceService(client=client)
    ctx = _make_tool_context()
    ctx.state["_sidclaw_trace_id"] = "trc_err"

    await svc.after_tool_callback(
        _make_tool(),
        {},
        ctx,
        {"error": "something failed"},
    )

    call_args = client.record_outcome.call_args[0]
    assert call_args[1]["status"] == "error"

  @pytest.mark.asyncio
  async def test_no_trace_id_is_noop(self):
    """Does nothing when no trace ID is in state."""
    client = _make_client()
    svc = SidClawGovernanceService(client=client)
    ctx = _make_tool_context()

    await svc.after_tool_callback(
        _make_tool(), {}, ctx, "result"
    )

    client.record_outcome.assert_not_awaited()

  @pytest.mark.asyncio
  async def test_record_failure_does_not_raise(self):
    """Recording failure is logged but does not propagate."""
    client = _make_client()
    client.record_outcome = AsyncMock(
        side_effect=Exception("network error")
    )
    svc = SidClawGovernanceService(client=client)
    ctx = _make_tool_context()
    ctx.state["_sidclaw_trace_id"] = "trc_fail"

    # Should not raise.
    await svc.after_tool_callback(
        _make_tool(), {}, ctx, "result"
    )


# -------------------------------------------------------------------
# Tests: _safe_serialize
# -------------------------------------------------------------------


class TestSafeSerialize:
  """Tests for the argument serialiser."""

  def test_dict_passthrough(self):
    result = _safe_serialize({"key": "val", "n": 42})
    assert result == {"key": "val", "n": 42}

  def test_complex_values_stringified(self):
    result = _safe_serialize({"obj": object()})
    assert isinstance(result["obj"], str)

  def test_non_dict_wrapped(self):
    result = _safe_serialize("raw string")
    assert result == {"raw": "raw string"}
