# Copyright 2025 Google LLC
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

"""Tests for VeronicaGovernancePlugin."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from google.adk_community.governance import GovernanceConfig
from google.adk_community.governance import VeronicaGovernancePlugin


def _make_callback_context(agent_name: str = "test_agent"):
  """Create a minimal mock CallbackContext."""
  ctx = mock.MagicMock()
  ctx.agent_name = agent_name
  ctx.function_call_id = "call_001"
  return ctx


def _make_llm_request(model: str = "gemini-2.5-pro"):
  """Create a minimal mock LlmRequest."""
  req = mock.MagicMock()
  req.model = model
  return req


def _make_llm_response(
    input_tokens: int = 100,
    output_tokens: int = 50,
    error_code: str | None = None,
):
  """Create a minimal mock LlmResponse."""
  resp = mock.MagicMock()
  resp.error_code = error_code
  usage = SimpleNamespace(
      prompt_token_count=input_tokens,
      candidates_token_count=output_tokens,
  )
  resp.usage_metadata = usage
  return resp


def _make_tool(name: str = "search"):
  """Create a minimal mock BaseTool."""
  tool = mock.MagicMock()
  tool.name = name
  return tool


def _make_invocation_context():
  """Create a minimal mock InvocationContext."""
  return mock.MagicMock()


class TestBeforeModelCallback:

  @pytest.mark.asyncio
  async def test_allows_within_budget(self):
    plugin = VeronicaGovernancePlugin()
    ctx = _make_callback_context()
    req = _make_llm_request()
    result = await plugin.before_model_callback(
        callback_context=ctx, llm_request=req
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_blocks_when_budget_exceeded(self):
    config = GovernanceConfig(max_cost_usd=0.001, agent_max_cost_usd=0.001)
    plugin = VeronicaGovernancePlugin(config=config)
    plugin._budget.record("test_agent", 0.002)

    ctx = _make_callback_context()
    req = _make_llm_request()
    result = await plugin.before_model_callback(
        callback_context=ctx, llm_request=req
    )
    assert result is not None
    assert result.error_code == "GOVERNANCE_BUDGET_EXCEEDED"

  @pytest.mark.asyncio
  async def test_blocks_when_circuit_open(self):
    config = GovernanceConfig(failure_threshold=2)
    plugin = VeronicaGovernancePlugin(config=config)
    plugin._circuit_breaker.record_failure("test_agent")
    plugin._circuit_breaker.record_failure("test_agent")

    ctx = _make_callback_context()
    req = _make_llm_request()
    result = await plugin.before_model_callback(
        callback_context=ctx, llm_request=req
    )
    assert result is not None
    assert result.error_code == "GOVERNANCE_CIRCUIT_OPEN"

  @pytest.mark.asyncio
  async def test_degrades_model_near_budget_limit(self):
    config = GovernanceConfig(
        max_cost_usd=1.0,
        degradation_threshold=0.8,
        fallback_model="gemini-2.0-flash-lite",
    )
    plugin = VeronicaGovernancePlugin(config=config)
    plugin._budget.record("test_agent", 0.85)

    ctx = _make_callback_context()
    req = _make_llm_request(model="gemini-2.5-pro")
    result = await plugin.before_model_callback(
        callback_context=ctx, llm_request=req
    )
    # Returns None (proceeds), but request model was mutated
    assert result is None
    assert req.model == "gemini-2.0-flash-lite"

  @pytest.mark.asyncio
  async def test_degradation_recorded_in_events(self):
    config = GovernanceConfig(
        max_cost_usd=1.0,
        degradation_threshold=0.5,
        fallback_model="gemini-2.0-flash-lite",
    )
    plugin = VeronicaGovernancePlugin(config=config)
    plugin._budget.record("test_agent", 0.6)

    ctx = _make_callback_context()
    req = _make_llm_request(model="gemini-2.5-pro")
    await plugin.before_model_callback(callback_context=ctx, llm_request=req)
    events = plugin._degradation.events
    assert len(events) == 1
    assert events[0].agent_name == "test_agent"
    assert events[0].original_model == "gemini-2.5-pro"
    assert events[0].fallback_model == "gemini-2.0-flash-lite"

  @pytest.mark.asyncio
  async def test_no_degradation_when_not_configured(self):
    config = GovernanceConfig(
        max_cost_usd=1.0,
        degradation_threshold=0.8,
        fallback_model=None,
    )
    plugin = VeronicaGovernancePlugin(config=config)
    plugin._budget.record("test_agent", 0.9)

    ctx = _make_callback_context()
    req = _make_llm_request(model="gemini-2.5-pro")
    await plugin.before_model_callback(callback_context=ctx, llm_request=req)
    assert req.model == "gemini-2.5-pro"


class TestAfterModelCallback:

  @pytest.mark.asyncio
  async def test_records_cost(self):
    plugin = VeronicaGovernancePlugin()
    ctx = _make_callback_context()
    resp = _make_llm_response(input_tokens=1000, output_tokens=500)

    await plugin.after_model_callback(callback_context=ctx, llm_response=resp)
    snap = plugin._budget.snapshot()
    assert snap.org_spent_usd > 0

  @pytest.mark.asyncio
  async def test_resets_circuit_breaker_on_success(self):
    config = GovernanceConfig(failure_threshold=3)
    plugin = VeronicaGovernancePlugin(config=config)
    plugin._circuit_breaker.record_failure("test_agent")
    plugin._circuit_breaker.record_failure("test_agent")

    ctx = _make_callback_context()
    resp = _make_llm_response()
    await plugin.after_model_callback(callback_context=ctx, llm_response=resp)
    from google.adk_community.governance._circuit_breaker import CircuitState

    assert (
        plugin._circuit_breaker.get_state("test_agent") == CircuitState.CLOSED
    )

  @pytest.mark.asyncio
  async def test_handles_missing_usage_metadata(self):
    plugin = VeronicaGovernancePlugin()
    ctx = _make_callback_context()
    resp = mock.MagicMock()
    resp.usage_metadata = None
    result = await plugin.after_model_callback(
        callback_context=ctx, llm_response=resp
    )
    assert result is None
    assert plugin._budget.snapshot().org_spent_usd == 0.0


class TestModelErrorCallback:

  @pytest.mark.asyncio
  async def test_records_failure(self):
    config = GovernanceConfig(failure_threshold=2)
    plugin = VeronicaGovernancePlugin(config=config)
    ctx = _make_callback_context()
    req = _make_llm_request()

    await plugin.on_model_error_callback(
        callback_context=ctx,
        llm_request=req,
        error=RuntimeError("test"),
    )
    from google.adk_community.governance._circuit_breaker import CircuitState

    assert (
        plugin._circuit_breaker.get_state("test_agent") == CircuitState.CLOSED
    )

    await plugin.on_model_error_callback(
        callback_context=ctx,
        llm_request=req,
        error=RuntimeError("test"),
    )
    assert plugin._circuit_breaker.get_state("test_agent") == CircuitState.OPEN


class TestToolCallbacks:

  @pytest.mark.asyncio
  async def test_allows_normal_tool(self):
    plugin = VeronicaGovernancePlugin()
    tool = _make_tool("search")
    ctx = _make_callback_context()
    result = await plugin.before_tool_callback(
        tool=tool, tool_args={}, tool_context=ctx
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_blocks_disallowed_tool(self):
    config = GovernanceConfig(blocked_tools=["delete_all"])
    plugin = VeronicaGovernancePlugin(config=config)
    tool = _make_tool("delete_all")
    ctx = _make_callback_context()
    result = await plugin.before_tool_callback(
        tool=tool, tool_args={}, tool_context=ctx
    )
    assert result is not None
    assert "blocked" in result["error"]

  @pytest.mark.asyncio
  async def test_disables_tool_on_degrade(self):
    config = GovernanceConfig(
        max_cost_usd=1.0,
        degradation_threshold=0.5,
        disable_tools_on_degrade=["expensive_search"],
    )
    plugin = VeronicaGovernancePlugin(config=config)
    plugin._budget.record("test_agent", 0.6)

    tool = _make_tool("expensive_search")
    ctx = _make_callback_context()
    result = await plugin.before_tool_callback(
        tool=tool, tool_args={}, tool_context=ctx
    )
    assert result is not None
    assert "disabled" in result["error"]

  @pytest.mark.asyncio
  async def test_after_tool_callback_returns_none(self):
    plugin = VeronicaGovernancePlugin()
    tool = _make_tool("search")
    ctx = _make_callback_context()
    # Simulate before_tool to set start time
    await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=ctx)
    result = await plugin.after_tool_callback(
        tool=tool, tool_args={}, tool_context=ctx, result={"ok": True}
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_tool_error_trips_circuit_breaker(self):
    config = GovernanceConfig(failure_threshold=1)
    plugin = VeronicaGovernancePlugin(config=config)
    tool = _make_tool("search")
    ctx = _make_callback_context()
    await plugin.on_tool_error_callback(
        tool=tool,
        tool_args={},
        tool_context=ctx,
        error=RuntimeError("fail"),
    )
    assert plugin._circuit_breaker.is_open("test_agent")

  @pytest.mark.asyncio
  async def test_tool_error_cleans_up_timing_entry(self):
    """on_tool_error_callback must remove the timing key set by before_tool_callback."""
    plugin = VeronicaGovernancePlugin()
    tool = _make_tool("search")
    ctx = _make_callback_context()
    # Record a start time as before_tool_callback would.
    await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=ctx)
    assert len(plugin._tool_start_times) == 1
    # Error path must clean it up.
    await plugin.on_tool_error_callback(
        tool=tool,
        tool_args={},
        tool_context=ctx,
        error=RuntimeError("fail"),
    )
    assert len(plugin._tool_start_times) == 0

  @pytest.mark.asyncio
  async def test_repeated_tool_errors_no_timing_leak(self):
    """_tool_start_times must not grow unbounded across many tool errors."""
    plugin = VeronicaGovernancePlugin()
    tool = _make_tool("search")
    for i in range(50):
      ctx = mock.MagicMock()
      ctx.agent_name = "test_agent"
      ctx.function_call_id = f"call_{i}"
      await plugin.before_tool_callback(
          tool=tool, tool_args={}, tool_context=ctx
      )
      await plugin.on_tool_error_callback(
          tool=tool,
          tool_args={},
          tool_context=ctx,
          error=RuntimeError("fail"),
      )
    assert len(plugin._tool_start_times) == 0


class TestRunLifecycle:

  @pytest.mark.asyncio
  async def test_before_run_returns_none(self):
    plugin = VeronicaGovernancePlugin()
    ic = _make_invocation_context()
    result = await plugin.before_run_callback(invocation_context=ic)
    assert result is None

  @pytest.mark.asyncio
  async def test_after_run_logs_summary(self, caplog):
    import logging

    plugin = VeronicaGovernancePlugin()
    ic = _make_invocation_context()
    await plugin.before_run_callback(invocation_context=ic)
    plugin._budget.record("agent_a", 0.05)

    with caplog.at_level(logging.INFO):
      await plugin.after_run_callback(invocation_context=ic)

    assert "[GOVERNANCE]" in caplog.text
    assert "Run complete" in caplog.text
    assert "Budget:" in caplog.text

  @pytest.mark.asyncio
  async def test_after_run_logs_degradation_events(self, caplog):
    import logging

    config = GovernanceConfig(
        max_cost_usd=1.0,
        degradation_threshold=0.5,
        fallback_model="gemini-2.0-flash-lite",
    )
    plugin = VeronicaGovernancePlugin(config=config)
    ic = _make_invocation_context()
    await plugin.before_run_callback(invocation_context=ic)

    # Trigger degradation
    plugin._budget.record("agent_a", 0.6)
    ctx = _make_callback_context("agent_a")
    req = _make_llm_request("gemini-2.5-pro")
    await plugin.before_model_callback(callback_context=ctx, llm_request=req)

    with caplog.at_level(logging.INFO):
      await plugin.after_run_callback(invocation_context=ic)

    assert "Degradation events" in caplog.text
    assert "gemini-2.0-flash-lite" in caplog.text


class TestErrorCodeHandling:

  @pytest.mark.asyncio
  async def test_soft_error_counts_as_failure(self):
    """error_code in LlmResponse should trigger circuit breaker failure."""
    config = GovernanceConfig(failure_threshold=2)
    plugin = VeronicaGovernancePlugin(config=config)
    ctx = _make_callback_context()
    resp = mock.MagicMock()
    resp.error_code = "500"
    resp.usage_metadata = None

    await plugin.after_model_callback(callback_context=ctx, llm_response=resp)
    await plugin.after_model_callback(callback_context=ctx, llm_response=resp)
    from google.adk_community.governance._circuit_breaker import CircuitState

    assert plugin._circuit_breaker.get_state("test_agent") == CircuitState.OPEN


class TestDegradationDedup:

  @pytest.mark.asyncio
  async def test_degradation_fires_once_per_agent(self):
    """Second call for same agent should not create another event."""
    config = GovernanceConfig(
        max_cost_usd=1.0,
        degradation_threshold=0.5,
        fallback_model="gemini-2.0-flash-lite",
    )
    plugin = VeronicaGovernancePlugin(config=config)
    plugin._budget.record("test_agent", 0.6)

    ctx = _make_callback_context()
    req1 = _make_llm_request("gemini-2.5-pro")
    await plugin.before_model_callback(callback_context=ctx, llm_request=req1)
    req2 = _make_llm_request("gemini-2.5-pro")
    await plugin.before_model_callback(callback_context=ctx, llm_request=req2)

    assert len(plugin._degradation.events) == 1
    assert req2.model == "gemini-2.0-flash-lite"


class TestAgentNameSanitization:

  @pytest.mark.asyncio
  async def test_newline_in_agent_name_sanitized(self):
    """Agent names with control chars should be sanitized."""
    plugin = VeronicaGovernancePlugin()
    ctx = _make_callback_context("evil\nagent")
    req = _make_llm_request()
    resp = _make_llm_response(input_tokens=100, output_tokens=50)
    await plugin.before_model_callback(callback_context=ctx, llm_request=req)
    await plugin.after_model_callback(callback_context=ctx, llm_response=resp)
    snap = plugin._budget.snapshot()
    assert len(snap.agent_spent) > 0
    for name in snap.agent_spent:
      assert "\n" not in name

  @pytest.mark.asyncio
  async def test_missing_agent_name_returns_unknown(self):
    """CallbackContext without agent_name should return 'unknown'."""
    plugin = VeronicaGovernancePlugin()
    ctx = mock.MagicMock()
    ctx.agent_name = None
    ctx.function_call_id = None
    req = _make_llm_request()
    result = await plugin.before_model_callback(
        callback_context=ctx, llm_request=req
    )
    assert result is None


class TestHalfOpenProbe:

  @pytest.mark.asyncio
  async def test_half_open_allows_probe(self):
    """HALF_OPEN state should allow one probe request."""
    config = GovernanceConfig(failure_threshold=1, recovery_timeout_s=0.0)
    plugin = VeronicaGovernancePlugin(config=config)
    # Trip the circuit
    plugin._circuit_breaker.record_failure("test_agent")
    # recovery_timeout_s=0 -> immediately HALF_OPEN
    ctx = _make_callback_context()
    req = _make_llm_request()
    result = await plugin.before_model_callback(
        callback_context=ctx, llm_request=req
    )
    # Probe should be allowed (returns None)
    assert result is None

  @pytest.mark.asyncio
  async def test_half_open_blocks_second_probe(self):
    """Second concurrent probe should be rejected."""
    config = GovernanceConfig(failure_threshold=1, recovery_timeout_s=0.0)
    plugin = VeronicaGovernancePlugin(config=config)
    plugin._circuit_breaker.record_failure("test_agent")
    ctx = _make_callback_context()
    # First probe succeeds
    req1 = _make_llm_request()
    result1 = await plugin.before_model_callback(
        callback_context=ctx, llm_request=req1
    )
    assert result1 is None
    # Second probe blocked (probe_in_flight still True)
    req2 = _make_llm_request()
    result2 = await plugin.before_model_callback(
        callback_context=ctx, llm_request=req2
    )
    assert result2 is not None
    assert "probe already in flight" in result2.error_message


class TestOpenedAtIdempotency:

  def test_opened_at_not_reset_on_repeated_failure(self):
    """opened_at must not change once circuit is OPEN."""
    from google.adk_community.governance._circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_s=60.0)
    cb.record_failure("agent_a")  # trips to OPEN
    opened_at_1 = cb._agents["agent_a"].opened_at
    assert opened_at_1 is not None

    import time as _time

    _time.sleep(0.01)  # ensure monotonic advances
    cb.record_failure("agent_a")  # second failure while OPEN
    opened_at_2 = cb._agents["agent_a"].opened_at
    assert opened_at_2 == opened_at_1  # must NOT reset


class TestToolErrorTimingCleanup:

  @pytest.mark.asyncio
  async def test_tool_error_cleans_timing_entry(self):
    """on_tool_error_callback must remove the timing entry."""
    plugin = VeronicaGovernancePlugin()
    tool = _make_tool("search")
    ctx = _make_callback_context()
    await plugin.before_tool_callback(tool=tool, tool_args={}, tool_context=ctx)
    assert len(plugin._tool_start_times) == 1
    await plugin.on_tool_error_callback(
        tool=tool,
        tool_args={},
        tool_context=ctx,
        error=RuntimeError("fail"),
    )
    assert len(plugin._tool_start_times) == 0


class TestZeroBudgetConfig:

  def test_zero_budget_rejected_by_config(self):
    """GovernanceConfig should reject zero budget."""
    with pytest.raises(Exception):
      GovernanceConfig(max_cost_usd=0.0)
