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

"""VERONICA governance plugin for ADK.

Provides budget enforcement, circuit breaking, tool policy, and
degradation for multi-agent ADK workflows. Inspired by
veronica-core (https://github.com/amabito/veronica-core).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.genai import types
from pydantic import BaseModel
from pydantic import Field

from ._budget import BudgetTracker
from ._circuit_breaker import CircuitBreaker
from ._circuit_breaker import CircuitState
from ._degradation import DegradationManager
from ._policy import ToolPolicy

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[GOVERNANCE]"


class GovernanceConfig(BaseModel):
  """Configuration for VeronicaGovernancePlugin."""

  # Budget
  max_cost_usd: float = Field(
      default=10.0,
      gt=0,
      description="Org-level spending ceiling (USD). Must be positive.",
  )
  agent_max_cost_usd: float = Field(
      default=5.0,
      gt=0,
      description="Per-agent spending ceiling (USD). Must be positive.",
  )
  cost_per_1k_input_tokens: float = Field(
      default=0.00025,
      ge=0,
      description="Cost per 1,000 input tokens (USD).",
  )
  cost_per_1k_output_tokens: float = Field(
      default=0.0005,
      ge=0,
      description="Cost per 1,000 output tokens (USD).",
  )

  # Circuit Breaker
  failure_threshold: int = Field(
      default=5,
      ge=1,
      description="Consecutive failures before isolating an agent.",
  )
  recovery_timeout_s: float = Field(
      default=60.0,
      ge=0,
      description="Seconds before a tripped agent gets a probe request.",
  )

  # Tool Policy
  blocked_tools: list[str] = Field(
      default_factory=list,
      description="Tool names to block unconditionally.",
  )
  allowed_tools: Optional[list[str]] = Field(
      default=None,
      description="If set, only these tools are permitted.",
  )

  # Degradation
  degradation_threshold: float = Field(
      default=0.8,
      ge=0,
      le=1.0,
      description="Budget utilization ratio that triggers degradation.",
  )
  fallback_model: Optional[str] = Field(
      default=None,
      description="Model to switch to when degradation triggers.",
  )
  disable_tools_on_degrade: list[str] = Field(
      default_factory=list,
      description="Tools to disable when degradation triggers.",
  )


class VeronicaGovernancePlugin(BasePlugin):
  """Budget, circuit-breaker, and policy enforcement for ADK agents.

  Register this plugin on an ADK Runner (or App) to add runtime
  containment to any agent workflow. The plugin intercepts model and
  tool callbacks to enforce spending limits, block disallowed tools,
  isolate failing agents, and degrade to cheaper models
  when budget runs low.

  See: https://github.com/amabito/veronica-core
  """

  def __init__(
      self,
      config: Optional[GovernanceConfig] = None,
  ) -> None:
    super().__init__(name="veronica_governance")
    self._config = config or GovernanceConfig()

    self._budget = BudgetTracker(
        org_limit_usd=self._config.max_cost_usd,
        agent_limit_usd=self._config.agent_max_cost_usd,
        cost_per_1k_input_tokens=self._config.cost_per_1k_input_tokens,
        cost_per_1k_output_tokens=self._config.cost_per_1k_output_tokens,
    )
    self._circuit_breaker = CircuitBreaker(
        failure_threshold=self._config.failure_threshold,
        recovery_timeout_s=self._config.recovery_timeout_s,
    )
    self._policy = ToolPolicy(
        blocked_tools=self._config.blocked_tools,
        allowed_tools=self._config.allowed_tools,
    )
    self._degradation = DegradationManager(
        threshold=self._config.degradation_threshold,
        fallback_model=self._config.fallback_model,
        disable_tools_on_degrade=self._config.disable_tools_on_degrade,
    )

    self._tool_start_times: dict[str, float] = {}
    self._total_model_calls: int = 0
    self._total_tool_calls: int = 0
    self._run_start_time: float = time.monotonic()
    self._stats_lock = threading.Lock()
    self._degraded_agents: set[str] = set()

  # ------------------------------------------------------------------
  # Runner lifecycle
  # ------------------------------------------------------------------

  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[types.Content]:
    self._run_start_time = time.monotonic()
    logger.info(
        "%s Run started. Budget: $%.4f org / $%.4f per-agent.",
        _LOG_PREFIX,
        self._config.max_cost_usd,
        self._config.agent_max_cost_usd,
    )
    return None

  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    elapsed = time.monotonic() - self._run_start_time
    snap = self._budget.snapshot()
    cb_summary = self._circuit_breaker.summary()
    deg_events = self._degradation.events

    logger.info(
        "%s Run complete in %.1fs. Model calls: %d, Tool calls: %d.",
        _LOG_PREFIX,
        elapsed,
        self._total_model_calls,
        self._total_tool_calls,
    )
    logger.info(
        "%s Budget: $%.4f / $%.4f (%.1f%% used).",
        _LOG_PREFIX,
        snap.org_spent_usd,
        snap.org_limit_usd,
        snap.org_utilization * 100,
    )
    if snap.agent_spent:
      for agent, spent in snap.agent_spent.items():
        logger.info(
            "%s  Agent '%s': $%.4f / $%.4f.",
            _LOG_PREFIX,
            agent,
            spent,
            snap.agent_limit_usd,
        )
    if cb_summary:
      for agent, state in cb_summary.items():
        if state != "closed":
          logger.warning(
              "%s  Circuit breaker '%s': %s.",
              _LOG_PREFIX,
              agent,
              state,
          )
    if deg_events:
      logger.info("%s Degradation events (%d):", _LOG_PREFIX, len(deg_events))
      for ev in deg_events:
        logger.info(
            "%s  Agent '%s' at %.1f%% -- degraded %s -> %s.",
            _LOG_PREFIX,
            ev.agent_name,
            ev.utilization_pct,
            ev.original_model,
            ev.fallback_model,
        )

  # ------------------------------------------------------------------
  # Model callbacks
  # ------------------------------------------------------------------

  async def before_model_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
  ) -> Optional[LlmResponse]:
    agent_name = self._agent_name(callback_context)

    # Circuit breaker check -- block OPEN, allow one probe in HALF_OPEN.
    cb_state = self._circuit_breaker.get_state(agent_name)
    if cb_state == CircuitState.OPEN:
      logger.warning(
          "%s Agent '%s' circuit OPEN -- blocking model call.",
          _LOG_PREFIX,
          agent_name,
      )
      return LlmResponse(
          error_code="GOVERNANCE_CIRCUIT_OPEN",
          error_message=(
              f"Agent '{agent_name}' is isolated (circuit breaker open)."
          ),
      )
    if cb_state == CircuitState.HALF_OPEN:
      if not self._circuit_breaker.claim_probe(agent_name):
        return LlmResponse(
            error_code="GOVERNANCE_CIRCUIT_OPEN",
            error_message=f"Agent '{agent_name}' probe already in flight.",
        )
      logger.info(
          "%s Agent '%s' circuit HALF_OPEN -- allowing probe request.",
          _LOG_PREFIX,
          agent_name,
      )

    # Budget check
    allowed, reason = self._budget.check(agent_name)
    if not allowed:
      logger.warning(
          "%s Budget exceeded -- blocking model call. %s",
          _LOG_PREFIX,
          reason,
      )
      return LlmResponse(
          error_code="GOVERNANCE_BUDGET_EXCEEDED",
          error_message=reason,
      )

    # Degradation: switch model if near limit (once per agent).
    utilization = self._budget.utilization()
    if self._degradation.should_degrade(utilization):
      original = llm_request.model or "(default)"
      fallback = self._degradation.fallback_model
      if fallback:
        llm_request.model = fallback
        with self._stats_lock:
          is_first = agent_name not in self._degraded_agents
          if is_first:
            self._degraded_agents.add(agent_name)
        if is_first:
          event = self._degradation.record_event(
              agent_name=agent_name,
              utilization=utilization,
              original_model=original,
          )
          logger.info(
              "%s Budget at %.0f%% -- degraded to %s (agent '%s', was %s).",
              _LOG_PREFIX,
              event.utilization_pct,
              fallback,
              agent_name,
              original,
          )

    with self._stats_lock:
      self._total_model_calls += 1
    return None  # proceed with (possibly modified) request

  async def after_model_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_response: LlmResponse,
  ) -> Optional[LlmResponse]:
    agent_name = self._agent_name(callback_context)

    # Circuit breaker: soft errors (error_code set) count as failures.
    if llm_response.error_code:
      self._circuit_breaker.record_failure(agent_name)
    else:
      self._circuit_breaker.record_success(agent_name)

    # Record cost from usage metadata
    usage = llm_response.usage_metadata
    if usage is not None:
      input_tokens = getattr(usage, "prompt_token_count", 0) or 0
      output_tokens = getattr(usage, "candidates_token_count", 0) or 0
      cost = self._budget.estimate_cost(input_tokens, output_tokens)
      self._budget.record(agent_name, cost)

    return None

  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> Optional[LlmResponse]:
    agent_name = self._agent_name(callback_context)
    new_state = self._circuit_breaker.record_failure(agent_name)

    if new_state == CircuitState.OPEN:
      logger.warning(
          "%s Agent '%s' circuit tripped to OPEN"
          " after %d consecutive failures.",
          _LOG_PREFIX,
          agent_name,
          self._config.failure_threshold,
      )

    return None  # let the error propagate

  # ------------------------------------------------------------------
  # Tool callbacks
  # ------------------------------------------------------------------

  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: CallbackContext,
  ) -> Optional[dict]:
    # Policy check
    allowed, reason = self._policy.check(tool.name)
    if not allowed:
      logger.warning("%s Tool blocked: %s", _LOG_PREFIX, reason)
      return {"error": reason}

    # Degradation: disable expensive tools near budget limit
    utilization = self._budget.utilization()
    if self._degradation.should_disable_tool(tool.name, utilization):
      msg = f"Tool '{tool.name}' disabled -- budget at {utilization * 100:.0f}%"
      logger.info("%s %s", _LOG_PREFIX, msg)
      return {"error": msg}

    call_id = getattr(tool_context, "function_call_id", None) or ""
    timing_key = f"{tool.name}:{call_id}"
    with self._stats_lock:
      self._total_tool_calls += 1
      self._tool_start_times[timing_key] = time.monotonic()
    return None

  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: CallbackContext,
      result: dict,
  ) -> Optional[dict]:
    call_id = getattr(tool_context, "function_call_id", None) or ""
    timing_key = f"{tool.name}:{call_id}"
    with self._stats_lock:
      start = self._tool_start_times.pop(timing_key, None)
    if start is not None:
      elapsed_ms = (time.monotonic() - start) * 1000
      logger.debug(
          "%s Tool '%s' completed in %.1fms.",
          _LOG_PREFIX,
          tool.name,
          elapsed_ms,
      )
    return None

  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: CallbackContext,
      error: Exception,
  ) -> Optional[dict]:
    # Clean up timing entry that before_tool_callback recorded.
    call_id = getattr(tool_context, "function_call_id", None) or ""
    timing_key = f"{tool.name}:{call_id}"
    with self._stats_lock:
      self._tool_start_times.pop(timing_key, None)

    agent_name = self._agent_name(tool_context)
    new_state = self._circuit_breaker.record_failure(agent_name)

    if new_state == CircuitState.OPEN:
      logger.warning(
          "%s Agent '%s' circuit tripped to OPEN after tool error in '%s'.",
          _LOG_PREFIX,
          agent_name,
          tool.name,
      )

    return None  # let the error propagate

  # ------------------------------------------------------------------
  # Helpers
  # ------------------------------------------------------------------

  @staticmethod
  def _agent_name(callback_context: CallbackContext) -> str:
    """Extract agent name from callback context."""
    name = getattr(callback_context, "agent_name", None) or "unknown"
    # Sanitize to prevent log injection via newlines or control chars.
    return re.sub(r"[^\w\-.]", "_", name)

  @property
  def budget(self) -> BudgetTracker:
    """Access the budget tracker for inspection."""
    return self._budget

  @property
  def circuit_breaker(self) -> CircuitBreaker:
    """Access the circuit breaker for inspection."""
    return self._circuit_breaker
