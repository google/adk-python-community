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

"""SidClaw governance service for Google ADK agents.

Provides policy evaluation, human-in-the-loop approval, and
tamper-proof audit trails for ADK agent tool calls via the
SidClaw platform (https://sidclaw.com).

Usage::

    from sidclaw import AsyncSidClaw
    from google.adk_community.governance import (
        SidClawGovernanceConfig,
        SidClawGovernanceService,
    )
    from google.adk.agents import Agent

    client = AsyncSidClaw(
        api_key="ai_...",
        base_url="https://api.sidclaw.com",
        agent_id="my-adk-agent",
    )
    governance = SidClawGovernanceService(client=client)

    agent = Agent(
        name="my-agent",
        model="gemini-2.5-flash",
        instruction="You are a helpful assistant.",
        tools=[search_docs, query_database],
        before_tool_callback=governance.before_tool_callback,
        after_tool_callback=governance.after_tool_callback,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel
from pydantic import Field

logger = logging.getLogger("google_adk." + __name__)

# SidClaw SDK imports are deferred to __init__ of the service
# to give a clear error if the package is not installed.
_SIDCLAW_IMPORT_ERROR: Optional[ImportError] = None
try:
  from sidclaw import AsyncSidClaw
except ImportError as e:
  _SIDCLAW_IMPORT_ERROR = e
  AsyncSidClaw = None  # type: ignore[assignment,misc]


class SidClawGovernanceConfig(BaseModel):
  """Configuration for SidClaw governance callbacks.

  Attributes:
    default_classification: Default data classification for tool
      calls when no per-tool override is set. One of ``public``,
      ``internal``, ``confidential``, ``restricted``.
    tool_classifications: Per-tool data classification overrides
      keyed by tool name.
    resource_scope: Resource scope sent to the SidClaw policy
      engine.
    wait_for_approval: If ``True``, the before-tool callback
      blocks until a human reviewer approves or denies the
      request.  If ``False``, tool calls requiring approval
      are rejected immediately.
    approval_timeout_seconds: Maximum seconds to wait for an
      approval decision when ``wait_for_approval`` is enabled.
    approval_poll_interval_seconds: Polling interval in seconds
      when waiting for an approval decision.
    state_key: Key used to store the SidClaw trace ID in
      ``tool_context.state`` between before/after callbacks.
  """

  default_classification: str = "internal"
  tool_classifications: Dict[str, str] = Field(default_factory=dict)
  resource_scope: str = "google_adk"
  wait_for_approval: bool = True
  fail_open: bool = False
  approval_timeout_seconds: float = 300.0
  approval_poll_interval_seconds: float = 2.0
  state_key: str = "_sidclaw_trace_id"


class SidClawGovernanceService:
  """Governance service that provides ADK agent tool callbacks.

  Creates ``before_tool_callback`` and ``after_tool_callback``
  functions that integrate with ADK's agent lifecycle.  The
  before-callback evaluates each tool call against SidClaw
  policies; the after-callback records the execution outcome
  for the audit trail.

  Example::

      governance = SidClawGovernanceService(client=my_client)

      agent = Agent(
          name="governed-agent",
          model="gemini-2.5-flash",
          before_tool_callback=governance.before_tool_callback,
          after_tool_callback=governance.after_tool_callback,
          tools=[my_tool],
      )
  """

  def __init__(
      self,
      client: "AsyncSidClaw",
      config: Optional[SidClawGovernanceConfig] = None,
  ) -> None:
    """Initialise the governance service.

    Args:
      client: An ``AsyncSidClaw`` instance configured with an
        API key and agent ID.
      config: Optional governance configuration.  Uses sensible
        defaults if not provided.

    Raises:
      ImportError: If the ``sidclaw`` package is not installed.
    """
    if _SIDCLAW_IMPORT_ERROR is not None:
      raise ImportError(
          "The sidclaw package is required for SidClaw "
          "governance. Install it with: pip install sidclaw"
      ) from _SIDCLAW_IMPORT_ERROR

    self._client = client
    self._config = config or SidClawGovernanceConfig()

  # ------------------------------------------------------------------
  # Before-tool callback
  # ------------------------------------------------------------------

  async def before_tool_callback(
      self,
      tool: Any,
      args: Dict[str, Any],
      tool_context: Any,
  ) -> Optional[Dict[str, Any]]:
    """Evaluate tool call against SidClaw policies.

    This callback is invoked by ADK before every tool
    execution.  It sends the tool name and arguments to the
    SidClaw policy engine and handles the decision:

    - **allow**: returns ``None`` so the tool executes.
    - **deny**: returns an error dict, blocking execution.
    - **approval_required**: waits for human approval (if
      configured) or returns an error dict.

    Args:
      tool: The ADK tool about to be invoked.
      args: Arguments that will be passed to the tool.
      tool_context: ADK tool context with state and session.

    Returns:
      ``None`` to proceed with tool execution, or a ``dict``
      with an ``error`` key to short-circuit.
    """
    tool_name = getattr(tool, "name", "unknown_tool")
    classification = self._config.tool_classifications.get(
        tool_name, self._config.default_classification
    )

    try:
      decision = await self._client.evaluate({
          "operation": tool_name,
          "target_integration": "google_adk",
          "resource_scope": self._config.resource_scope,
          "data_classification": classification,
          "context": {
              "tool_name": tool_name,
              "args": _safe_serialize(args),
          },
      })
    except Exception:
      logger.exception(
          "SidClaw policy evaluation failed for tool %s",
          tool_name,
      )
      if self._config.fail_open:
        return None
      return {
          "error": (
              "Governance evaluation failed — tool blocked. "
              "Set fail_open=True to allow tools when SidClaw "
              "is unreachable."
          ),
      }

    # Store trace ID for the after-tool callback.
    tool_context.state[self._config.state_key] = (
        decision.trace_id
    )

    if decision.decision == "allow":
      logger.debug(
          "Tool %s allowed by policy (trace=%s)",
          tool_name,
          decision.trace_id,
      )
      return None

    if decision.decision == "deny":
      logger.info(
          "Tool %s denied by policy: %s (trace=%s)",
          tool_name,
          decision.reason,
          decision.trace_id,
      )
      return {"error": f"Policy denied: {decision.reason}"}

    # approval_required
    if (
        not self._config.wait_for_approval
        or not decision.approval_request_id
    ):
      logger.info(
          "Tool %s requires approval (not waiting): %s",
          tool_name,
          decision.reason,
      )
      return {
          "error": (
              f"Approval required: {decision.reason}. "
              f"Approval ID: {decision.approval_request_id}"
          ),
      }

    # Wait for human approval.
    logger.info(
        "Tool %s waiting for approval (id=%s)",
        tool_name,
        decision.approval_request_id,
    )
    try:
      status = await self._client.wait_for_approval(
          decision.approval_request_id,
          options={
              "timeout": self._config.approval_timeout_seconds,
              "poll_interval": (
                  self._config.approval_poll_interval_seconds
              ),
          },
      )
    except Exception:
      logger.exception(
          "Approval wait failed for tool %s", tool_name
      )
      return {
          "error": (
              "Approval wait timed out or failed. "
              f"Approval ID: {decision.approval_request_id}"
          ),
      }

    if status.status == "approved":
      logger.info(
          "Tool %s approved by %s",
          tool_name,
          status.approver_name,
      )
      return None

    note = (
        f": {status.decision_note}" if status.decision_note
        else ""
    )
    logger.info(
        "Tool %s approval %s%s",
        tool_name,
        status.status,
        note,
    )
    return {"error": f"Approval {status.status}{note}"}

  # ------------------------------------------------------------------
  # After-tool callback
  # ------------------------------------------------------------------

  async def after_tool_callback(
      self,
      tool: Any,
      args: Dict[str, Any],
      tool_context: Any,
      tool_response: Any,
  ) -> None:
    """Record tool execution outcome in SidClaw audit trail.

    This callback is invoked by ADK after every successful
    tool execution.  It records the outcome so the trace is
    finalized with a ``success`` or ``error`` status.

    Args:
      tool: The ADK tool that was invoked.
      args: Arguments that were passed to the tool.
      tool_context: ADK tool context with state and session.
      tool_response: The value returned by the tool.

    Returns:
      ``None`` — the original tool response is used.
    """
    trace_id = tool_context.state.get(self._config.state_key)
    if not trace_id:
      return None

    tool_name = getattr(tool, "name", "unknown_tool")
    is_error = (
        isinstance(tool_response, dict)
        and "error" in tool_response
    )

    try:
      await self._client.record_outcome(
          trace_id,
          {
              "status": "error" if is_error else "success",
              "metadata": {
                  "tool_name": tool_name,
              },
          },
      )
      logger.debug(
          "Recorded outcome for tool %s (trace=%s)",
          tool_name,
          trace_id,
      )
    except Exception:
      logger.exception(
          "Failed to record outcome for tool %s (trace=%s)",
          tool_name,
          trace_id,
      )

    return None


def _safe_serialize(obj: Any) -> Any:
  """Convert args to a JSON-safe dict for the SidClaw context."""
  if isinstance(obj, dict):
    return {
        k: str(v) if not isinstance(v, (str, int, float, bool))
        else v
        for k, v in obj.items()
    }
  return {"raw": str(obj)}
