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

"""ADK plugin for Tulip admission-gate policy enforcement.

Evaluates a Tulip ``ControlPolicy`` before tool execution using Tulip's
admission gate (https://tulipagents.ai) — a policy check in real code,
outside the model, between a tool call and its side effect. Every decision
is appended to a tamper-evident, hash-chained ``AuditTrail`` regardless of
outcome, whether the call proceeds or not.

Unlike a binary allow/deny check, Tulip's policy has three outcomes: allow,
require a human, or deny. This plugin does not collapse ``require_human``
into either of the other two — it short-circuits the call with a distinct
``held_for_approval`` response, separate from ``policy_denied``, so the
caller can tell "this needs a person" apart from "this is refused" and
route accordingly. See the ``Scope`` note on
:class:`TulipGovernancePlugin` for what that does and does not do today.

Note on ``ControlPolicy``: ``require_verification_score`` defaults to 0.8
in Tulip, and ``approve()`` treats an action with no ``VerificationResult``
as failing that bar — "no verification provided" — which escalates to at
least ``require_human``. This plugin has no fact/evidence step to produce a
``VerificationResult`` for an arbitrary ADK tool call, so it never passes
one. A policy built for this plugin should generally set
``require_verification_score=0.0`` and drive allow/hold/deny purely off
``deny_for``/``require_human_for``/``max_blast_radius`` — otherwise every
call is held for a human by default, verification bar included or not.

Requires: ``pip install tulip-agents``
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)


class _TulipUnavailableError(ImportError):
  """Raised when tulip-agents is not installed and fail_open=False."""


class TulipGovernancePlugin(BasePlugin):
  """ADK plugin that gates tool calls behind Tulip's admission control.

  For every tool call, builds a Tulip ``Action`` describing its risk and
  weighs it against a ``ControlPolicy`` with Tulip's ``approve()`` — a
  policy check in real code, outside the model. The decision (allow /
  require a human / deny) is appended to a SHA-256 hash-chained
  ``AuditTrail`` either way, so a denied or held call is recorded exactly
  as durably as one that proceeded.

  Args:
      policy: The ``tulip.security.policy.ControlPolicy`` to enforce.
      action: Builds a ``tulip.control.Action`` describing a tool call's
          risk from the tool and its arguments — name, asset, blast
          radius, environment, kind, tags. Must be provided explicitly;
          there is no way to infer an action's blast radius or
          environment from a bare tool name.
      trail: An ``AuditTrail`` decisions are appended to. A fresh one is
          created if omitted; hand in a shared instance to accumulate
          decisions across plugins, tools, or runs. Available afterward
          as ``plugin.trail``.
      principal: Identifies the caller in each audit record.
      fail_open: If ``True``, tool calls proceed when ``tulip-agents`` is
          not installed (logs a warning). If ``False`` (default), raises
          ``ImportError`` at construction time.

  Raises:
      ImportError: If ``tulip-agents`` is not installed and ``fail_open``
          is False.

  Scope:
      A ``require_human`` decision short-circuits the call with a
      ``held_for_approval`` response — it does not pause the run and wait
      for a person to decide, then let the original call proceed. Wiring
      that resume path (e.g. through this repository's own
      ``tools/hitl`` gateway, or Tulip's own approval bridge) is left to
      a follow-up; today "held" means "not this turn," and the caller
      (or a human, out of band) decides what happens next.

      This plugin only ever sees tool calls — ``before_tool_callback``
      has no visibility into what the model *says* in a plain-text
      response. Harmful or misleading conversational content (no tool
      call involved) is structurally outside what any admission gate
      built on this hook can catch, regardless of how the ``action``
      callback is written. Measured directly against a real, independent,
      human-labeled benchmark (1,476 rows, Atarogic/ASSEBench,
      Apache-2.0): a reasonable ``action`` callback caught 88.7% of real
      labeled risks, and every miss traced to one of two causes — a gap
      in the callback's own risk heuristic (fixable by widening it), or a
      risk expressed as conversational text rather than a tool call (not
      fixable from inside this hook at all). See
      https://gist.github.com/fede-kamel/2ece9704d15978b9eb45e8c7dde5e8bc
      for the full methodology, including a first attempt at tightening
      the callback that *reduced* recall from 91.3% to 23.6% by defaulting
      ambiguous cases to allow instead of hold — kept in the writeup
      rather than erased, because it's the actual lesson: an ``action``
      callback for this plugin must default an ambiguous case to
      ``require_human``, never to allow. A gate that fails open on
      uncertainty is worse than no gate.

  Example::

      from google.adk.apps.app import App
      from google.adk.runners import Runner
      from google.adk_community.plugins import TulipGovernancePlugin
      from tulip.control import Action
      from tulip.security.policy import ControlPolicy

      plugin = TulipGovernancePlugin(
          # require_verification_score=0.0: see the "Note on ControlPolicy"
          # in this module's docstring before changing this.
          policy=ControlPolicy(
              require_verification_score=0.0,
              deny_for={"irreversible"},
              require_human_for={"production", "payment"},
          ),
          action=lambda tool, args: Action(
              name=tool.name,
              asset=str(args.get("order_id", "")),
              kind="payment",
              environment="production",
          ),
      )
      # plugins go on the App, not on Runner directly — Runner(plugins=...)
      # is deprecated.
      app = App(name="my_app", root_agent=my_agent, plugins=[plugin])
      runner = Runner(app=app, ...)
  """

  def __init__(
      self,
      *,
      policy: Any,
      action: Callable[[BaseTool, dict[str, Any]], Any],
      trail: Any | None = None,
      principal: str = "agent",
      fail_open: bool = False,
  ) -> None:
    super().__init__(name="tulip_governance")
    self._policy = policy
    self._build_action = action
    self._principal = principal
    self._approve = None
    self.trail = trail
    self._setup(fail_open=fail_open)

  def _setup(self, *, fail_open: bool) -> None:
    """Import tulip-agents and set up the trail. Lazy: this module must
    import cleanly without tulip-agents installed."""
    try:
      from tulip.security.audit import AuditTrail
      from tulip.security.policy import approve

      self._approve = approve
      if self.trail is None:
        self.trail = AuditTrail()
      logger.info("TulipGovernancePlugin initialized")
    except ImportError:
      if not fail_open:
        raise _TulipUnavailableError(
            "tulip-agents is required for governance enforcement. "
            "Install with: pip install tulip-agents"
        )
      logger.warning(
          "tulip-agents not installed; governance checks disabled. "
          "Install with: pip install tulip-agents"
      )

  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[dict]:
    """Evaluate the Tulip policy before a tool call.

    Returns ``None`` to allow the tool to proceed, or a dict response to
    short-circuit execution when the policy denies the call or holds it
    for a human.
    """
    if self._approve is None:
      return None

    action = self._build_action(tool, tool_args)
    decision = self._approve(action, policy=self._policy)

    # self.trail is always set by this point: _setup() only leaves
    # self._approve non-None (the guard above) after also setting self.trail,
    # so there is no reachable state here where it is still None.
    self.trail.record(
        "tool_admission",
        {
            "principal": self._principal,
            "tool": tool.name,
            "action": action.name,
            "asset": action.asset,
            "outcome": decision.outcome,
            "reason": decision.reason,
        },
    )

    if decision.outcome == "deny":
      logger.warning(
          "Tulip policy denied tool '%s': %s", tool.name, decision.reason
      )
      return {"error": "policy_denied", "reason": decision.reason}

    if decision.outcome == "require_human":
      logger.info(
          "Tulip policy held tool '%s' for a human: %s",
          tool.name,
          decision.reason,
      )
      return {"error": "held_for_approval", "reason": decision.reason}

    return None


__all__ = ["TulipGovernancePlugin"]
