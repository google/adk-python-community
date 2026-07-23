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

"""ADK plugin for authority-posture routing (ADVISE / EXECUTE / DEFER / STOP).

Tool schemas and permissions govern *access* — which tools exist and who may
call them. This plugin governs *authorization* — whether the agent was allowed
to act at all before any tool runs. That is the layer where "the user asked for
advice and the agent executed" and "one bounded edit became a dependent
workflow" failures actually live.

Every proposed tool call is classified into one of four postures:

======== =================================================================
Posture  Meaning
======== =================================================================
ADVISE   Produce recommendations only; no tool side effects are authorized.
EXECUTE  Act, within an explicitly bounded scope.
DEFER    Pause and request authorization when scope or approval is unclear.
STOP     Refuse or block when the requested work is not allowed.
======== =================================================================

Two stages decide the final posture:

1. A **posture router** (optional, caller-supplied ``Callable``) — typically a
   model call — that classifies the request into a posture with a scope
   statement. Its verdict is *advisory*.
2. A **deterministic guard** — an irreversibility keyword tripwire plus an
   approval-state check — that composes with the router under
   **most-restrictive-wins**. Plain code can only make the posture *stricter*,
   never looser: if the router says EXECUTE but the guard trips, the floor is
   DEFER; if the router says STOP and the guard finds nothing, it stays STOP.

The design is **fail-closed**: an unparseable or raised router verdict defaults
to DEFER, because an unreadable authorization is not an authorization. With no
router configured, the plugin runs guard-only — a fully deterministic,
dependency-free authorization layer that DEFERs irreversible or approval-pending
tool calls and lets everything else EXECUTE.

Requires: no extra dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import enum
import logging
from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Union

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger("google_adk_community." + __name__)


class Posture(str, enum.Enum):
  """The four authority postures, ordered loosest to strictest."""

  ADVISE = "ADVISE"
  EXECUTE = "EXECUTE"
  DEFER = "DEFER"
  STOP = "STOP"


# Strictness ladder: a guard may only move a posture UP this ladder, never down.
# EXECUTE is the *loosest* (act freely within scope); STOP is the strictest.
_STRICTNESS: dict[Posture, int] = {
    Posture.EXECUTE: 0,
    Posture.ADVISE: 1,
    Posture.DEFER: 2,
    Posture.STOP: 3,
}


def stricter(a: Posture, b: Posture) -> Posture:
  """Most-restrictive-wins composition: returns the stricter of two postures."""
  return a if _STRICTNESS[a] >= _STRICTNESS[b] else b


@dataclass
class PostureVerdict:
  """A validated authority decision.

  Attributes:
    posture: The chosen posture.
    scope: For EXECUTE, the explicit boundary of the authorized action;
      otherwise a short statement of why this posture was chosen. An EXECUTE
      verdict without a scope is how bounded edits balloon into workflows.
    rationale: Free-text explanation, useful for audit trails.
  """

  posture: Posture
  scope: str = ""
  rationale: str = ""


@dataclass
class AuthorityRequest:
  """The unit a posture router is asked to authorize: one proposed tool call.

  Attributes:
    action: The tool name being proposed.
    tool_args: The arguments the agent wants to call the tool with.
    context: Standing context (pending approvals, prior refusals, scope of the
      current task) pulled from session state. The guard scans this for
      approval-state markers.
  """

  action: str
  tool_args: dict[str, Any]
  context: str


# A router maps a request to a posture. It may return a full PostureVerdict, a
# bare Posture, or a string naming a posture ("EXECUTE"); anything else is
# treated as unparseable and fails closed.
PostureRouter = Callable[
    [AuthorityRequest], Union[PostureVerdict, Posture, str]
]


# The tripwire is deliberately blunt. A false DEFER costs a clarifying question;
# a false EXECUTE can cost an irreversible action. Bias toward the question.
DEFAULT_IRREVERSIBLE_KEYWORDS: tuple[str, ...] = (
    "delete",
    "drop",
    "wipe",
    "truncate",
    "rm -",
    "overwrite",
    "purge",
    "send",
    "email",
    "publish",
    "post",
    "deploy",
    "merge",
    "push",
    "release",
    "pay",
    "refund",
    "charge",
    "transfer",
    "wire",
    "password",
    "secret",
    "token",
    "credential",
    "api key",
    ".env",
    "revoke",
    "uninstall",
    "reset",
)

DEFAULT_PENDING_APPROVAL_MARKERS: tuple[str, ...] = (
    "pending approval",
    "awaiting approval",
    "not yet approved",
    "approval required",
    "do not proceed",
    "don't do this yet",
    "hold off",
    "waiting for sign-off",
)


class AuthorityRoutingPlugin(BasePlugin):
  """ADK plugin that enforces an authority posture before tool execution.

  On every tool call the plugin computes a final posture by composing an
  optional caller-supplied posture router with a deterministic guard under
  most-restrictive-wins. When the final posture is anything other than EXECUTE,
  the call is short-circuited with a structured block response (per the ADK
  plugin contract); an EXECUTE posture returns ``None`` and the tool proceeds.

  Args:
    posture_router: Optional callable classifying an ``AuthorityRequest`` into a
      posture (typically a model call). May return a ``PostureVerdict``, a
      ``Posture``, or a posture name as a string. If omitted, the plugin runs
      guard-only and every non-tripped call is treated as EXECUTE.
    context_state_key: Session-state key whose string value is treated as the
      standing context scanned for approval-state markers. Defaults to
      ``"authority_context"``.
    irreversible_keywords: Override the tripwire keyword list. The tool name and
      argument values are scanned (case-insensitively) for these substrings; a
      hit forces the floor to DEFER.
    pending_approval_markers: Override the approval-state marker list scanned in
      the context string; a hit forces the floor to DEFER.
    fail_closed: When ``True`` (default), a router that raises or returns an
      unparseable verdict yields DEFER. When ``False``, such a router is ignored
      (posture EXECUTE) and a warning is logged.
    name: Plugin name registered with ADK. Defaults to ``"authority_routing"``.

  Example::

      from google.adk_community.plugins import AuthorityRoutingPlugin

      # Guard-only: no model, fully deterministic.
      plugin = AuthorityRoutingPlugin()
      runner = Runner(agent=my_agent, plugins=[plugin], ...)

      # With a model-backed router:
      def route(req):
          verdict = my_model_call(req.action, req.tool_args, req.context)
          return PostureVerdict(posture=..., scope=...)

      plugin = AuthorityRoutingPlugin(posture_router=route)
  """

  def __init__(
      self,
      *,
      posture_router: Optional[PostureRouter] = None,
      context_state_key: str = "authority_context",
      irreversible_keywords: Optional[Sequence[str]] = None,
      pending_approval_markers: Optional[Sequence[str]] = None,
      fail_closed: bool = True,
      name: str = "authority_routing",
  ) -> None:
    super().__init__(name=name)
    self._posture_router = posture_router
    self._context_state_key = context_state_key
    self._irreversible_keywords = tuple(
        kw.lower()
        for kw in (
            irreversible_keywords
            if irreversible_keywords is not None
            else DEFAULT_IRREVERSIBLE_KEYWORDS
        )
    )
    self._pending_markers = tuple(
        m.lower()
        for m in (
            pending_approval_markers
            if pending_approval_markers is not None
            else DEFAULT_PENDING_APPROVAL_MARKERS
        )
    )
    self._fail_closed = fail_closed

  # -- verdict coercion --------------------------------------------------------

  def _coerce_verdict(self, raw: Any) -> Optional[PostureVerdict]:
    """Normalize a router return value into a PostureVerdict, or None if bad."""
    if isinstance(raw, PostureVerdict):
      if isinstance(raw.posture, Posture):
        return raw
      # A PostureVerdict carrying a stringy posture — coerce the inner value.
      coerced = self._coerce_posture(raw.posture)
      if coerced is None:
        return None
      return PostureVerdict(
          posture=coerced, scope=raw.scope, rationale=raw.rationale
      )
    if isinstance(raw, Posture):
      return PostureVerdict(posture=raw)
    if isinstance(raw, str):
      coerced = self._coerce_posture(raw)
      if coerced is None:
        return None
      return PostureVerdict(posture=coerced)
    return None

  @staticmethod
  def _coerce_posture(value: Any) -> Optional[Posture]:
    if isinstance(value, Posture):
      return value
    if isinstance(value, str):
      try:
        return Posture(value.strip().upper())
      except ValueError:
        return None
    return None

  # -- deterministic guard -----------------------------------------------------

  def _deterministic_floor(
      self, action_probe: str, context: str
  ) -> tuple[Posture, str]:
    """Return the strictest posture the guard requires, with the reason.

    EXECUTE means "no floor raised". The guard never returns ADVISE or STOP;
    those are the router's judgment calls, not deterministic ones.
    """
    low_ctx = context.lower()
    for marker in self._pending_markers:
      if marker in low_ctx:
        return Posture.DEFER, f"approval-state marker in context: '{marker}'"
    for kw in self._irreversible_keywords:
      if kw in action_probe:
        return Posture.DEFER, f"irreversibility tripwire: '{kw}'"
    return Posture.EXECUTE, "no deterministic concerns"

  # -- ADK lifecycle -----------------------------------------------------------

  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[dict]:
    """Decide the authority posture for a tool call before it runs.

    Returns ``None`` to let the tool proceed (posture EXECUTE), or a structured
    block dict to short-circuit execution (posture ADVISE / DEFER / STOP).
    """
    context = self._read_context(tool_context)
    request = AuthorityRequest(
        action=tool.name, tool_args=tool_args, context=context
    )

    # 1. Router verdict (advisory). Fail closed on error / unparseable output.
    verdict = self._route(request)

    # 2. Deterministic guard floor.
    action_probe = self._action_probe(tool.name, tool_args)
    floor, floor_reason = self._deterministic_floor(action_probe, context)

    # 3. Compose: strictest wins.
    final = stricter(verdict.posture, floor)

    if final is Posture.EXECUTE:
      return None

    reason = self._explain(final, verdict, floor, floor_reason)
    logger.warning(
        "[%s] Blocked tool '%s': posture=%s (%s)",
        self.name,
        tool.name,
        final.value,
        reason,
    )
    return {
        "error": "authority_blocked",
        "posture": final.value,
        "reason": reason,
        "scope": verdict.scope,
    }

  # -- helpers -----------------------------------------------------------------

  def _read_context(self, tool_context: ToolContext) -> str:
    state = getattr(tool_context, "state", None)
    if state is None:
      return ""
    try:
      value = state.get(self._context_state_key, "")
    except Exception:  # pragma: no cover - defensive against exotic state objs
      return ""
    return value if isinstance(value, str) else str(value)

  @staticmethod
  def _action_probe(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Flatten the proposed call into one lowercase string for tripwire scans."""
    parts = [tool_name]
    parts.extend(str(v) for v in tool_args.values())
    return " ".join(parts).lower()

  def _route(self, request: AuthorityRequest) -> PostureVerdict:
    """Run the router with fail-closed handling; EXECUTE if no router."""
    if self._posture_router is None:
      return PostureVerdict(
          posture=Posture.EXECUTE, scope="", rationale="no router"
      )

    try:
      raw = self._posture_router(request)
    except Exception as exc:  # noqa: BLE001 - any router failure must fail closed
      return self._router_fallback(f"router raised: {exc!r}")

    verdict = self._coerce_verdict(raw)
    if verdict is None:
      return self._router_fallback(f"router output unparseable: {raw!r}")
    return verdict

  def _router_fallback(self, reason: str) -> PostureVerdict:
    if self._fail_closed:
      logger.warning("[%s] Failing closed to DEFER: %s", self.name, reason)
      return PostureVerdict(posture=Posture.DEFER, scope="", rationale=reason)
    logger.warning(
        "[%s] Router ignored (fail_closed=False), posture EXECUTE: %s",
        self.name,
        reason,
    )
    return PostureVerdict(posture=Posture.EXECUTE, scope="", rationale=reason)

  @staticmethod
  def _explain(
      final: Posture,
      verdict: PostureVerdict,
      floor: Posture,
      floor_reason: str,
  ) -> str:
    if final is floor and floor is not verdict.posture:
      # The guard raised the posture above the router's verdict.
      return f"deterministic guard: {floor_reason}"
    if verdict.rationale:
      return verdict.rationale
    if verdict.scope:
      return verdict.scope
    return {
        Posture.ADVISE: "request authorized advice only, not tool execution",
        Posture.DEFER: "authorization ambiguous or approval pending",
        Posture.STOP: "requested action is not permitted",
    }.get(final, floor_reason)
