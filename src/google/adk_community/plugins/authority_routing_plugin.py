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

"""ADK plugin for authority-based tool routing (ADVISE / EXECUTE / DEFER / STOP).

Tool schemas define what an agent *can* reach and permission systems define
what calls are *allowed*, but a large class of real agent failures happens
inside those boundaries: the user asked for advice and the agent executed; the
user authorized one bounded edit and the agent expanded into a dependent
workflow; an approval was clearly required and the agent proceeded anyway.

This plugin classifies the *authority* of each user turn into one of four
postures and enforces it on every tool call:

* ``ADVISE``  - the user wants a recommendation; no side effects authorized.
* ``EXECUTE`` - action authorized within an explicit, bounded scope.
* ``DEFER``   - authorization is ambiguous or a required approval is missing.
* ``STOP``    - the work is disallowed by policy, safety, or a prior refusal.

It composes two layers under one rule -- **the strictest posture wins**:

1. A pluggable model **router** (:class:`PostureRouter`) classifies the turn.
   The router is best-effort and can be argued out of a posture; it can only
   ever be *tightened* by the guard, never loosened.
2. A deterministic **guard** (plain keyword code) that a model cannot be talked
   past. It raises an irreversibility / pending-approval floor the router
   output is composed against.

The design fails **closed**: an unparseable or errored routing decision, or a
tool call that runs before any turn was routed, denies side effects rather than
allowing them.

Ported from a production authority-routing pattern, also proposed to the
Anthropic cookbook (anthropics/claude-cookbooks#787). The model router there is
Anthropic-backed; here it is a provider-agnostic ABC so any model (Gemini via
``google-genai``, or otherwise) can back it.
"""

from __future__ import annotations

import abc
import asyncio
import inspect
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Sequence

from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

logger = logging.getLogger(__name__)


class Posture(str, Enum):
    """Authority posture for a user turn."""

    ADVISE = "ADVISE"
    EXECUTE = "EXECUTE"
    DEFER = "DEFER"
    STOP = "STOP"


# Strictness ladder: a guard may only move the posture UP this ladder, never
# down. EXECUTE is the only posture that authorizes side effects.
_STRICTNESS = {
    Posture.EXECUTE: 0,
    Posture.ADVISE: 1,
    Posture.DEFER: 2,
    Posture.STOP: 3,
}


def stricter(a: Posture, b: Posture) -> Posture:
    """Return the most-restrictive of two postures (most-restrictive-wins)."""
    return a if _STRICTNESS[a] >= _STRICTNESS[b] else b


@dataclass
class PostureVerdict:
    """A router's authority classification for a single user turn.

    Attributes:
        posture: The classified authority posture.
        scope: For ``EXECUTE``, the explicit boundary of the authorized action;
            otherwise a short reason for the chosen posture. Advisory only --
            it is surfaced to the agent and logged, not string-matched.
        rationale: Human-readable justification for the verdict.
    """

    posture: Posture
    scope: str = ""
    rationale: str = ""


# Default deterministic tripwires. Anything in a request that touches deletion,
# sending, payments, credentials, or deployment lowers the floor to DEFER even
# if the router said EXECUTE.
DEFAULT_IRREVERSIBLE_KEYWORDS: tuple[str, ...] = (
    "delete", "drop", "wipe", "truncate", "rm -", "overwrite", "purge",
    "send", "email", "publish", "post", "deploy", "merge", "push", "release",
    "pay", "refund", "charge", "transfer", "wire",
    "password", "secret", "token", "credential", "api key", ".env",
    "revoke", "uninstall", "reset",
)

# Markers in the standing context that signal an approval is not yet resolved.
DEFAULT_PENDING_APPROVAL_MARKERS: tuple[str, ...] = (
    "pending approval", "awaiting approval", "not yet approved",
    "approval required", "do not proceed", "don't do this yet", "hold off",
    "waiting for sign-off",
)


class PostureRouter(abc.ABC):
    """Pluggable authority router.

    Implement :meth:`route` with any model backend. Keep it best-effort: the
    plugin composes the router output against a deterministic guard that can
    only tighten it, and fails closed if :meth:`route` raises.
    """

    @abc.abstractmethod
    async def route(self, request: str, context: str) -> PostureVerdict:
        """Classify a user turn's authority.

        Args:
            request: The user's request text for this turn.
            context: The standing context (prior authorizations, approval
                state) the plugin was configured to read.

        Returns:
            A :class:`PostureVerdict`. Implementations that parse structured
            model output should fail closed (return ``DEFER``) on parse errors
            rather than raising, though a raised exception is also handled by
            the plugin as a closed failure.
        """


class CallableRouter(PostureRouter):
    """Adapt a plain callable into a :class:`PostureRouter`.

    The callable receives ``(request, context)`` and returns a
    :class:`PostureVerdict`. Sync callables run in a thread executor so they
    never block the event loop.
    """

    def __init__(
        self,
        fn: Callable[[str, str], PostureVerdict | Awaitable[PostureVerdict]],
    ) -> None:
        self._fn = fn

    async def route(self, request: str, context: str) -> PostureVerdict:
        result = self._fn(request, context)
        if inspect.isawaitable(result):
            return await result
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: result)


class AuthorityRoutingPlugin(BasePlugin):
    """Enforce a per-turn authority posture on every tool call.

    On each user message the plugin routes an authority posture (via the
    optional model router, tightened by the deterministic guard) and stores it
    in session state. Before every tool call it enforces that posture: only
    ``EXECUTE`` authorizes side effects; ``ADVISE`` / ``DEFER`` / ``STOP``
    short-circuit the tool with a denial response (per the ADK plugin
    contract), unless the tool is on the read-only allowlist.

    Args:
        router: The authority router. If ``None``, the plugin runs in
            guard-only mode: the base posture is ``EXECUTE`` and only the
            deterministic floor can tighten it. A router is strongly
            recommended for anything beyond keyword tripwires.
        standing_context_key: Session-state key the plugin reads the standing
            context from (prior authorizations, approval state). Missing or
            non-string values are treated as empty context.
        posture_state_key: Session-state key the plugin writes the routed
            decision to and reads it back from at tool time.
        read_only_tool_names: Tool names that gather information without side
            effects. These remain permitted under ``ADVISE`` and ``DEFER`` (an
            agent may still read to advise or to decide what to ask), but are
            blocked under ``STOP``.
        extra_irreversible_keywords: Additional request substrings that lower
            the deterministic floor to ``DEFER``.
        pending_approval_markers: Override the default context markers that
            force ``DEFER``. ``None`` keeps the defaults.
        fail_closed: If ``True`` (default), a routing error or a tool call that
            runs before any turn was routed denies side effects. If ``False``,
            such cases fall back to the guard-only base (``EXECUTE``).
        audit_log_size: Max entries kept in the in-memory audit log (FIFO).

    Example::

        from google.adk_community.plugins import (
            AuthorityRoutingPlugin, CallableRouter, PostureVerdict, Posture,
        )

        def my_router(request, context):
            # call Gemini (or any model) and parse a PostureVerdict
            ...
            return PostureVerdict(posture=Posture.ADVISE, scope="", rationale="...")

        plugin = AuthorityRoutingPlugin(
            router=CallableRouter(my_router),
            read_only_tool_names={"read_file", "list_dir"},
        )
        runner = Runner(agent=my_agent, plugins=[plugin], ...)
    """

    def __init__(
        self,
        router: Optional[PostureRouter] = None,
        *,
        standing_context_key: str = "authority:standing_context",
        posture_state_key: str = "authority:posture",
        read_only_tool_names: Optional[Sequence[str]] = None,
        extra_irreversible_keywords: Optional[Sequence[str]] = None,
        pending_approval_markers: Optional[Sequence[str]] = None,
        fail_closed: bool = True,
        audit_log_size: int = 1000,
    ) -> None:
        super().__init__(name="authority_routing")
        self._router = router
        self._standing_context_key = standing_context_key
        self._posture_state_key = posture_state_key
        self._read_only = set(read_only_tool_names or ())
        self._irreversible = tuple(DEFAULT_IRREVERSIBLE_KEYWORDS) + tuple(
            extra_irreversible_keywords or ()
        )
        self._pending_markers = tuple(
            DEFAULT_PENDING_APPROVAL_MARKERS
            if pending_approval_markers is None
            else pending_approval_markers
        )
        self._fail_closed = fail_closed
        self._audit: deque[dict[str, Any]] = deque(maxlen=audit_log_size)

    # -- deterministic guard -------------------------------------------------

    def deterministic_floor(self, request: str, context: str) -> tuple[Posture, str]:
        """Return the strictest posture the guard requires, with the reason.

        Pending-approval markers in the context and irreversibility keywords in
        the request each raise the floor to ``DEFER``. With no concern found the
        floor is ``EXECUTE`` (no floor raised).
        """
        low_req, low_ctx = request.lower(), context.lower()
        for marker in self._pending_markers:
            if marker in low_ctx:
                return Posture.DEFER, f"approval-state marker in context: '{marker}'"
        for kw in self._irreversible:
            if kw in low_req:
                return Posture.DEFER, f"irreversibility tripwire: '{kw}'"
        return Posture.EXECUTE, "no deterministic concerns"

    # -- routing on each user turn ------------------------------------------

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> Optional[types.Content]:
        """Route and store the authority posture for this turn."""
        request = _extract_text(user_message)
        state = invocation_context.session.state
        raw_ctx = state.get(self._standing_context_key, "")
        context = raw_ctx if isinstance(raw_ctx, str) else ""

        # Base posture from the router (best-effort), or EXECUTE in guard-only
        # mode. A raised router is a closed failure.
        if self._router is None:
            verdict = PostureVerdict(
                posture=Posture.EXECUTE,
                scope="",
                rationale="guard-only mode (no router configured)",
            )
        else:
            try:
                verdict = await self._router.route(request, context)
            except Exception as exc:  # noqa: BLE001 - router is untrusted
                floor_posture = Posture.DEFER if self._fail_closed else Posture.EXECUTE
                verdict = PostureVerdict(
                    posture=floor_posture,
                    scope="",
                    rationale=f"router raised {type(exc).__name__}: {exc}",
                )
                logger.warning(
                    "Authority router raised; failing %s: %s",
                    "closed" if self._fail_closed else "open", exc,
                )

        floor, floor_reason = self.deterministic_floor(request, context)
        final = stricter(verdict.posture, floor)

        decision = {
            "posture": final.value,
            "scope": verdict.scope,
            "router_posture": verdict.posture.value,
            "router_rationale": verdict.rationale,
            "floor": floor.value,
            "floor_reason": floor_reason,
        }
        state[self._posture_state_key] = decision
        self._audit.append({"event": "route", **decision})
        logger.info(
            "Authority posture=%s (router=%s, floor=%s: %s)",
            final.value, verdict.posture.value, floor.value, floor_reason,
        )
        return None

    # -- enforcement before each tool call ----------------------------------

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict]:
        """Allow the tool under ``EXECUTE``; otherwise short-circuit it."""
        decision = tool_context.state.get(self._posture_state_key)

        if not isinstance(decision, dict) or "posture" not in decision:
            # No turn was routed (e.g. tool fired before any user message was
            # classified). Fail closed by default.
            if self._fail_closed:
                return self._deny(
                    tool, posture=Posture.DEFER.value, scope="",
                    reason="no authority decision on record for this turn",
                )
            return None

        posture = decision["posture"]
        scope = decision.get("scope", "")

        if posture == Posture.EXECUTE.value:
            return None

        if posture in (Posture.ADVISE.value, Posture.DEFER.value) and tool.name in self._read_only:
            self._audit.append(
                {"event": "allow_read_only", "tool": tool.name, "posture": posture}
            )
            return None

        return self._deny(
            tool, posture=posture, scope=scope,
            reason=decision.get("floor_reason") or decision.get("router_rationale", ""),
        )

    def _deny(self, tool: BaseTool, *, posture: str, scope: str, reason: str) -> dict:
        self._audit.append(
            {"event": "deny", "tool": tool.name, "posture": posture, "reason": reason}
        )
        logger.warning(
            "Authority denied tool '%s' under posture %s: %s",
            tool.name, posture, reason,
        )
        return {
            "error": "authority_denied",
            "posture": posture,
            "scope": scope,
            "reason": reason,
            "message": (
                f"Tool '{tool.name}' was not executed: the authority router "
                f"classified this turn as {posture}, which does not authorize "
                f"side-effecting actions. Only EXECUTE authorizes tool side effects."
            ),
        }

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        """A copy of the in-memory audit log (oldest first)."""
        return list(self._audit)


def _extract_text(content: Optional[types.Content]) -> str:
    """Join the text parts of a Content into a single string."""
    if content is None or not getattr(content, "parts", None):
        return ""
    texts = [p.text for p in content.parts if getattr(p, "text", None)]
    return "\n".join(texts)
