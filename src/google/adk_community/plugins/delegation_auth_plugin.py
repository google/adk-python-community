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

"""ADK plugin for permission-scoped agent authorization.

Verifies that an agent holds a valid credential with sufficient permissions
before executing a tool. Uses a pluggable verifier interface so any identity
system (DID, JWT, ZKP, API keys, etc.) can be wired in.

A structural verifier is included for development; plug in a real verifier
for production.

Example::

    from google.adk_community.plugins import DelegationAuthPlugin

    plugin = DelegationAuthPlugin(
        required_permissions={"read_data", "financial_small"},
    )
    runner = Runner(agent=my_agent, plugins=[plugin], ...)
"""

from __future__ import annotations

import asyncio
import collections
import functools
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)

_MAX_AUDIT_ENTRIES_DEFAULT = 1000


# ---------------------------------------------------------------------------
# Pluggable verifier interface
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Result of a credential verification check."""

    valid: bool
    agent_id: str = ""
    permissions: set[str] = field(default_factory=set)
    expiry: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class CredentialVerifier(ABC):
    """Interface for agent credential verification.

    Implement this to plug in any identity system: DID verification,
    JWT validation, ZKP proof checking, API key lookup, etc.

    Note: ``verify()`` is called in a thread executor to avoid blocking
    the event loop. Implementations may perform synchronous I/O (network
    calls, file reads) safely.
    """

    @abstractmethod
    def verify(self, credential: str) -> VerificationResult:
        """Verify a credential string and return the result.

        Args:
            credential: The raw credential from the agent (typically from
                a header or session context).

        Returns:
            VerificationResult with valid=True if the credential checks out.
        """


class StructuralVerifier(CredentialVerifier):
    """Development verifier that checks credential structure only.

    Accepts any JSON-parseable credential with the required fields.
    NOT for production -- use a real verifier (DID, JWT, etc.).
    """

    def verify(self, credential: str) -> VerificationResult:
        try:
            data = json.loads(credential)
        except (json.JSONDecodeError, TypeError):
            return VerificationResult(
                valid=False, reason="credential is not valid JSON"
            )

        agent_id = data.get("agent_id", "")
        if not agent_id:
            return VerificationResult(
                valid=False, reason="missing agent_id field"
            )

        permissions = set(data.get("permissions", []))
        expiry = data.get("expiry", 0)

        if expiry and expiry < time.time():
            return VerificationResult(
                valid=False,
                agent_id=agent_id,
                reason="credential expired",
            )

        return VerificationResult(
            valid=True,
            agent_id=agent_id,
            permissions=permissions,
            expiry=expiry,
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class DelegationAuthPlugin(BasePlugin):
    """ADK plugin that enforces permission-scoped authorization.

    Before each tool call, the plugin:
    1. Reads the agent's credential from session state
    2. Verifies it using the configured verifier (in a thread executor)
    3. Checks that the credential's permissions cover the tool's requirements
    4. Blocks execution if any check fails

    Permission scoping: this plugin enforces permission checks at execution
    time. Delegation scope narrowing (ensuring child agents can only hold a
    subset of parent permissions) is enforced at credential issuance by the
    identity system, not by this plugin.

    Args:
        required_permissions: Default permissions required for any tool call.
            Individual tools can override via ``tool_permissions``.
        tool_permissions: Per-tool permission requirements. Keys are tool
            names, values are sets of required permission strings.
        verifier: Credential verifier instance. Defaults to
            ``StructuralVerifier`` (development only).
        credential_key: Session state key where the agent's credential
            is stored. Defaults to ``"agent_credential"``.
        fail_open: If True, allow tool calls when no credential is present.
            Defaults to False.
        max_audit_entries: Maximum audit log entries before FIFO eviction.
            Defaults to 1000.

    Example::

        plugin = DelegationAuthPlugin(
            required_permissions={"read_data"},
            tool_permissions={
                "execute_payment": {"read_data", "financial_small"},
                "sign_contract": {"read_data", "sign_on_behalf"},
            },
        )
    """

    def __init__(
        self,
        required_permissions: Optional[set[str]] = None,
        tool_permissions: Optional[dict[str, set[str]]] = None,
        verifier: Optional[CredentialVerifier] = None,
        credential_key: str = "agent_credential",
        fail_open: bool = False,
        max_audit_entries: int = _MAX_AUDIT_ENTRIES_DEFAULT,
    ) -> None:
        super().__init__(name="delegation_auth")
        self._required = required_permissions or {"read_data"}
        self._tool_permissions = tool_permissions or {}
        self._verifier = verifier or StructuralVerifier()
        self._credential_key = credential_key
        self._fail_open = fail_open
        self._audit_log: collections.deque[dict[str, Any]] = (
            collections.deque(maxlen=max_audit_entries)
        )

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict]:
        """Verify agent authorization before tool execution.

        Returns None to allow the tool to proceed, or a dict response
        to short-circuit execution when authorization fails.
        """
        # Read credential from session state
        credential = None
        if hasattr(tool_context, "state") and tool_context.state:
            credential = tool_context.state.get(self._credential_key)

        if not credential:
            if self._fail_open:
                logger.warning(
                    "No credential found for tool %s; fail_open=True, allowing",
                    tool.name,
                )
                return None
            self._log_denial(tool.name, "no_credential")
            return {
                "error": "authorization_required",
                "message": (
                    f"Tool '{tool.name}' requires agent authorization. "
                    f"Set '{self._credential_key}' in session state."
                ),
            }

        # Verify the credential in a thread executor to avoid blocking
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                functools.partial(self._verifier.verify, credential),
            )
        except Exception as exc:
            logger.error("Verifier raised for tool %s: %s", tool.name, exc)
            self._log_denial(tool.name, "verifier_error")
            return {
                "error": "verification_failed",
                "message": f"Credential verification error: {exc}",
            }

        if not result.valid:
            self._log_denial(
                tool.name,
                result.reason or "invalid_credential",
                agent_id=result.agent_id,
            )
            return {
                "error": "authorization_denied",
                "message": (
                    f"Agent '{result.agent_id}' denied access to "
                    f"'{tool.name}': {result.reason}"
                ),
            }

        # Check permissions
        required = self._tool_permissions.get(tool.name, self._required)
        missing = required - result.permissions
        if missing:
            self._log_denial(
                tool.name,
                f"missing_permissions: {sorted(missing)}",
                agent_id=result.agent_id,
            )
            return {
                "error": "insufficient_permissions",
                "message": (
                    f"Agent '{result.agent_id}' lacks permissions "
                    f"{sorted(missing)} for tool '{tool.name}'"
                ),
            }

        # Authorized
        logger.info(
            "Agent %s authorized for %s (permissions: %s)",
            result.agent_id,
            tool.name,
            sorted(result.permissions),
        )
        self._log_allow(tool.name, result.agent_id)
        return None

    def _log_allow(self, tool: str, agent_id: str) -> None:
        self._audit_log.append({
            "action": "allow",
            "tool": tool,
            "agent_id": agent_id,
            "timestamp": time.time(),
        })

    def _log_denial(
        self,
        tool: str,
        reason: str,
        agent_id: str = "",
    ) -> None:
        self._audit_log.append({
            "action": "deny",
            "tool": tool,
            "agent_id": agent_id,
            "reason": reason,
            "timestamp": time.time(),
        })
        logger.warning("DENIED: tool=%s agent=%s reason=%s", tool, agent_id, reason)

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        """Read-only copy of the audit trail."""
        return list(self._audit_log)
