# Copyright 2026 Microsoft Corporation
#
# Licensed under the MIT License.
"""
GovernancePlugin for Google ADK.

Enforces policy-as-code rules before tool execution, verifies agent
identity, and produces tamper-evident audit trails using the Agent
Governance Toolkit (https://github.com/microsoft/agent-governance-toolkit).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GovernancePlugin:
    """ADK plugin that enforces governance policies before tool execution.

    Usage::

        from governance_plugin import GovernancePlugin

        governance = GovernancePlugin(policy_dir="./policies")
        agent = Agent(name="my-agent", plugins=[governance])
    """

    def __init__(
        self,
        policy_dir: str | Path = "./policies",
        agent_did: str = "did:mesh:adk-agent",
        sponsor_email: str = "adk-operator@example.com",
        default_action: str = "allow",
    ) -> None:
        self._policy_dir = Path(policy_dir)
        self._agent_did = agent_did
        self._sponsor_email = sponsor_email
        self._default_action = default_action
        self._engine = None
        self._audit = None
        self._setup()

    def _setup(self) -> None:
        """Initialize AGT policy engine and audit service."""
        try:
            from agentmesh.governance.policy import PolicyEngine
            from agentmesh.services.audit import AuditService

            self._engine = PolicyEngine()
            self._audit = AuditService()

            if self._policy_dir.exists():
                for f in sorted(self._policy_dir.glob("*.yaml")):
                    try:
                        self._engine.load_yaml(f.read_text())
                        logger.info("Loaded policy: %s", f.name)
                    except Exception as exc:
                        logger.warning("Skipped %s: %s", f.name, exc)

            logger.info(
                "GovernancePlugin initialized (agent=%s, policies=%s)",
                self._agent_did,
                self._policy_dir,
            )
        except ImportError:
            logger.warning(
                "agentmesh-platform not installed. "
                "Install with: pip install agentmesh-platform"
            )

    def before_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate governance policy before a tool call.

        Returns:
            Dict with 'allowed' (bool), 'decision', 'reason', and
            'audit_entry_id'.
        """
        if self._engine is None:
            return {"allowed": True, "decision": "allow", "reason": "AGT not installed"}

        context = {
            "action": tool_name,
            "tool_args": args or {},
            **kwargs,
        }

        result = self._engine.evaluate(
            agent_did=self._agent_did,
            context=context,
        )

        if self._audit:
            entry = self._audit.log_policy_decision(
                agent_did=self._agent_did,
                action=tool_name,
                decision=result.action,
                policy_name=result.policy_name or "",
                data={"tool_args": args or {}, "reason": result.reason},
            )
            audit_id = entry.entry_id
        else:
            audit_id = None

        return {
            "allowed": result.allowed,
            "decision": result.action,
            "reason": result.reason,
            "matched_rule": result.matched_rule,
            "audit_entry_id": audit_id,
        }

    def get_audit_summary(self) -> dict[str, Any]:
        """Return audit service summary."""
        if self._audit:
            return self._audit.summary()
        return {"error": "Audit service not initialized"}