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

"""ADK plugin for Agent Governance Toolkit policy enforcement.

Evaluates policy-as-code rules before tool execution using the Agent
Governance Toolkit (https://github.com/microsoft/agent-governance-toolkit).
Denied tool calls are short-circuited with a policy violation response.

Requires: ``pip install agentmesh-platform``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)


class _GovernanceUnavailableError(ImportError):
    """Raised when agentmesh-platform is not installed and fail_open=False."""


class AgentGovernancePlugin(BasePlugin):
    """ADK plugin that enforces governance policies before tool execution.

    Uses the Agent Governance Toolkit to evaluate YAML/OPA/Cedar policies.
    When a tool call is denied by policy, the plugin returns a dict response
    that short-circuits tool execution (per the ADK plugin contract).

    Args:
        policy_dir: Absolute or relative path to the directory containing
            ``*.yaml`` policy files. Resolved relative to the caller's
            working directory. Must be provided explicitly.
        agent_did: Decentralized identifier for the agent.
        fail_open: If ``True``, tool calls proceed when ``agentmesh-platform``
            is not installed (logs a warning). If ``False`` (default), raises
            ``ImportError`` at construction time.

    Raises:
        ImportError: If ``agentmesh-platform`` is not installed and
            ``fail_open`` is False.

    Example::

        from google.adk_community.plugins import AgentGovernancePlugin

        plugin = AgentGovernancePlugin(
            policy_dir=Path(__file__).parent / "policies",
        )
        runner = Runner(agent=my_agent, plugins=[plugin], ...)
    """

    def __init__(
        self,
        policy_dir: str | Path,
        agent_did: str = "did:mesh:adk-agent",
        fail_open: bool = False,
    ) -> None:
        super().__init__(name="agent_governance")
        self._policy_dir = Path(policy_dir).resolve()
        self._agent_did = agent_did
        self._engine = None
        self._audit = None
        self._setup(fail_open=fail_open)

    def _setup(self, *, fail_open: bool) -> None:
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
            else:
                logger.warning(
                    "Policy directory does not exist: %s", self._policy_dir
                )

            logger.info(
                "AgentGovernancePlugin initialized (agent=%s, policies=%s)",
                self._agent_did,
                self._policy_dir,
            )
        except ImportError:
            if not fail_open:
                raise _GovernanceUnavailableError(
                    "agentmesh-platform is required for governance enforcement. "
                    "Install with: pip install agentmesh-platform"
                )
            logger.warning(
                "agentmesh-platform not installed; governance checks disabled. "
                "Install with: pip install agentmesh-platform"
            )

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict]:
        """Evaluate governance policy before a tool call.

        Returns ``None`` to allow the tool to proceed, or a dict response
        to short-circuit execution when the policy denies the call.
        """
        if self._engine is None:
            return None

        context = {
            "action": tool.name,
            "tool_args": tool_args,
        }

        result = self._engine.evaluate(
            agent_did=self._agent_did,
            context=context,
        )

        if self._audit:
            self._audit.log_policy_decision(
                agent_did=self._agent_did,
                action=tool.name,
                decision=result.action,
                policy_name=result.policy_name or "",
                data={"tool_args": tool_args, "reason": result.reason},
            )

        if not result.allowed:
            logger.warning(
                "Policy denied tool '%s': %s (rule: %s)",
                tool.name,
                result.reason,
                result.matched_rule,
            )
            return {
                "error": "policy_denied",
                "reason": result.reason,
                "matched_rule": result.matched_rule,
            }

        return None
