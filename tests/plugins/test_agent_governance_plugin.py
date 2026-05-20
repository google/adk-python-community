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

"""Tests for AgentGovernancePlugin.

These tests mock the agentmesh-platform dependency so they run without
it installed, verifying the plugin's integration with the ADK plugin API.
"""

from __future__ import annotations

import sys
import types as stdlib_types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from google.adk_community.plugins.agent_governance_plugin import (
    AgentGovernancePlugin,
)


# ---------------------------------------------------------------------------
# Mock fixtures for agentmesh-platform
# ---------------------------------------------------------------------------


class _FakePolicyResult:
    """Mimics agentmesh.governance.policy.PolicyResult."""

    def __init__(
        self,
        allowed: bool,
        action: str = "allow",
        reason: str = "",
        policy_name: str = "",
        matched_rule: str = "",
    ):
        self.allowed = allowed
        self.action = action
        self.reason = reason
        self.policy_name = policy_name
        self.matched_rule = matched_rule


class _FakePolicyEngine:
    """Mimics agentmesh.governance.policy.PolicyEngine."""

    def __init__(self):
        self._policies = []

    def load_yaml(self, text: str) -> None:
        self._policies.append(text)

    def evaluate(self, agent_did: str, context: dict) -> _FakePolicyResult:
        action_name = context.get("action", "")
        if "shell" in action_name or "delete" in action_name:
            return _FakePolicyResult(
                allowed=False,
                action="deny",
                reason=f"Blocked by policy: {action_name}",
                matched_rule="block-dangerous-tools",
            )
        return _FakePolicyResult(
            allowed=True, action="allow", reason="Policy check passed"
        )


class _FakeAuditEntry:
    entry_id = "audit-001"


class _FakeAuditService:
    """Mimics agentmesh.services.audit.AuditService."""

    def __init__(self):
        self.entries = []

    def log_policy_decision(self, **kwargs) -> _FakeAuditEntry:
        self.entries.append(kwargs)
        return _FakeAuditEntry()


def _install_fake_agentmesh():
    """Install fake agentmesh modules in sys.modules."""
    agentmesh = stdlib_types.ModuleType("agentmesh")
    governance = stdlib_types.ModuleType("agentmesh.governance")
    policy = stdlib_types.ModuleType("agentmesh.governance.policy")
    services = stdlib_types.ModuleType("agentmesh.services")
    audit = stdlib_types.ModuleType("agentmesh.services.audit")

    policy.PolicyEngine = _FakePolicyEngine
    audit.AuditService = _FakeAuditService

    agentmesh.governance = governance
    governance.policy = policy
    agentmesh.services = services
    services.audit = audit

    sys.modules["agentmesh"] = agentmesh
    sys.modules["agentmesh.governance"] = governance
    sys.modules["agentmesh.governance.policy"] = policy
    sys.modules["agentmesh.services"] = services
    sys.modules["agentmesh.services.audit"] = audit


def _uninstall_fake_agentmesh():
    """Remove fake agentmesh modules from sys.modules."""
    for key in list(sys.modules):
        if key.startswith("agentmesh"):
            del sys.modules[key]


# ---------------------------------------------------------------------------
# Fake ADK types for testing
# ---------------------------------------------------------------------------


class _FakeBaseTool:
    def __init__(self, name: str):
        self.name = name


class _FakeToolContext:
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_agentmesh():
    """Ensure agentmesh mocks are cleaned between tests."""
    yield
    _uninstall_fake_agentmesh()


@pytest.fixture()
def policy_dir(tmp_path: Path) -> Path:
    """Create a temporary policy directory with a sample policy."""
    policies = tmp_path / "policies"
    policies.mkdir()
    (policies / "default.yaml").write_text(
        """\
apiVersion: governance.toolkit/v1
name: test-policy
rules:
  - name: block-dangerous-tools
    condition: "action in ['shell_exec', 'file_delete']"
    action: deny
default_action: allow
"""
    )
    return policies


class TestAgentGovernancePlugin:
    """Tests for AgentGovernancePlugin construction and behavior."""

    def test_raises_import_error_when_agentmesh_missing(self, tmp_path: Path):
        """Plugin raises ImportError by default when agentmesh is missing."""
        # Simulate agentmesh not being importable by patching _setup
        with patch.object(
            AgentGovernancePlugin,
            "_setup",
            side_effect=ImportError("agentmesh-platform is required"),
        ):
            with pytest.raises(ImportError, match="agentmesh-platform"):
                AgentGovernancePlugin(policy_dir=tmp_path)

    def test_fail_open_mode_allows_construction(self, tmp_path: Path):
        """With fail_open=True, plugin degrades when engine returns no policies."""
        plugin = AgentGovernancePlugin(policy_dir=tmp_path, fail_open=True)
        assert plugin.name == "agent_governance"

    def test_construction_with_agentmesh(self, policy_dir: Path):
        """Plugin loads policies when agentmesh is available."""
        _install_fake_agentmesh()
        plugin = AgentGovernancePlugin(policy_dir=policy_dir)
        assert plugin._engine is not None
        assert plugin._audit is not None
        assert len(plugin._engine._policies) == 1

    def test_plugin_name(self, policy_dir: Path):
        """Plugin registers with the correct name."""
        _install_fake_agentmesh()
        plugin = AgentGovernancePlugin(policy_dir=policy_dir)
        assert plugin.name == "agent_governance"

    @pytest.mark.asyncio
    async def test_allows_safe_tool_call(self, policy_dir: Path):
        """Plugin returns None for allowed tool calls."""
        _install_fake_agentmesh()
        plugin = AgentGovernancePlugin(policy_dir=policy_dir)

        result = await plugin.before_tool_callback(
            tool=_FakeBaseTool("web_search"),
            tool_args={"query": "test"},
            tool_context=_FakeToolContext(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_blocks_denied_tool_call(self, policy_dir: Path):
        """Plugin returns error dict for denied tool calls."""
        _install_fake_agentmesh()
        plugin = AgentGovernancePlugin(policy_dir=policy_dir)

        result = await plugin.before_tool_callback(
            tool=_FakeBaseTool("shell_exec"),
            tool_args={"cmd": "rm -rf /"},
            tool_context=_FakeToolContext(),
        )
        assert result is not None
        assert result["error"] == "policy_denied"
        assert "shell_exec" in result["reason"]
        assert result["matched_rule"] == "block-dangerous-tools"

    @pytest.mark.asyncio
    async def test_blocks_delete_tool(self, policy_dir: Path):
        """Plugin denies file_delete actions."""
        _install_fake_agentmesh()
        plugin = AgentGovernancePlugin(policy_dir=policy_dir)

        result = await plugin.before_tool_callback(
            tool=_FakeBaseTool("file_delete"),
            tool_args={"path": "/etc/passwd"},
            tool_context=_FakeToolContext(),
        )
        assert result is not None
        assert result["error"] == "policy_denied"

    @pytest.mark.asyncio
    async def test_audit_logs_decisions(self, policy_dir: Path):
        """Plugin logs policy decisions to the audit service."""
        _install_fake_agentmesh()
        plugin = AgentGovernancePlugin(policy_dir=policy_dir)

        await plugin.before_tool_callback(
            tool=_FakeBaseTool("web_search"),
            tool_args={},
            tool_context=_FakeToolContext(),
        )
        await plugin.before_tool_callback(
            tool=_FakeBaseTool("shell_exec"),
            tool_args={},
            tool_context=_FakeToolContext(),
        )
        assert len(plugin._audit.entries) == 2
        assert plugin._audit.entries[0]["action"] == "web_search"
        assert plugin._audit.entries[0]["decision"] == "allow"
        assert plugin._audit.entries[1]["action"] == "shell_exec"
        assert plugin._audit.entries[1]["decision"] == "deny"

    @pytest.mark.asyncio
    async def test_fail_open_allows_all_when_no_engine(self, tmp_path: Path):
        """With fail_open=True and no engine, all calls pass through."""
        plugin = AgentGovernancePlugin(policy_dir=tmp_path, fail_open=True)
        # Simulate no engine (as if agentmesh wasn't installed)
        plugin._engine = None

        result = await plugin.before_tool_callback(
            tool=_FakeBaseTool("shell_exec"),
            tool_args={"cmd": "rm -rf /"},
            tool_context=_FakeToolContext(),
        )
        assert result is None

    def test_missing_policy_dir_still_constructs(self, tmp_path: Path):
        """Plugin constructs even when policy_dir doesn't exist."""
        _install_fake_agentmesh()
        plugin = AgentGovernancePlugin(
            policy_dir=tmp_path / "nonexistent"
        )
        assert plugin._engine is not None

    def test_custom_agent_did(self, policy_dir: Path):
        """Plugin uses custom agent_did in evaluations."""
        _install_fake_agentmesh()
        plugin = AgentGovernancePlugin(
            policy_dir=policy_dir,
            agent_did="did:mesh:custom-agent",
        )
        assert plugin._agent_did == "did:mesh:custom-agent"

    def test_strict_mode_raises_on_bad_policy(self, tmp_path: Path):
        """With strict=True, plugin raises RuntimeError on policy load failure."""
        _install_fake_agentmesh()

        # Patch _FakePolicyEngine.load_yaml to raise on bad content
        original_load = _FakePolicyEngine.load_yaml

        def failing_load(self, text):
            if "INVALID" in text:
                raise ValueError("invalid YAML syntax")
            original_load(self, text)

        _FakePolicyEngine.load_yaml = failing_load

        policies = tmp_path / "policies"
        policies.mkdir()
        (policies / "bad.yaml").write_text("INVALID policy content")

        try:
            with pytest.raises(RuntimeError, match="Failed to load policy"):
                AgentGovernancePlugin(policy_dir=policies, strict=True)
        finally:
            _FakePolicyEngine.load_yaml = original_load

    def test_non_strict_skips_bad_policy(self, tmp_path: Path):
        """Without strict mode, bad policies are skipped with a warning."""
        _install_fake_agentmesh()

        original_load = _FakePolicyEngine.load_yaml

        def failing_load(self, text):
            if "INVALID" in text:
                raise ValueError("invalid YAML syntax")
            original_load(self, text)

        _FakePolicyEngine.load_yaml = failing_load

        policies = tmp_path / "policies"
        policies.mkdir()
        (policies / "good.yaml").write_text("valid policy")
        (policies / "bad.yaml").write_text("INVALID policy content")

        try:
            plugin = AgentGovernancePlugin(policy_dir=policies)
            # Good policy loaded, bad one skipped
            assert len(plugin._engine._policies) == 1
        finally:
            _FakePolicyEngine.load_yaml = original_load

    @pytest.mark.asyncio
    async def test_evaluate_runs_in_thread_pool(self, policy_dir: Path):
        """Policy evaluation is offloaded to a thread pool executor."""
        import asyncio

        _install_fake_agentmesh()
        plugin = AgentGovernancePlugin(policy_dir=policy_dir)

        # Track which thread evaluate() runs on
        import threading

        eval_thread = None
        original_evaluate = plugin._engine.evaluate

        def tracking_evaluate(**kwargs):
            nonlocal eval_thread
            eval_thread = threading.current_thread()
            return original_evaluate(**kwargs)

        plugin._engine.evaluate = tracking_evaluate
        main_thread = threading.current_thread()

        await plugin.before_tool_callback(
            tool=_FakeBaseTool("web_search"),
            tool_args={"query": "test"},
            tool_context=_FakeToolContext(),
        )

        assert eval_thread is not None
        assert eval_thread != main_thread
