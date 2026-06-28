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

"""Tests for DelegationAuthPlugin."""

from __future__ import annotations

import json
import time
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from google.adk_community.plugins.delegation_auth_plugin import (
    CredentialVerifier,
    DelegationAuthPlugin,
    StructuralVerifier,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(name: str = "test_tool") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


def _make_context(credential: Optional[str] = None) -> MagicMock:
    ctx = MagicMock()
    ctx.state = {}
    if credential is not None:
        ctx.state["agent_credential"] = credential
    return ctx


def _valid_credential(
    agent_id: str = "agent-1",
    permissions: list[str] | None = None,
    expiry: float | None = None,
) -> str:
    data = {
        "agent_id": agent_id,
        "permissions": permissions or ["read_data"],
    }
    if expiry is not None:
        data["expiry"] = expiry
    return json.dumps(data)


# ---------------------------------------------------------------------------
# StructuralVerifier tests
# ---------------------------------------------------------------------------

class TestStructuralVerifier:
    def test_valid_credential(self):
        v = StructuralVerifier()
        result = v.verify(_valid_credential())
        assert result.valid
        assert result.agent_id == "agent-1"
        assert "read_data" in result.permissions

    def test_invalid_json(self):
        v = StructuralVerifier()
        result = v.verify("not json")
        assert not result.valid
        assert "not valid JSON" in result.reason

    def test_missing_agent_id(self):
        v = StructuralVerifier()
        result = v.verify(json.dumps({"permissions": ["read_data"]}))
        assert not result.valid
        assert "missing agent_id" in result.reason

    def test_expired_credential(self):
        v = StructuralVerifier()
        result = v.verify(_valid_credential(expiry=time.time() - 100))
        assert not result.valid
        assert "expired" in result.reason

    def test_future_expiry_is_valid(self):
        v = StructuralVerifier()
        result = v.verify(_valid_credential(expiry=time.time() + 3600))
        assert result.valid


# ---------------------------------------------------------------------------
# DelegationAuthPlugin tests
# ---------------------------------------------------------------------------

class TestDelegationAuthPlugin:
    @pytest.mark.asyncio
    async def test_no_credential_blocks(self):
        plugin = DelegationAuthPlugin()
        result = await plugin.before_tool_callback(
            tool=_make_tool(),
            tool_args={},
            tool_context=_make_context(credential=None),
        )
        assert result is not None
        assert result["error"] == "authorization_required"

    @pytest.mark.asyncio
    async def test_no_credential_fail_open(self):
        plugin = DelegationAuthPlugin(fail_open=True)
        result = await plugin.before_tool_callback(
            tool=_make_tool(),
            tool_args={},
            tool_context=_make_context(credential=None),
        )
        assert result is None  # allowed

    @pytest.mark.asyncio
    async def test_valid_credential_allows(self):
        plugin = DelegationAuthPlugin(
            required_permissions={"read_data"},
        )
        result = await plugin.before_tool_callback(
            tool=_make_tool(),
            tool_args={},
            tool_context=_make_context(
                credential=_valid_credential(permissions=["read_data"])
            ),
        )
        assert result is None  # allowed

    @pytest.mark.asyncio
    async def test_missing_permissions_blocks(self):
        plugin = DelegationAuthPlugin(
            required_permissions={"read_data", "financial_small"},
        )
        result = await plugin.before_tool_callback(
            tool=_make_tool(),
            tool_args={},
            tool_context=_make_context(
                credential=_valid_credential(permissions=["read_data"])
            ),
        )
        assert result is not None
        assert result["error"] == "insufficient_permissions"

    @pytest.mark.asyncio
    async def test_per_tool_permissions(self):
        plugin = DelegationAuthPlugin(
            required_permissions={"read_data"},
            tool_permissions={"pay_tool": {"read_data", "financial_small"}},
        )
        # Default tool — read_data is enough
        r1 = await plugin.before_tool_callback(
            tool=_make_tool("query_tool"),
            tool_args={},
            tool_context=_make_context(
                credential=_valid_credential(permissions=["read_data"])
            ),
        )
        assert r1 is None

        # pay_tool — needs financial_small too
        r2 = await plugin.before_tool_callback(
            tool=_make_tool("pay_tool"),
            tool_args={},
            tool_context=_make_context(
                credential=_valid_credential(permissions=["read_data"])
            ),
        )
        assert r2 is not None
        assert r2["error"] == "insufficient_permissions"

    @pytest.mark.asyncio
    async def test_invalid_credential_blocks(self):
        plugin = DelegationAuthPlugin()
        result = await plugin.before_tool_callback(
            tool=_make_tool(),
            tool_args={},
            tool_context=_make_context(credential="not-json"),
        )
        assert result is not None
        assert result["error"] == "authorization_denied"

    @pytest.mark.asyncio
    async def test_verifier_exception_returns_error(self):
        class ExplodingVerifier(CredentialVerifier):
            def verify(self, credential: str) -> VerificationResult:
                raise RuntimeError("boom")

        plugin = DelegationAuthPlugin(verifier=ExplodingVerifier())
        result = await plugin.before_tool_callback(
            tool=_make_tool(),
            tool_args={},
            tool_context=_make_context(credential="anything"),
        )
        assert result is not None
        assert result["error"] == "verification_failed"

    @pytest.mark.asyncio
    async def test_audit_log_records(self):
        plugin = DelegationAuthPlugin(
            required_permissions={"read_data"},
        )
        await plugin.before_tool_callback(
            tool=_make_tool(),
            tool_args={},
            tool_context=_make_context(
                credential=_valid_credential(permissions=["read_data"])
            ),
        )
        assert len(plugin.audit_log) == 1
        assert plugin.audit_log[0]["action"] == "allow"

    @pytest.mark.asyncio
    async def test_custom_credential_key(self):
        plugin = DelegationAuthPlugin(credential_key="my_cred")
        ctx = _make_context()
        ctx.state["my_cred"] = _valid_credential(permissions=["read_data"])
        result = await plugin.before_tool_callback(
            tool=_make_tool(),
            tool_args={},
            tool_context=ctx,
        )
        assert result is None
