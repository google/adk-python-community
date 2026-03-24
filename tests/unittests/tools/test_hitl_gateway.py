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

"""Unit tests for the HITL tool wrapper (mocking the API calls)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from google.adk_community.tools.hitl.gateway import _serialise_args, hitl_tool

# ── _serialise_args ───────────────────────────────────────────────────────────


def test_serialise_args_positional():
    def fn(a, b, c):
        ...

    result = _serialise_args(fn, (1, 2), {"c": 3})
    assert result == {"a": 1, "b": 2, "c": 3}


def test_serialise_args_kwargs_only():
    def fn(x, y):
        ...

    result = _serialise_args(fn, (), {"x": "hello", "y": 42})
    assert result == {"x": "hello", "y": 42}


def test_serialise_args_non_serialisable_falls_back_to_str():
    class Foo:
        pass

    def fn(obj):
        ...

    result = _serialise_args(fn, (Foo(),), {})
    assert isinstance(result["obj"], str)


# ── hitl_tool — approved ──────────────────────────────────────────────────────


def _make_mock_client(status: str, request_id: str = "abc-123"):
    mock_client = AsyncMock()

    # Setup context manager correctly
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False

    post_resp = MagicMock()
    post_resp.json.return_value = {"id": request_id}
    mock_client.post.return_value = post_resp

    get_resp = MagicMock()
    get_resp.json.return_value = {"id": request_id, "status": status}
    mock_client.get.return_value = get_resp

    return mock_client


@pytest.mark.asyncio
@patch("google.adk_community.tools.hitl.gateway.httpx.AsyncClient")
async def test_approved_tool_runs(mock_client_cls):
    mock_client_cls.return_value = _make_mock_client("approved")

    @hitl_tool(agent_name="test_agent")
    def add(a: int, b: int) -> int:
        return a + b

    result = await add(2, 3)
    assert result == 5


@pytest.mark.asyncio
@patch("google.adk_community.tools.hitl.gateway.httpx.AsyncClient")
async def test_rejected_tool_raises(mock_client_cls):
    mock_client_cls.return_value = _make_mock_client("rejected")

    @hitl_tool(agent_name="test_agent")
    def delete_file(path: str) -> str:
        return "deleted"

    with pytest.raises(PermissionError, match="rejected"):
        await delete_file("/important/file.txt")


@pytest.mark.asyncio
@patch("google.adk_community.tools.hitl.gateway.httpx.AsyncClient")
async def test_escalated_tool_raises(mock_client_cls):
    mock_client_cls.return_value = _make_mock_client("escalated")

    @hitl_tool(agent_name="test_agent")
    def wire_transfer(amount: float) -> str:
        return "done"

    with pytest.raises(PermissionError, match="escalated"):
        await wire_transfer(10000.0)
