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

"""HITL tool wrapper — submits an approval request to the FastAPI API and
waits asynchronously for a supervisor to approve/reject via the Streamlit
dashboard before executing the wrapped tool.

Usage:
    from google.adk_community.tools.hitl.gateway import hitl_tool
    from google.adk.tools import FunctionTool

    @hitl_tool(agent_name="credit_agent")
    async def apply_credit(account_id: str, amount: float) -> str:
        ...  # only runs after a supervisor approves in the dashboard

    tool = FunctionTool(apply_credit)
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import uuid
from typing import Any, Callable, Optional

import httpx

API_BASE_URL = "http://localhost:8000"
POLL_INTERVAL_S = 2.0
POLL_TIMEOUT_S = 300.0  # 5 minutes


def hitl_tool(
    agent_name: str,
    api_base: str = API_BASE_URL,
    poll_interval: float = POLL_INTERVAL_S,
    timeout: float = POLL_TIMEOUT_S,
):
    """Decorator — wraps any async or sync function with a supervisor approval gate.

    Flow:
      1. Agent calls the wrapped function.
      2. Wrapper POSTs an approval request to the HITL API (status: pending).
      3. Wrapper polls GET /approvals/{id} with asyncio.sleep — non-blocking.
      4. Supervisor opens the Streamlit dashboard and clicks Approve/Reject.
      5. On approval the original function runs; on rejection a PermissionError
         is raised so the agent can relay the outcome to the user.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs) -> Any:
            session_id = kwargs.pop("_session_id", str(uuid.uuid4()))
            invocation_id = kwargs.pop("_invocation_id", None)

            payload = {
                "session_id": session_id,
                "invocation_id": invocation_id,
                "app_name": "adk_chatbot",
                "user_id": "current_user",
                "agent_name": agent_name,
                "tool_name": fn.__name__,
                "message": f"Approval requested for {fn.__name__}",
                "payload": _serialise_args(fn, args, kwargs),
            }

            async with httpx.AsyncClient(base_url=api_base) as client:
                resp = await client.post("/approvals/", json=payload)
                resp.raise_for_status()
                request_id = resp.json()["id"]

            status = await _poll_for_decision(
                api_base, request_id, poll_interval, timeout
            )

            if status == "approved":
                if inspect.iscoroutinefunction(fn):
                    return await fn(*args, **kwargs)
                else:
                    return fn(*args, **kwargs)
            elif status == "rejected":
                raise PermissionError(
                    f"Tool '{fn.__name__}' was rejected by a supervisor."
                )
            elif status == "escalated":
                raise PermissionError(
                    f"Tool '{fn.__name__}' was escalated — awaiting further review."
                )
            else:
                raise TimeoutError(
                    f"No decision received for '{fn.__name__}' within {timeout}s."
                )

        return wrapper

    return decorator


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _poll_for_decision(
    api_base: str,
    request_id: str,
    interval: float,
    timeout: float,
) -> Optional[str]:
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient(base_url=api_base) as client:
        while asyncio.get_event_loop().time() < deadline:
            resp = await client.get(f"/approvals/{request_id}")
            resp.raise_for_status()
            data = resp.json()
            if data["status"] != "pending":
                return data["status"]
            await asyncio.sleep(interval)
    return None


def _serialise_args(fn: Callable, args: tuple, kwargs: dict) -> dict:
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    named = {params[i]: args[i] for i in range(len(args)) if i < len(params)}
    named.update(kwargs)
    return json.loads(json.dumps(named, default=str))
