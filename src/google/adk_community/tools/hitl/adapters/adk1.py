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

"""Adapter for Google ADK 1.x Human-in-the-Loop feature.

Converts ADK `adk_request_confirmation` events to normalized ApprovalRequests,
and formats Streamlit dashboard decisions back into ADK FunctionResponses.
"""

from __future__ import annotations

from typing import Any, Dict

import httpx

from ..models import (ApprovalDecision, ApprovalRequest, ApprovalStatus)


def parse_confirmation_event(payload: Dict[str, Any]) -> ApprovalRequest:
    """Parse an incoming ADK 1.x Tool Confirmation event to a normalized ApprovalRequest."""

    call_id = payload.get("function_call_id")
    args = payload.get("arguments", {})
    hint = args.get("hint", "Please review this action.")
    tool_payload = args.get("payload", {})

    return ApprovalRequest(
        session_id=payload.get("session_id", "unknown_session"),
        invocation_id=payload.get("invocation_id"),
        function_call_id=call_id,
        app_name=payload.get("app_name", "unknown_app"),
        user_id=payload.get("user_id", "unknown_user"),
        agent_name=payload.get("agent_name", "unknown_agent"),
        tool_name=args.get("tool_name", "unknown_tool"),
        message=hint,
        payload=tool_payload,
        response_schema={},  # Native tool confirmation in ADK 1.x doesn't expose a schema
    )


async def submit_decision_to_adk(
    adk_base_url: str, request: ApprovalRequest, decision: ApprovalDecision
):
    """Resume the ADK 1.x agent by sending the human's decision back as a FunctionResponse."""

    confirmed = decision.decision == ApprovalStatus.APPROVED

    adk_payload = {
        "app_name": request.app_name,
        "user_id": request.user_id,
        "session_id": request.session_id,
        "invocation_id": request.invocation_id,
        "new_message": {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "id": request.function_call_id,
                        "name": "adk_request_confirmation",
                        "response": {
                            "confirmed": confirmed,
                            "payload": decision.payload or {},
                        },
                    }
                }
            ],
        },
    }

    async with httpx.AsyncClient() as client:
        # Assumes the ADK FastAPI server is running with the /run_sse endpoint
        url = f"{adk_base_url.rstrip('/')}/run_sse"
        resp = await client.post(url, json=adk_payload)
        resp.raise_for_status()
        return resp.json()
