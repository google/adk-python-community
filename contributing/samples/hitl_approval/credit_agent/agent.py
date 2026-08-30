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

"""Credit agent — external supervisor HITL demo.

This agent demonstrates the cross-user approval pattern:
  - Customer chats in ADK web (:8080)
  - Agent wants to apply a credit → submits request to HITL API (:8000)
  - Agent blocks (non-blocking async poll) waiting for a decision
  - Supervisor opens Streamlit dashboard (:8501), reviews and approves/rejects
  - Agent resumes and informs the customer of the outcome

Make sure all three services are running before chatting (see start_servers.sh):
  HITL API:   uvicorn google.adk_community.services.hitl_approval.api:app --port 8000
  Dashboard:  streamlit run dashboard/app.py --server.headless true
  ADK web:    adk web credit_agent/ --port 8080
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from google.adk_community.tools.hitl.gateway import hitl_tool


@hitl_tool(agent_name="credit_agent")
async def apply_account_credit(account_id: str, amount: float, reason: str) -> dict:
    """Apply a credit to a customer account. Requires supervisor approval.

    Args:
        account_id: The customer account ID to credit.
        amount: Credit amount in USD.
        reason: Business justification for the credit.

    Returns:
        Confirmation with the updated account balance.
    """
    # Real implementation would call your billing/CRM API here
    return {
        "status": "credited",
        "account_id": account_id,
        "amount_credited": amount,
        "new_balance": f"${amount:.2f} credit applied successfully.",
    }


root_agent = Agent(
    name="credit_agent",
    model="gemini-2.5-flash",
    description=(
        "Customer support agent that can apply account credits. "
        "Every credit requires supervisor approval via the HITL dashboard."
    ),
    instruction=(
        "You are a customer support agent. When a customer requests an account credit, "
        "call apply_account_credit with their account ID, the amount, and the reason. "
        "Let them know their request is being reviewed by a supervisor and that you will "
        "update them once a decision is made. "
        "If the credit is approved, confirm it to the customer. "
        "If rejected, apologise and explain that the supervisor did not approve it."
    ),
    tools=[FunctionTool(apply_account_credit)],
)
