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

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ApprovalStatus:
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    EXPIRED = "expired"


class RiskLevel:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ApprovalRequest(BaseModel):
    # Identity
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # ADK context — needed to resume the agent correctly
    session_id: str
    invocation_id: Optional[str] = None  # Required for ADK Resume feature
    function_call_id: Optional[str] = None  # Must match in FunctionResponse
    app_name: str
    user_id: str

    # Agent context — what the human needs to decide
    agent_name: str
    tool_name: str
    message: str  # Maps from ADK 1.x 'hint' OR ADK 2.0 'message'
    payload: dict  # The structured data awaiting approval
    response_schema: dict = Field(
        default_factory=dict
    )  # Empty in 1.x, populated in ADK 2.0
    risk_level: str = RiskLevel.MEDIUM

    # Status tracking
    status: str = ApprovalStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    decided_at: Optional[datetime] = None
    decided_by: Optional[str] = None
    decision_notes: Optional[str] = None

    # Escalation
    escalated_to: Optional[str] = None


class ApprovalDecision(BaseModel):
    decision: str  # approved / rejected / escalated
    reviewer_id: str
    notes: Optional[str] = None
    payload: dict = Field(default_factory=dict)  # Response data back to the agent
    escalate_to: Optional[str] = None
