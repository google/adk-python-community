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

"""Approval request CRUD endpoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .store import ApprovalRequestDB, get_db
from ...tools.hitl.models import (ApprovalDecision, ApprovalRequest, ApprovalStatus)

router = APIRouter(prefix="/approvals", tags=["approvals"])

# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("/", response_model=ApprovalRequest, status_code=201)
async def create_approval(payload: ApprovalRequest, db: AsyncSession = Depends(get_db)):
    """Agent submits a new approval request before executing a tool."""
    db_item = ApprovalRequestDB(
        id=payload.id,
        session_id=payload.session_id,
        invocation_id=payload.invocation_id,
        function_call_id=payload.function_call_id,
        app_name=payload.app_name,
        user_id=payload.user_id,
        agent_name=payload.agent_name,
        tool_name=payload.tool_name,
        message=payload.message,
        payload=json.dumps(payload.payload),
        response_schema=json.dumps(payload.response_schema),
        risk_level=payload.risk_level,
        status=payload.status,
        created_at=payload.created_at,
        decided_at=payload.decided_at,
        decided_by=payload.decided_by,
        decision_notes=payload.decision_notes,
        escalated_to=payload.escalated_to,
    )
    db.add(db_item)
    await db.commit()
    await db.refresh(db_item)
    return _to_pydantic(db_item)


@router.get("/pending", response_model=List[ApprovalRequest])
async def list_pending_approvals(db: AsyncSession = Depends(get_db)):
    """List all pending approvals."""
    q = (
        select(ApprovalRequestDB)
        .where(ApprovalRequestDB.status == ApprovalStatus.PENDING)
        .order_by(ApprovalRequestDB.created_at.desc())
    )
    result = await db.execute(q)
    return [_to_pydantic(r) for r in result.scalars()]


@router.get("/audit", response_model=List[ApprovalRequest])
async def get_audit_log(
    agent_name: Optional[str] = None,
    decision: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Audit log — queryable by agent, date, decision."""
    q = select(ApprovalRequestDB).order_by(ApprovalRequestDB.created_at.desc())
    if agent_name:
        q = q.where(ApprovalRequestDB.agent_name == agent_name)
    if decision:
        q = q.where(ApprovalRequestDB.status == decision)

    result = await db.execute(q)
    return [_to_pydantic(r) for r in result.scalars()]


@router.get("/{request_id}", response_model=ApprovalRequest)
async def get_approval(request_id: str, db: AsyncSession = Depends(get_db)):
    """Get single approval with full context."""
    db_item = await _get_or_404(request_id, db)
    return _to_pydantic(db_item)


@router.post("/{request_id}/decide", response_model=ApprovalRequest)
async def resolve_approval(
    request_id: str,
    decision: ApprovalDecision,
    db: AsyncSession = Depends(get_db),
):
    """Submit approve/reject/escalate decision."""
    db_item = await _get_or_404(request_id, db)
    if db_item.status != ApprovalStatus.PENDING:
        raise HTTPException(status_code=409, detail="Request already resolved.")

    db_item.status = decision.decision
    db_item.decided_by = decision.reviewer_id
    db_item.decision_notes = decision.notes
    db_item.escalated_to = decision.escalate_to
    db_item.decided_at = datetime.now(timezone.utc)

    # Optionally update payload if modified by reviewer
    if decision.payload:
        db_item.payload = json.dumps(decision.payload)

    await db.commit()
    await db.refresh(db_item)

    return _to_pydantic(db_item)


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _get_or_404(request_id: str, db: AsyncSession) -> ApprovalRequestDB:
    result = await db.execute(
        select(ApprovalRequestDB).where(ApprovalRequestDB.id == request_id)
    )
    db_item = result.scalar_one_or_none()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Approval request not found.")
    return db_item


def _to_pydantic(db_item: ApprovalRequestDB) -> ApprovalRequest:
    return ApprovalRequest(
        id=db_item.id,
        session_id=db_item.session_id,
        invocation_id=db_item.invocation_id,
        function_call_id=db_item.function_call_id,
        app_name=db_item.app_name,
        user_id=db_item.user_id,
        agent_name=db_item.agent_name,
        tool_name=db_item.tool_name,
        message=db_item.message,
        payload=json.loads(db_item.payload) if db_item.payload else {},
        response_schema=json.loads(db_item.response_schema)
        if db_item.response_schema
        else {},
        risk_level=db_item.risk_level,
        status=db_item.status,
        created_at=db_item.created_at,
        decided_at=db_item.decided_at,
        decided_by=db_item.decided_by,
        decision_notes=db_item.decision_notes,
        escalated_to=db_item.escalated_to,
    )
