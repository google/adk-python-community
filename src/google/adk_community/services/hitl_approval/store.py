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

"""Async SQLite database setup via SQLAlchemy."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class ApprovalRequestDB(Base):
    __tablename__ = "approval_requests"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False)
    invocation_id = Column(String, nullable=True)
    function_call_id = Column(String, nullable=True)
    app_name = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    agent_name = Column(String, nullable=False)
    tool_name = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    payload = Column(Text, nullable=False)  # JSON-serialised
    response_schema = Column(Text, nullable=True)  # JSON-serialised
    risk_level = Column(String, nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    decided_at = Column(DateTime, nullable=True)
    decided_by = Column(String, nullable=True)
    decision_notes = Column(Text, nullable=True)
    escalated_to = Column(String, nullable=True)


import os

db_path = os.getenv("HITL_DB_PATH", "./hitl.db")
DATABASE_URL = f"sqlite+aiosqlite:///{db_path}"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)


async def init_db() -> None:
    """Create tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency that yields a database session."""
    async with AsyncSessionLocal() as session:
        yield session
