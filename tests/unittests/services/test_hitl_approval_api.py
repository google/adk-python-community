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

"""Integration tests for the FastAPI approval endpoints."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from google.adk_community.services.hitl_approval.api import app
from google.adk_community.services.hitl_approval.store import Base, get_db

# Use an in-memory SQLite database for tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )
    async with session_factory() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_create_approval(client):
    payload = {
        "session_id": "sess-1",
        "app_name": "test_app",
        "user_id": "u-123",
        "agent_name": "email_agent",
        "tool_name": "send_email",
        "message": "Please approve sending email.",
        "payload": {"to": "alice@example.com"},
    }
    resp = await client.post("/approvals/", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "pending"
    assert data["tool_name"] == "send_email"
    return data["id"]


@pytest.mark.asyncio
async def test_resolve_approval(client):
    # Create first
    create_resp = await client.post(
        "/approvals/",
        json={
            "session_id": "sess-2",
            "app_name": "test_app",
            "user_id": "u-123",
            "agent_name": "file_agent",
            "tool_name": "delete_file",
            "message": "Approve delete?",
            "payload": {"path": "/tmp/test.txt"},
        },
    )
    assert create_resp.status_code == 201
    request_id = create_resp.json()["id"]

    # Resolve
    resolve_resp = await client.post(
        f"/approvals/{request_id}/decide",
        json={"decision": "approved", "reviewer_id": "rev-99", "notes": "Looks safe."},
    )
    assert resolve_resp.status_code == 200
    data = resolve_resp.json()
    assert data["status"] == "approved"
    assert data["decision_notes"] == "Looks safe."
    assert data["decided_at"] is not None


@pytest.mark.asyncio
async def test_double_resolve_returns_409(client):
    create_resp = await client.post(
        "/approvals/",
        json={
            "session_id": "sess-3",
            "app_name": "test_app",
            "user_id": "u-123",
            "agent_name": "researcher",
            "tool_name": "web_search",
            "message": "Search the web?",
            "payload": {"query": "latest news"},
        },
    )
    request_id = create_resp.json()["id"]

    await client.post(
        f"/approvals/{request_id}/decide",
        json={"decision": "rejected", "reviewer_id": "rev-1"},
    )
    resp2 = await client.post(
        f"/approvals/{request_id}/decide",
        json={"decision": "approved", "reviewer_id": "rev-1"},
    )
    assert resp2.status_code == 409


@pytest.mark.asyncio
async def test_list_pending(client):
    # Create two requests
    for tool in ["tool_a", "tool_b"]:
        await client.post(
            "/approvals/",
            json={
                "session_id": "s",
                "app_name": "app",
                "user_id": "u",
                "agent_name": "ag",
                "tool_name": tool,
                "message": "msg",
                "payload": {},
            },
        )

    resp = await client.get("/approvals/pending")
    assert resp.status_code == 200
    assert len(resp.json()) == 2
    assert all(r["status"] == "pending" for r in resp.json())
