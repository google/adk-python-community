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

"""Tests for FirestoreSessionService.

All Firestore interactions are mocked in-memory — no GCP project needed.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.base_session_service import GetSessionConfig
from google.genai import types


# ---------------------------------------------------------------------------
# Lightweight in-memory Firestore mock
# ---------------------------------------------------------------------------

class _FakeDocSnapshot:
  """Mimics a Firestore DocumentSnapshot."""

  def __init__(self, doc_id: str, data: Optional[dict] = None):
    self.id = doc_id
    self._data = data
    self.exists = data is not None
    self.reference = MagicMock()
    self.reference.delete = AsyncMock()

  def to_dict(self) -> dict:
    return copy.deepcopy(self._data) if self._data else {}


class _FakeDocRef:
  """Mimics a Firestore AsyncDocumentReference."""

  def __init__(self, store: dict, doc_id: str, parent_path: str = ""):
    self._store = store
    self._id = doc_id
    self._path = f"{parent_path}/{doc_id}" if parent_path else doc_id

  async def get(self, transaction=None) -> _FakeDocSnapshot:
    data = self._store.get(self._path)
    return _FakeDocSnapshot(self._id, copy.deepcopy(data))

  async def set(self, data: dict, merge: bool = False) -> None:
    if merge and self._path in self._store:
      existing = self._store[self._path]
      existing.update(data)
    else:
      self._store[self._path] = copy.deepcopy(data)

  async def update(self, data: dict) -> None:
    if self._path in self._store:
      self._store[self._path].update(copy.deepcopy(data))

  async def delete(self) -> None:
    self._store.pop(self._path, None)

  def collection(self, name: str):
    return _FakeCollection(self._store, f"{self._path}/{name}")


class _FakeQuery:
  """Mimics a Firestore query with where / order_by / limit_to_last."""

  def __init__(self, docs: list[_FakeDocSnapshot]):
    self._docs = docs
    self._filters: list[tuple[str, str, Any]] = []
    self._order_field: Optional[str] = None
    self._limit_last: Optional[int] = None

  def where(self, *, filter) -> _FakeQuery:
    self._filters.append(filter)
    return self

  def order_by(self, field: str) -> _FakeQuery:
    self._order_field = field
    return self

  def limit_to_last(self, n: int) -> _FakeQuery:
    self._limit_last = n
    return self

  async def stream(self):
    results = list(self._docs)
    for field, op, value in self._filters:
      filtered = []
      for doc in results:
        d = doc.to_dict()
        v = d.get(field)
        if op == "==" and v == value:
          filtered.append(doc)
        elif op == ">=" and v is not None and v >= value:
          filtered.append(doc)
      results = filtered

    if self._order_field:
      results.sort(
          key=lambda d: d.to_dict().get(self._order_field, 0)
      )

    if self._limit_last is not None:
      results = results[-self._limit_last:]

    for doc in results:
      yield doc


class _FakeCollection:
  """Mimics a Firestore AsyncCollectionReference."""

  def __init__(self, store: dict, path: str):
    self._store = store
    self._path = path

  def document(self, doc_id: str) -> _FakeDocRef:
    return _FakeDocRef(self._store, doc_id, self._path)

  def where(self, *, filter) -> _FakeQuery:
    docs = self._snapshot_docs()
    q = _FakeQuery(docs)
    q.where(filter=filter)
    return q

  def order_by(self, field: str) -> _FakeQuery:
    docs = self._snapshot_docs()
    q = _FakeQuery(docs)
    q.order_by(field)
    return q

  def _snapshot_docs(self) -> list[_FakeDocSnapshot]:
    prefix = self._path + "/"
    docs = []
    for key, data in self._store.items():
      if key.startswith(prefix):
        suffix = key[len(prefix):]
        if "/" not in suffix:
          docs.append(_FakeDocSnapshot(suffix, copy.deepcopy(data)))
    return docs

  async def stream(self):
    for doc in self._snapshot_docs():
      yield doc


class _FakeBatch:
  """Mimics a Firestore WriteBatch."""

  def __init__(self):
    self._ops: list[tuple[str, Any]] = []

  def delete(self, ref):
    self._ops.append(("delete", ref))

  async def commit(self):
    for op_type, ref in self._ops:
      if op_type == "delete":
        await ref.delete()
    self._ops.clear()


class _FakeTransaction:
  """Mimics a Firestore async transaction."""

  def __init__(self):
    self._writes: list[tuple] = []

  def set(self, ref, data, merge=False):
    self._writes.append(("set", ref, data, merge))

  def update(self, ref, data):
    self._writes.append(("update", ref, data))


class _FakeClient:
  """Mimics the Firestore AsyncClient."""

  def __init__(self):
    self._store: dict[str, dict] = {}

  def collection(self, path: str):
    return _FakeCollection(self._store, path)

  def transaction(self):
    return _FakeTransaction()

  @staticmethod
  def field_filter(field, op, value):
    return (field, op, value)

  def async_transactional(self, fn):
    """Wraps *fn* so it executes normally then applies writes."""

    async def wrapper(transaction):
      result = await fn(transaction)
      for op_type, ref, data, *rest in transaction._writes:
        if op_type == "set":
          merge = rest[0] if rest else False
          await ref.set(data, merge=merge)
        elif op_type == "update":
          await ref.update(data)
      return result

    return wrapper

  def batch(self):
    return _FakeBatch()

  def close(self):
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def service():
  """Creates a FirestoreSessionService with a fake in-memory backend."""
  with patch(
      "google.adk_community.sessions.firestore_session_service."
      "FirestoreSessionService.__init__",
      lambda self, **kw: None,
  ):
    from google.adk_community.sessions.firestore_session_service import (
        FirestoreSessionService,
    )

    svc = FirestoreSessionService()
    svc._db = _FakeClient()
    svc._prefix = ""
    yield svc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFirestoreSessionService:

  @pytest.mark.asyncio
  async def test_create_session(self, service):
    session = await service.create_session(
        app_name="app1", user_id="u1", state={"key": "val"}
    )
    assert session.app_name == "app1"
    assert session.user_id == "u1"
    assert session.id is not None
    assert session.state["key"] == "val"
    assert session.last_update_time > 0

  @pytest.mark.asyncio
  async def test_create_session_with_custom_id(self, service):
    session = await service.create_session(
        app_name="app1",
        user_id="u1",
        session_id="custom-id",
    )
    assert session.id == "custom-id"

  @pytest.mark.asyncio
  async def test_create_duplicate_session_raises(self, service):
    await service.create_session(
        app_name="app1", user_id="u1", session_id="dup"
    )
    with pytest.raises(ValueError, match="already exists"):
      await service.create_session(
          app_name="app1", user_id="u1", session_id="dup"
      )

  @pytest.mark.asyncio
  async def test_get_session(self, service):
    created = await service.create_session(
        app_name="app1", user_id="u1", state={"k": "v"}
    )
    fetched = await service.get_session(
        app_name="app1", user_id="u1", session_id=created.id
    )
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.state["k"] == "v"

  @pytest.mark.asyncio
  async def test_get_nonexistent_session(self, service):
    result = await service.get_session(
        app_name="app1", user_id="u1", session_id="nope"
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_get_session_wrong_app(self, service):
    created = await service.create_session(
        app_name="app1", user_id="u1"
    )
    result = await service.get_session(
        app_name="wrong_app", user_id="u1", session_id=created.id
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_list_sessions(self, service):
    for i in range(3):
      await service.create_session(
          app_name="app1",
          user_id="u1",
          session_id=f"s{i}",
      )
    resp = await service.list_sessions(app_name="app1", user_id="u1")
    assert len(resp.sessions) == 3
    ids = {s.id for s in resp.sessions}
    assert ids == {"s0", "s1", "s2"}

  @pytest.mark.asyncio
  async def test_delete_session(self, service):
    session = await service.create_session(
        app_name="app1", user_id="u1", session_id="del-me"
    )
    await service.delete_session(
        app_name="app1", user_id="u1", session_id="del-me"
    )
    result = await service.get_session(
        app_name="app1", user_id="u1", session_id="del-me"
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_delete_nonexistent_session(self, service):
    # Should not raise.
    await service.delete_session(
        app_name="app1", user_id="u1", session_id="ghost"
    )

  @pytest.mark.asyncio
  async def test_append_event(self, service):
    session = await service.create_session(
        app_name="app1", user_id="u1", session_id="ev-test"
    )
    event = Event(
        invocation_id="inv1",
        author="user",
        content=types.Content(
            role="user", parts=[types.Part(text="hello")]
        ),
    )
    returned = await service.append_event(session=session, event=event)
    assert returned.id == event.id

    fetched = await service.get_session(
        app_name="app1", user_id="u1", session_id="ev-test"
    )
    assert len(fetched.events) == 1
    assert fetched.events[0].content.parts[0].text == "hello"

  @pytest.mark.asyncio
  async def test_append_event_partial_skipped(self, service):
    session = await service.create_session(
        app_name="app1", user_id="u1", session_id="partial-test"
    )
    event = Event(author="user", partial=True)
    result = await service.append_event(session=session, event=event)
    assert result is event

    fetched = await service.get_session(
        app_name="app1", user_id="u1", session_id="partial-test"
    )
    assert len(fetched.events) == 0

  @pytest.mark.asyncio
  async def test_append_event_with_state_delta(self, service):
    session = await service.create_session(
        app_name="app1", user_id="u1", session_id="delta-test"
    )
    event = Event(
        invocation_id="inv1",
        author="agent",
        actions=EventActions(
            state_delta={
                "app:color": "blue",
                "user:lang": "en",
                "local_key": "local_val",
            }
        ),
    )
    await service.append_event(session=session, event=event)

    fetched = await service.get_session(
        app_name="app1", user_id="u1", session_id="delta-test"
    )
    assert fetched.state.get("app:color") == "blue"
    assert fetched.state.get("user:lang") == "en"
    assert fetched.state.get("local_key") == "local_val"

  @pytest.mark.asyncio
  async def test_app_state_shared_across_sessions(self, service):
    s1 = await service.create_session(
        app_name="shared",
        user_id="u1",
        session_id="s1",
        state={"app:version": "1.0"},
    )
    s2 = await service.create_session(
        app_name="shared", user_id="u1", session_id="s2"
    )
    assert s2.state.get("app:version") == "1.0"

  @pytest.mark.asyncio
  async def test_user_state_shared_across_sessions(self, service):
    s1 = await service.create_session(
        app_name="app1",
        user_id="u1",
        session_id="us1",
        state={"user:pref": "dark"},
    )
    s2 = await service.create_session(
        app_name="app1", user_id="u1", session_id="us2"
    )
    assert s2.state.get("user:pref") == "dark"

  @pytest.mark.asyncio
  async def test_get_session_num_recent_events(self, service):
    session = await service.create_session(
        app_name="app1", user_id="u1", session_id="recent"
    )
    for i in range(5):
      event = Event(
          invocation_id=f"inv{i}",
          author="user",
          timestamp=float(i + 1),
      )
      await service.append_event(session=session, event=event)

    config = GetSessionConfig(num_recent_events=2)
    fetched = await service.get_session(
        app_name="app1",
        user_id="u1",
        session_id="recent",
        config=config,
    )
    assert len(fetched.events) == 2
    assert fetched.events[0].timestamp == 4.0
    assert fetched.events[1].timestamp == 5.0

  @pytest.mark.asyncio
  async def test_get_session_after_timestamp(self, service):
    session = await service.create_session(
        app_name="app1", user_id="u1", session_id="after"
    )
    for i in range(5):
      event = Event(
          invocation_id=f"inv{i}",
          author="user",
          timestamp=float(i + 1),
      )
      await service.append_event(session=session, event=event)

    config = GetSessionConfig(after_timestamp=3.0)
    fetched = await service.get_session(
        app_name="app1",
        user_id="u1",
        session_id="after",
        config=config,
    )
    assert len(fetched.events) == 3
    assert fetched.events[0].timestamp == 3.0

  @pytest.mark.asyncio
  async def test_close_and_context_manager(self, service):
    async with service:
      session = await service.create_session(
          app_name="app1", user_id="u1"
      )
      assert session is not None

  @pytest.mark.asyncio
  async def test_temp_state_not_persisted(self, service):
    session = await service.create_session(
        app_name="app1",
        user_id="u1",
        session_id="temp-test",
        state={"temp:scratch": "gone", "keep": "this"},
    )
    assert session.state.get("keep") == "this"
    assert "temp:scratch" not in session.state

  @pytest.mark.asyncio
  async def test_collection_prefix(self):
    with patch(
        "google.adk_community.sessions.firestore_session_service."
        "FirestoreSessionService.__init__",
        lambda self, **kw: None,
    ):
      from google.adk_community.sessions.firestore_session_service import (
          FirestoreSessionService,
      )

      svc = FirestoreSessionService()
      svc._db = _FakeClient()
      svc._prefix = "test_"

      session = await svc.create_session(
          app_name="app1", user_id="u1", session_id="prefixed"
      )
      assert session.id == "prefixed"

      fetched = await svc.get_session(
          app_name="app1", user_id="u1", session_id="prefixed"
      )
      assert fetched is not None
