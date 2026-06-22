# Copyright 2025 Google LLC
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

"""Tests for BufferedFirestoreSessionService.

Uses an in-memory fake Firestore AsyncClient (no external services), a
deterministic clock, and a recording sleeper. The service's transactional
wrapper is replaced with an identity (or gated/flaky) runner so the fake
transaction is driven directly.
"""

import asyncio

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.base_session_service import GetSessionConfig
from google.genai import types
import pytest

from google.adk_community.sessions.firestore_session_service import BufferedFirestoreSessionService
from google.adk_community.sessions.firestore_session_service import SessionPersistenceError

APP = "app"
USER = "user-1"
SID = "session-1"


# --- fake Firestore ----------------------------------------------------------


class FakeSnapshot:

  def __init__(self, doc_id, data, ref):
    self.id = doc_id
    self._data = data
    self.reference = ref

  @property
  def exists(self):
    return self._data is not None

  def to_dict(self):
    return dict(self._data) if self._data is not None else None


class FakeDoc:

  def __init__(self, doc_id):
    self.id = doc_id
    self.data = None
    self._subcollections = {}
    self.reference = self

  async def get(self, transaction=None):
    return FakeSnapshot(self.id, self.data, self)

  def collection(self, name):
    return self._subcollections.setdefault(name, FakeCollection(name))

  async def delete(self):
    self.data = None

  def set(self, data, merge=False):
    if merge and isinstance(self.data, dict):
      merged = dict(self.data)
      merged.update(data)
      self.data = merged
    else:
      self.data = dict(data)

  def update(self, data):
    self.data = {**(self.data or {}), **data}


def _match(actual, op, value):
  if actual is None:
    return False
  if op == "==":
    return actual == value
  if op == ">=":
    return actual >= value
  return False


class FakeQuery:

  def __init__(self, docs):
    self._docs = docs
    self._order = None
    self._filters = []
    self._limit_last = None

  def order_by(self, field):
    self._order = field
    return self

  def where(self, field, op, value):
    self._filters.append((field, op, value))
    return self

  def limit_to_last(self, n):
    self._limit_last = n
    return self

  async def get(self):
    rows = [d for d in self._docs if d.data is not None]
    for field, op, value in self._filters:
      rows = [d for d in rows if _match(d.data.get(field), op, value)]
    if self._order:
      rows = sorted(rows, key=lambda d: d.data.get(self._order))
    if self._limit_last is not None:
      rows = rows[-self._limit_last :]
    return [FakeSnapshot(d.id, d.data, d) for d in rows]


class FakeCollection:

  def __init__(self, name):
    self.name = name
    self.docs = {}

  def document(self, doc_id):
    if doc_id not in self.docs:
      self.docs[doc_id] = FakeDoc(doc_id)
    return self.docs[doc_id]

  def order_by(self, field):
    return FakeQuery(list(self.docs.values())).order_by(field)

  def where(self, field, op, value):
    return FakeQuery(list(self.docs.values())).where(field, op, value)

  async def get(self):
    return await FakeQuery(list(self.docs.values())).get()

  async def stream(self):
    for d in list(self.docs.values()):
      if d.data is not None:
        yield FakeSnapshot(d.id, d.data, d)


class FakeTransaction:

  def set(self, ref, data, merge=False):
    ref.set(data, merge=merge)

  def update(self, ref, data):
    ref.update(data)


class FakeBatch:

  def __init__(self):
    self._ops = []

  def delete(self, ref):
    self._ops.append(ref)

  async def commit(self):
    for ref in self._ops:
      ref.data = None
    self._ops = []


class FakeFirestore:

  def __init__(self):
    self.collections = {}
    self.transaction_count = 0

  def collection(self, name):
    return self.collections.setdefault(name, FakeCollection(name))

  def collection_group(self, name):
    return FakeQuery(self._gather_group(name))

  def transaction(self):
    self.transaction_count += 1
    return FakeTransaction()

  def batch(self):
    return FakeBatch()

  def _gather_group(self, name):
    result = []

    def walk(coll):
      for doc in coll.docs.values():
        for sub_name, sub in doc._subcollections.items():
          if sub_name == name:
            result.extend(d for d in sub.docs.values() if d.data is not None)
          walk(sub)

    for coll in self.collections.values():
      if coll.name == name:
        result.extend(d for d in coll.docs.values() if d.data is not None)
      walk(coll)
    return result


# --- helpers -----------------------------------------------------------------


class Clock:

  def __init__(self, start=1000.0):
    self.now = start

  def __call__(self):
    return self.now

  def advance(self, seconds):
    self.now += seconds


class RecordingSleeper:

  def __init__(self):
    self.delays = []

  async def __call__(self, delay):
    self.delays.append(delay)


class Aborted(Exception):
  """Name matches the retryable allowlist."""


def _identity_transactional(fn):

  async def run(transaction):
    return await fn(transaction)

  return run


def _make(**kwargs):
  client = FakeFirestore()
  clock = Clock()
  sleeper = RecordingSleeper()
  service = BufferedFirestoreSessionService(
      client, clock=clock, sleeper=sleeper, **kwargs
  )
  service._transactional = _identity_transactional
  return service, client, clock, sleeper


def _event(author, text, timestamp, *, state_delta=None):
  return Event(
      invocation_id=f"inv-{timestamp}",
      author=author,
      timestamp=timestamp,
      content=types.Content(
          role="user" if author == "user" else "model",
          parts=[types.Part(text=text)],
      ),
      actions=EventActions(state_delta=state_delta or {}),
  )


def _session_doc(client, session_id=SID):
  return (
      client.collection("adk-session")
      .document(APP)
      .collection("users")
      .document(USER)
      .collection("sessions")
      .document(session_id)
  )


def _persisted_event_count(client, session_id=SID):
  events = _session_doc(client, session_id)._subcollections.get("events")
  if events is None:
    return 0
  return sum(1 for d in events.docs.values() if d.data is not None)


# --- tests -------------------------------------------------------------------


async def test_create_session_writes_metadata():
  service, client, *_ = _make()
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id=SID
  )
  assert session.id == SID
  doc = _session_doc(client).data
  assert doc["appName"] == APP
  assert doc["userId"] == USER
  assert doc["revision"] == 1


async def test_buffered_append_defers_persistence():
  service, client, *_ = _make(buffer_max_events=10)
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id=SID
  )
  base = client.transaction_count
  for i in range(9):
    await service.append_event(session, _event("user", f"m{i}", float(i)))
  assert _persisted_event_count(client) == 0
  assert client.transaction_count == base


async def test_flush_persists_whole_batch_in_one_transaction():
  service, client, *_ = _make()
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id=SID
  )
  for i in range(9):
    await service.append_event(session, _event("user", f"m{i}", float(i)))
  before = client.transaction_count
  await service.flush_session(SID)
  assert client.transaction_count - before == 1
  assert _persisted_event_count(client) == 9
  assert _session_doc(client).data["revision"] == 2


async def test_reaching_max_events_auto_flushes():
  service, client, *_ = _make(buffer_max_events=10)
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id=SID
  )
  for i in range(10):
    await service.append_event(session, _event("user", f"m{i}", float(i)))
  assert _persisted_event_count(client) == 10


async def test_durable_mode_writes_each_event_immediately():
  service, client, *_ = _make(durable_mode=True)
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id=SID
  )
  base = client.transaction_count
  for i in range(3):
    await service.append_event(session, _event("user", f"m{i}", float(i)))
  assert client.transaction_count - base == 3
  assert _persisted_event_count(client) == 3
  assert SID not in service._buffers


async def test_periodic_flush_after_interval():
  service, client, clock, _ = _make(flush_interval_seconds=120.0)
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id=SID
  )
  for i in range(3):
    await service.append_event(session, _event("user", f"m{i}", float(i)))
  assert _persisted_event_count(client) == 0
  clock.advance(121.0)
  await asyncio.gather(*await service._flush_due())
  assert _persisted_event_count(client) == 3


async def test_flush_hook_and_stop_final_flush():
  service, client, *_ = _make()
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id=SID
  )
  await service.append_event(session, _event("user", "a", 1.0))
  await service.flush()  # ADK Runner.close() hook
  assert _persisted_event_count(client) == 1


async def test_get_session_merges_and_orders_without_duplicates():
  service, *_ = _make()
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id=SID
  )
  await service.append_event(session, _event("user", "persisted", 1.0))
  await service.flush_session(SID)
  await service.append_event(session, _event("user", "buffered", 2.0))
  loaded = await service.get_session(app_name=APP, user_id=USER, session_id=SID)
  texts = [e.content.parts[0].text for e in loaded.events]
  assert texts == ["persisted", "buffered"]
  assert len(texts) == len({e.id for e in loaded.events})


async def test_state_delta_scoping():
  service, *_ = _make()
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id=SID
  )
  await service.append_event(
      session,
      _event(
          "user",
          "a",
          1.0,
          state_delta={
              "app:shared": "yes",
              "user:goal": "fat loss",
              "sessionOnly": "kept",
              "temp:scratch": "discard",
          },
      ),
  )
  await service.flush_session(SID)
  loaded = await service.get_session(app_name=APP, user_id=USER, session_id=SID)
  assert loaded.state["app:shared"] == "yes"
  assert loaded.state["user:goal"] == "fat loss"
  assert loaded.state["sessionOnly"] == "kept"
  assert "temp:scratch" not in loaded.state


async def test_get_user_state():
  service, *_ = _make()
  await service.create_session(
      app_name=APP, user_id=USER, session_id=SID, state={"user:goal": "lose"}
  )
  state = await service.get_user_state(app_name=APP, user_id=USER)
  assert state == {"goal": "lose"}


async def test_retryable_failures_backoff_then_succeed():
  service, client, _, sleeper = _make(max_retry_attempts=5)
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id=SID
  )
  errors = [Aborted(), Aborted()]

  def flaky(fn):

    async def run(transaction):
      if errors:
        raise errors.pop(0)
      return await fn(transaction)

    return run

  service._transactional = flaky
  await service.append_event(session, _event("user", "a", 1.0))
  await service.flush_session(SID)
  assert _persisted_event_count(client) == 1
  assert len(sleeper.delays) == 2
  assert sleeper.delays[1] > sleeper.delays[0]


async def test_permanent_failure_not_retried():
  service, _, _, sleeper = _make()
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id=SID
  )

  def boom(fn):

    async def run(transaction):
      raise ValueError("permission denied")

    return run

  service._transactional = boom
  await service.append_event(session, _event("user", "a", 1.0))
  with pytest.raises(SessionPersistenceError):
    await service.flush_session(SID)
  assert sleeper.delays == []


async def test_events_appended_during_flush_not_lost():
  service, client, *_ = _make()
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id=SID
  )
  gate = asyncio.Event()
  entered = asyncio.Event()

  def gated(fn):

    async def run(transaction):
      entered.set()
      await gate.wait()
      return await fn(transaction)

    return run

  await service.append_event(session, _event("user", "a", 1.0))
  await service.append_event(session, _event("user", "b", 2.0))
  service._transactional = gated

  flush_task = asyncio.create_task(service.flush_session(SID))
  await entered.wait()
  await service.append_event(session, _event("user", "c", 3.0))
  gate.set()
  await flush_task

  pending = service._buffers[SID].pending_events
  assert [e.content.parts[0].text for e in pending] == ["c"]
  assert _persisted_event_count(client) == 2


async def test_concurrent_flushes_do_not_duplicate():
  service, client, *_ = _make()
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id=SID
  )
  gate = asyncio.Event()
  entered = asyncio.Event()

  def gated(fn):

    async def run(transaction):
      entered.set()
      await gate.wait()
      return await fn(transaction)

    return run

  await service.append_event(session, _event("user", "a", 1.0))
  await service.append_event(session, _event("user", "b", 2.0))
  service._transactional = gated
  before = client.transaction_count

  t1 = asyncio.create_task(service.flush_session(SID))
  await entered.wait()
  t2 = asyncio.create_task(service.flush_session(SID))
  gate.set()
  await asyncio.gather(t1, t2)

  assert client.transaction_count - before == 1
  assert _persisted_event_count(client) == 2


async def test_delete_session_removes_events_and_buffer():
  service, client, *_ = _make()
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id=SID
  )
  await service.append_event(session, _event("user", "a", 1.0))
  await service.flush_session(SID)
  await service.delete_session(app_name=APP, user_id=USER, session_id=SID)
  assert _session_doc(client).data is None
  assert _persisted_event_count(client) == 0
  assert SID not in service._buffers


async def test_get_session_not_found_returns_none():
  service, *_ = _make()
  result = await service.get_session(
      app_name=APP, user_id=USER, session_id="missing"
  )
  assert result is None


async def test_list_sessions():
  service, *_ = _make()
  await service.create_session(app_name=APP, user_id=USER, session_id="s1")
  await service.create_session(app_name=APP, user_id=USER, session_id="s2")
  per_user = await service.list_sessions(app_name=APP, user_id=USER)
  all_users = await service.list_sessions(app_name=APP)
  assert {s.id for s in per_user.sessions} == {"s1", "s2"}
  assert {s.id for s in all_users.sessions} == {"s1", "s2"}


async def test_get_session_with_config_num_recent_events():
  service, *_ = _make()
  session = await service.create_session(
      app_name=APP, user_id=USER, session_id=SID
  )
  for i in range(3):
    await service.append_event(session, _event("user", f"m{i}", float(i)))
  await service.flush_session(SID)
  loaded = await service.get_session(
      app_name=APP,
      user_id=USER,
      session_id=SID,
      config=GetSessionConfig(num_recent_events=2),
  )
  assert [e.content.parts[0].text for e in loaded.events] == ["m1", "m2"]


async def test_create_session_duplicate_raises():
  from google.adk.errors.already_exists_error import AlreadyExistsError

  service, *_ = _make()
  await service.create_session(app_name=APP, user_id=USER, session_id=SID)
  with pytest.raises(AlreadyExistsError):
    await service.create_session(app_name=APP, user_id=USER, session_id=SID)


async def test_start_stop_cancellation_is_clean():
  service, *_ = _make()
  service._sleeper = asyncio.sleep  # real sleep so the loop blocks
  await service.start()
  await service.start()  # idempotent
  task = service._task
  await service.stop()
  await service.stop()  # idempotent
  assert task.cancelled() or task.done()
