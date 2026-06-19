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

"""A Firestore-backed ADK session service with batched, buffered event writes.

``BufferedFirestoreSessionService`` mirrors the data model of the builtin
``google.adk.integrations.firestore.FirestoreSessionService`` (same collection
hierarchy, app/user/session state scoping, optimistic concurrency via a
``revision`` field, and idempotent event documents keyed by ``event.id``) but
**owns** the Firestore I/O so it can persist a whole batch of buffered events in
a **single transaction**.

Collection hierarchy::

    adk-session/{app}/users/{user}/sessions/{session}/events/{event}
    app_states/{app}
    user_states/{app}/users/{user}

Events accumulate in a per-session in-memory buffer and flush when the buffer
reaches ``buffer_max_events``, when ``flush_interval_seconds`` elapses (the
background task started by :meth:`start`), when ``flush_session`` / ``flush_all``
/ ``flush`` is called, or when :meth:`stop` runs. Set ``durable_mode=True`` to
persist every event immediately (no buffering).

Batching does not change the event-document count, but it collapses the repeated
session-doc + state-doc updates and per-event transactions from N to 1 (fewer
round-trips and less optimistic-lock contention). On an abrupt process death
before a flush, up to ``flush_interval_seconds`` of events (or
``buffer_max_events - 1`` per session) may be lost; ``stop()`` flushes on
graceful shutdown but cannot protect against crashes.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable
from collections.abc import Callable
import copy
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
import logging
import random
import time
from typing import Any
from typing import Optional
import uuid

from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.events.event import Event
from google.adk.sessions import _session_util
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.base_session_service import ListSessionsResponse
from google.adk.sessions.session import Session
from google.adk.sessions.state import State
from typing_extensions import override

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_ROOT_COLLECTION = "adk-session"
DEFAULT_SESSIONS_COLLECTION = "sessions"
DEFAULT_EVENTS_COLLECTION = "events"
DEFAULT_APP_STATE_COLLECTION = "app_states"
DEFAULT_USER_STATE_COLLECTION = "user_states"

# Transient Firestore / gRPC failures worth retrying. Matched by class name to
# avoid a hard dependency on google.api_core being importable everywhere.
_RETRYABLE_ERROR_NAMES = frozenset({
    "DeadlineExceeded",
    "ServiceUnavailable",
    "Aborted",
    "ResourceExhausted",
    "InternalServerError",
    "Internal",
    "Cancelled",
    "RetryError",
    "TooManyRequests",
})
_NON_RETRYABLE_TYPES: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    AlreadyExistsError,
    PermissionError,
)
_NON_RETRYABLE_ERROR_NAMES = frozenset({
    "PermissionDenied",
    "InvalidArgument",
    "NotFound",
    "Unauthenticated",
    "FailedPrecondition",
})


class SessionPersistenceError(RuntimeError):
  """Raised when an explicit flush fails to persist after exhausting retries."""


def is_retryable_error(exc: BaseException) -> bool:
  """Classifies an error as transient/retryable vs. a permanent caller error."""
  if isinstance(exc, _NON_RETRYABLE_TYPES):
    return False
  name = type(exc).__name__
  if name in _NON_RETRYABLE_ERROR_NAMES:
    return False
  if name in _RETRYABLE_ERROR_NAMES:
    return True
  return False


@dataclass
class _SessionBuffer:
  """In-memory pending state for a single session."""

  pending_events: deque[Event] = field(default_factory=deque)
  last_flush_monotonic: float = 0.0
  lock: asyncio.Lock = field(default_factory=asyncio.Lock)
  flush_in_progress: bool = False


class BufferedFirestoreSessionService(BaseSessionService):  # type: ignore[misc]
  """A Firestore-backed session service with batched, buffered event writes."""

  def __init__(
      self,
      client: Any = None,
      root_collection: Optional[str] = None,
      *,
      sessions_collection: str = DEFAULT_SESSIONS_COLLECTION,
      events_collection: str = DEFAULT_EVENTS_COLLECTION,
      app_state_collection: str = DEFAULT_APP_STATE_COLLECTION,
      user_state_collection: str = DEFAULT_USER_STATE_COLLECTION,
      flat_layout: bool = False,
      durable_mode: bool = False,
      buffer_max_events: int = 10,
      flush_interval_seconds: float = 120.0,
      max_retry_attempts: int = 5,
      retry_base_delay_seconds: float = 0.5,
      clock: Callable[[], float] = time.monotonic,
      sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
  ) -> None:
    """Initializes the buffered Firestore session service.

    Args:
      client: An optional Firestore ``AsyncClient``. If not provided, a new one
        is created (requires ``google-cloud-firestore``).
      root_collection: Root collection name. Defaults to ``'adk-session'``.
      sessions_collection: Subcollection name for sessions. Defaults to
        ``'sessions'``.
      events_collection: Subcollection name for events. Defaults to
        ``'events'``.
      app_state_collection: Root collection for app-scoped state. Defaults to
        ``'app_states'``.
      user_state_collection: Root collection for user-scoped state. Defaults
        to ``'user_states'``.
      flat_layout: When True, session documents live directly in
        ``root_collection/{session_id}`` (no ``{app}/users/{user}/sessions/``
        nesting). Useful when the session id already encodes the user (e.g.
        ``{phone}-{date}``). Defaults to False.
      durable_mode: When True, every event is persisted immediately and no
        buffering happens.
      buffer_max_events: Flush a session once this many events are buffered.
      flush_interval_seconds: Background flush cadence (see :meth:`start`).
      max_retry_attempts: Max attempts when a flush hits a retryable error.
      retry_base_delay_seconds: Base delay for exponential backoff with jitter.
      clock: Monotonic clock, injectable for tests.
      sleeper: Async sleep function, injectable for tests.
    """
    try:
      from google.cloud import firestore
    except ImportError as e:
      raise ImportError(
          "BufferedFirestoreSessionService requires google-cloud-firestore."
          " Install it with: pip install google-adk-community[firestore]"
      ) from e

    self._firestore = firestore
    self.client = client if client is not None else firestore.AsyncClient()
    self.root_collection = root_collection or DEFAULT_ROOT_COLLECTION
    self.sessions_collection = sessions_collection
    self.events_collection = events_collection
    self.app_state_collection = app_state_collection
    self.user_state_collection = user_state_collection
    # flat_layout=True: sessions/{session_id} (no {app}/users/{user} nesting)
    # flat_layout=False (default): {root}/{app}/users/{user}/{sessions}/{session_id}
    self._flat_layout = flat_layout

    self._durable_mode = durable_mode
    self._buffer_max_events = buffer_max_events
    self._flush_interval_seconds = flush_interval_seconds
    self._max_retry_attempts = max_retry_attempts
    self._retry_base_delay_seconds = retry_base_delay_seconds
    self._clock = clock
    self._sleeper = sleeper
    # Injectable so tests can drive a fake client without the real transactional
    # retry wrapper.
    self._transactional = firestore.async_transactional

    self._buffers: dict[str, _SessionBuffer] = {}
    self._session_refs: dict[str, Session] = {}
    self._buffers_guard = asyncio.Lock()
    self._task: Optional[asyncio.Task[None]] = None
    self._check_interval = max(1.0, min(flush_interval_seconds, 5.0))

  # -- Firestore refs / helpers ---------------------------------------------

  def _get_sessions_ref(self, app_name: str, user_id: str) -> Any:
    if self._flat_layout:
      return self.client.collection(self.root_collection)
    return (
        self.client.collection(self.root_collection)
        .document(app_name)
        .collection("users")
        .document(user_id)
        .collection(self.sessions_collection)
    )

  def _app_state_ref(self, app_name: str) -> Any:
    return self.client.collection(self.app_state_collection).document(app_name)

  def _user_state_ref(self, app_name: str, user_id: str) -> Any:
    return (
        self.client.collection(self.user_state_collection)
        .document(app_name)
        .collection("users")
        .document(user_id)
    )

  @staticmethod
  def _merge_state(
      app_state: dict[str, Any],
      user_state: dict[str, Any],
      session_state: dict[str, Any],
  ) -> dict[str, Any]:
    merged = copy.deepcopy(session_state)
    for key, value in app_state.items():
      merged[State.APP_PREFIX + key] = value
    for key, value in user_state.items():
      merged[State.USER_PREFIX + key] = value
    return merged

  async def _read_state(self, ref: Any) -> dict[str, Any]:
    doc = await ref.get()
    return (doc.to_dict() or {}) if doc.exists else {}

  @staticmethod
  def _coerce_timestamp(value: Any) -> float:
    if isinstance(value, datetime):
      return value.timestamp()
    try:
      return float(value)
    except (ValueError, TypeError):
      return 0.0

  # -- CRUD ------------------------------------------------------------------

  @override
  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    """Creates a new session (raises AlreadyExistsError on a duplicate id)."""
    session_id = session_id or str(uuid.uuid4())
    deltas = _session_util.extract_state_delta(state or {})
    session_ref = self._get_sessions_ref(app_name, user_id).document(session_id)
    app_ref = self._app_state_ref(app_name)
    user_ref = self._user_state_ref(app_name, user_id)
    session_data = {
        "id": session_id,
        "appName": app_name,
        "userId": user_id,
        "state": deltas["session"],
        "createTime": self._firestore.SERVER_TIMESTAMP,
        "updateTime": self._firestore.SERVER_TIMESTAMP,
        "revision": 1,
    }

    async def _create_txn(transaction: Any) -> None:
      snap = await session_ref.get(transaction=transaction)
      if snap.exists:
        raise AlreadyExistsError(f"Session {session_id} already exists.")
      if deltas["app"]:
        app_snap = await app_ref.get(transaction=transaction)
        current = app_snap.to_dict() if app_snap.exists else {}
        current.update(deltas["app"])
        transaction.set(app_ref, current, merge=True)
      if deltas["user"]:
        user_snap = await user_ref.get(transaction=transaction)
        current = user_snap.to_dict() if user_snap.exists else {}
        current.update(deltas["user"])
        transaction.set(user_ref, current, merge=True)
      transaction.set(session_ref, session_data)

    await self._transactional(_create_txn)(self.client.transaction())

    merged = self._merge_state(
        await self._read_state(app_ref),
        await self._read_state(user_ref),
        deltas["session"],
    )
    session = Session(
        id=session_id,
        app_name=app_name,
        user_id=user_id,
        state=merged,
        events=[],
        last_update_time=datetime.now(timezone.utc).timestamp(),
    )
    session._storage_update_marker = "1"
    return session

  @override
  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    """Gets a session, merging persisted and not-yet-flushed buffered events."""
    session_ref = self._get_sessions_ref(app_name, user_id).document(session_id)
    doc = await session_ref.get()
    if not doc.exists:
      return None
    data = doc.to_dict() or {}

    query = session_ref.collection(self.events_collection).order_by("timestamp")
    if config:
      if config.after_timestamp:
        query = query.where(
            "timestamp",
            ">=",
            datetime.fromtimestamp(config.after_timestamp, tz=timezone.utc),
        )
      if config.num_recent_events:
        query = query.limit_to_last(config.num_recent_events)
    events: list[Event] = []
    for event_doc in await query.get():
      event_data = event_doc.to_dict() or {}
      if "event_data" in event_data:
        events.append(Event.model_validate(event_data["event_data"]))

    merged_state = self._merge_state(
        await self._read_state(self._app_state_ref(app_name)),
        await self._read_state(self._user_state_ref(app_name, user_id)),
        data.get("state", {}) or {},
    )
    revision = data.get("revision", 0)
    session = Session(
        id=session_id,
        app_name=app_name,
        user_id=user_id,
        state=merged_state,
        events=events,
        last_update_time=self._coerce_timestamp(data.get("updateTime")),
    )
    session._storage_update_marker = str(revision) if revision > 0 else None
    return self._merge_buffered(session)

  @override
  async def list_sessions(
      self, *, app_name: str, user_id: Optional[str] = None
  ) -> ListSessionsResponse:
    """Lists sessions for an app (optionally a single user)."""
    if self._flat_layout:
      query = self.client.collection(self.root_collection).where(
          "appName", "==", app_name
      )
      if user_id:
        query = query.where("userId", "==", user_id)
      docs = await query.get()
    elif user_id:
      docs = await (
          self._get_sessions_ref(app_name, user_id)
          .where("appName", "==", app_name)
          .get()
      )
    else:
      docs = await (
          self.client.collection_group(self.sessions_collection)
          .where("appName", "==", app_name)
          .get()
      )

    app_state = await self._read_state(self._app_state_ref(app_name))
    user_states: dict[str, dict[str, Any]] = {}
    if user_id:
      user_states[user_id] = await self._read_state(
          self._user_state_ref(app_name, user_id)
      )
    else:
      users_ref = (
          self.client.collection(self.user_state_collection)
          .document(app_name)
          .collection("users")
      )
      for u_doc in await users_ref.get():
        user_states[u_doc.id] = u_doc.to_dict() or {}

    sessions: list[Session] = []
    for doc in docs:
      data = doc.to_dict()
      if not data:
        continue
      sessions.append(
          Session(
              id=data["id"],
              app_name=data["appName"],
              user_id=data["userId"],
              state=self._merge_state(
                  app_state,
                  user_states.get(data["userId"], {}),
                  data.get("state", {}) or {},
              ),
              events=[],
              last_update_time=0.0,
          )
      )
    return ListSessionsResponse(sessions=sessions)

  @override
  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    """Deletes a session, its events, and drops any pending buffer."""
    async with self._buffers_guard:
      self._buffers.pop(session_id, None)
      self._session_refs.pop(session_id, None)

    session_ref = self._get_sessions_ref(app_name, user_id).document(session_id)
    events_ref = session_ref.collection(self.events_collection)
    batch = self.client.batch()
    count = 0
    async for event_doc in events_ref.stream():
      batch.delete(event_doc.reference)
      count += 1
      if count >= 500:
        await batch.commit()
        batch = self.client.batch()
        count = 0
    if count > 0:
      await batch.commit()
    await session_ref.delete()

  @override
  async def get_user_state(
      self, *, app_name: str, user_id: str
  ) -> dict[str, Any]:
    """Returns the raw (un-prefixed) user-scoped state for an app/user."""
    return dict(await self._read_state(self._user_state_ref(app_name, user_id)))

  # -- buffered append -------------------------------------------------------

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    """Appends an event in memory and buffers (or immediately persists) it."""
    event = await super().append_event(session=session, event=event)
    if event.partial:
      return event

    buffered = event.model_copy(deep=True)
    if self._durable_mode:
      await self._persist_batch(session, [buffered])
      return event

    buffer = await self._get_or_create_buffer(session)
    async with buffer.lock:
      buffer.pending_events.append(buffered)
      pending = len(buffer.pending_events)

    if pending >= self._buffer_max_events:
      await self._flush(session.id, explicit=False)
    return event

  async def flush_session(self, session_id: str) -> None:
    """Explicitly flushes a session's buffer, raising on failure."""
    await self._flush(session_id, explicit=True)

  async def flush_all(self) -> None:
    """Flushes every buffered session. Failures are logged, events kept."""
    for session_id in list(self._buffers.keys()):
      try:
        await self._flush(session_id, explicit=False)
      except Exception:  # noqa: BLE001 - never abort shutdown; already logged
        logger.exception("flush_all_session_failed session_id=%s", session_id)

  async def flush(self) -> None:
    """ADK lifecycle hook (Runner.close()): flushes all buffered sessions."""
    await self.flush_all()

  async def _flush(self, session_id: str, *, explicit: bool) -> None:
    buffer = self._buffers.get(session_id)
    if buffer is None:
      return

    async with buffer.lock:
      if buffer.flush_in_progress:
        return  # only one flush per session at a time
      if not buffer.pending_events:
        buffer.last_flush_monotonic = self._clock()
        return
      buffer.flush_in_progress = True
      batch = list(buffer.pending_events)
      buffer.pending_events.clear()
      buffer.last_flush_monotonic = self._clock()
      session = self._session_refs.get(session_id)

    if session is None:  # pragma: no cover - defensive
      async with buffer.lock:
        buffer.pending_events.extendleft(reversed(batch))
        buffer.flush_in_progress = False
      return

    try:
      await self._persist_with_retry(session, batch, session_id)
    except Exception as exc:  # noqa: BLE001 - reclassified; never silently dropped
      async with buffer.lock:
        buffer.pending_events.extendleft(reversed(batch))
        buffer.flush_in_progress = False
      if explicit:
        raise SessionPersistenceError(
            f"Failed to flush session {session_id} after retries."
        ) from exc
      return

    async with buffer.lock:
      buffer.flush_in_progress = False

  async def _persist_with_retry(
      self, session: Session, batch: list[Event], session_id: str
  ) -> None:
    attempt = 0
    while True:
      attempt += 1
      try:
        await self._persist_batch(session, batch)
        return
      except Exception as exc:  # noqa: BLE001 - retryable vs permanent
        if not is_retryable_error(exc) or attempt >= self._max_retry_attempts:
          logger.error(
              "session_flush_failed session_id=%s events=%s attempt=%s"
              " error=%s",
              session_id,
              len(batch),
              attempt,
              type(exc).__name__,
          )
          raise
        delay = self._retry_base_delay_seconds * (2 ** (attempt - 1))
        delay += random.uniform(0.0, self._retry_base_delay_seconds)
        await self._sleeper(delay)

  async def _persist_batch(self, session: Session, events: list[Event]) -> None:
    """Persists a batch of events for one session in a single transaction."""
    session_ref = self._get_sessions_ref(
        session.app_name, session.user_id
    ).document(session.id)
    app_ref = self._app_state_ref(session.app_name)
    user_ref = self._user_state_ref(session.app_name, session.user_id)

    agg: dict[str, dict[str, Any]] = {"app": {}, "user": {}, "session": {}}
    for event in events:
      delta = (
          event.actions.state_delta
          if event.actions and event.actions.state_delta
          else {}
      )
      scoped = _session_util.extract_state_delta(delta)
      agg["app"].update(scoped["app"])
      agg["user"].update(scoped["user"])
      agg["session"].update(scoped["session"])
    has_app, has_user = bool(agg["app"]), bool(agg["user"])

    async def _append_txn(transaction: Any) -> int:
      snap = await session_ref.get(transaction=transaction)
      if not snap.exists:
        raise ValueError(f"Session {session.id} not found.")
      doc = snap.to_dict() or {}
      if doc.get("status") == "DELETING":
        raise ValueError(f"Session {session.id} is currently being deleted.")
      current_revision = doc.get("revision", 0)
      marker = getattr(session, "_storage_update_marker", None)
      if marker is not None and marker != str(current_revision):
        raise ValueError(
            "The session has been modified in storage since it was loaded."
            " Please reload the session before appending more events."
        )

      app_snap = await app_ref.get(transaction=transaction) if has_app else None
      user_snap = (
          await user_ref.get(transaction=transaction) if has_user else None
      )

      if has_app:
        current = app_snap.to_dict() if app_snap.exists else {}
        current.update(agg["app"])
        transaction.set(app_ref, current, merge=True)
      if has_user:
        current = user_snap.to_dict() if user_snap.exists else {}
        current.update(agg["user"])
        transaction.set(user_ref, current, merge=True)
      for key, value in agg["session"].items():
        session.state[key] = value

      for event in events:
        event_ref = session_ref.collection(self.events_collection).document(
            event.id
        )
        transaction.set(
            event_ref,
            {
                "event_data": event.model_dump(exclude_none=True, mode="json"),
                # The event's own timestamp (not SERVER_TIMESTAMP) so order is
                # preserved within a batch that shares a commit time.
                "timestamp": datetime.fromtimestamp(
                    event.timestamp, tz=timezone.utc
                ),
                "appName": session.app_name,
                "userId": session.user_id,
            },
        )

      new_revision = current_revision + 1
      session_only_state = {
          k: v
          for k, v in session.state.items()
          if not k.startswith(State.APP_PREFIX)
          and not k.startswith(State.USER_PREFIX)
          and not k.startswith(State.TEMP_PREFIX)
      }
      transaction.update(
          session_ref,
          {
              "state": session_only_state,
              "updateTime": self._firestore.SERVER_TIMESTAMP,
              "revision": new_revision,
          },
      )
      return new_revision

    new_revision = await self._transactional(_append_txn)(
        self.client.transaction()
    )
    session._storage_update_marker = str(new_revision)
    if events:
      session.last_update_time = events[-1].timestamp

  # -- periodic flushing -----------------------------------------------------

  async def start(self) -> None:
    """Starts the background periodic-flush task (idempotent)."""
    if self._task is not None and not self._task.done():
      return
    self._task = asyncio.create_task(self._periodic_flush_loop())

  async def stop(self) -> None:
    """Stops the background task and performs a final flush (idempotent)."""
    task = self._task
    self._task = None
    if task is not None:
      task.cancel()
      try:
        await task
      except asyncio.CancelledError:
        pass
    await self.flush_all()

  async def close(self) -> None:
    """Closes the underlying Firestore AsyncClient."""
    closer = getattr(self.client, "close", None)
    if closer is not None:
      result = closer()
      if asyncio.iscoroutine(result):
        await result

  async def _periodic_flush_loop(self) -> None:
    try:
      while True:
        await self._sleeper(self._check_interval)
        await self._flush_due()
    except asyncio.CancelledError:
      raise

  async def _flush_due(self) -> list[asyncio.Task[None]]:
    now = self._clock()
    tasks: list[asyncio.Task[None]] = []
    for session_id, buffer in list(self._buffers.items()):
      due = (now - buffer.last_flush_monotonic) >= self._flush_interval_seconds
      if buffer.pending_events and due:
        tasks.append(
            asyncio.create_task(self._safe_background_flush(session_id))
        )
    return tasks

  async def _safe_background_flush(self, session_id: str) -> None:
    try:
      await self._flush(session_id, explicit=False)
    except Exception:  # noqa: BLE001 - background task must not raise unhandled
      logger.exception("background_flush_failed session_id=%s", session_id)

  # -- internal helpers ------------------------------------------------------

  async def _get_or_create_buffer(self, session: Session) -> _SessionBuffer:
    async with self._buffers_guard:
      buffer = self._buffers.get(session.id)
      if buffer is None:
        buffer = _SessionBuffer(last_flush_monotonic=self._clock())
        self._buffers[session.id] = buffer
      self._session_refs[session.id] = session
    return buffer

  def _merge_buffered(self, session: Session) -> Session:
    buffer = self._buffers.get(session.id)
    if buffer is None or not buffer.pending_events:
      return session
    seen = {e.id for e in session.events}
    merged = list(session.events)
    for event in list(buffer.pending_events):
      if event.id not in seen:
        merged.append(event)
        seen.add(event.id)
    merged.sort(key=lambda e: (e.timestamp or 0.0))
    session.events = merged
    return session
