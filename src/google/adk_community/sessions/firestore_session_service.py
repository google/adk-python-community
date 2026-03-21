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

"""Firestore-backed session service for Google ADK.

Provides persistent, serverless session storage using Google Cloud
Firestore.  Well-suited for Cloud Run, Cloud Functions, or any GCP
environment where managing a SQL database is undesirable.

Firestore collection layout::

    {prefix}adk_app_states/{app_name}
    {prefix}adk_user_states/{app_name}_{user_id}
    {prefix}adk_sessions/{session_id}
        -> subcollection: events/{event_id}

Requires the ``google-cloud-firestore`` package::

    pip install google-cloud-firestore
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Optional
import uuid

from typing_extensions import override

from google.adk.events.event import Event
from google.adk.sessions.base_session_service import (
    BaseSessionService,
    GetSessionConfig,
    ListSessionsResponse,
)
from google.adk.sessions.session import Session
from google.adk.sessions.state import State

logger = logging.getLogger("google_adk." + __name__)

_APP_STATES_COLLECTION = "adk_app_states"
_USER_STATES_COLLECTION = "adk_user_states"
_SESSIONS_COLLECTION = "adk_sessions"
_EVENTS_SUBCOLLECTION = "events"

_FIELD_APP_NAME = "app_name"
_FIELD_USER_ID = "user_id"
_FIELD_STATE = "state"
_FIELD_CREATE_TIME = "create_time"
_FIELD_UPDATE_TIME = "update_time"
_FIELD_EVENT_DATA = "event_data"
_FIELD_TIMESTAMP = "timestamp"
_FIELD_INVOCATION_ID = "invocation_id"

_BATCH_DELETE_LIMIT = 500


def _user_state_doc_id(app_name: str, user_id: str) -> str:
  return f"{app_name}_{user_id}"


def _extract_state_delta(
    state: Optional[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
  """Splits a state dict into app / user / session buckets."""
  deltas: dict[str, dict[str, Any]] = {"app": {}, "user": {}, "session": {}}
  if not state:
    return deltas
  for key, value in state.items():
    if key.startswith(State.APP_PREFIX):
      deltas["app"][key.removeprefix(State.APP_PREFIX)] = value
    elif key.startswith(State.USER_PREFIX):
      deltas["user"][key.removeprefix(State.USER_PREFIX)] = value
    elif not key.startswith(State.TEMP_PREFIX):
      deltas["session"][key] = value
  return deltas


def _merge_state(
    app_state: dict[str, Any],
    user_state: dict[str, Any],
    session_state: dict[str, Any],
) -> dict[str, Any]:
  """Combines app / user / session state into the flat dict ADK expects."""
  merged = copy.deepcopy(session_state)
  for key, value in app_state.items():
    merged[State.APP_PREFIX + key] = value
  for key, value in user_state.items():
    merged[State.USER_PREFIX + key] = value
  return merged


class FirestoreSessionService(BaseSessionService):
  """A session service backed by Google Cloud Firestore.

  Args:
    project: GCP project ID.  ``None`` uses Application Default
        Credentials.
    database: Firestore database ID.  Defaults to ``"(default)"``.
    collection_prefix: Optional prefix for all collection names (useful
        for multi-tenant setups or test isolation).
  """

  def __init__(
      self,
      *,
      project: Optional[str] = None,
      database: str = "(default)",
      collection_prefix: str = "",
  ):
    try:
      from google.cloud.firestore_v1 import AsyncClient  # noqa: F401
    except ImportError as e:
      raise ImportError(
          "FirestoreSessionService requires google-cloud-firestore. "
          "Install it with: pip install google-cloud-firestore"
      ) from e

    self._db: Any = AsyncClient(project=project, database=database)
    self._prefix = collection_prefix

  # -- collection helpers --------------------------------------------------

  def _col_app_states(self):
    return self._db.collection(f"{self._prefix}{_APP_STATES_COLLECTION}")

  def _col_user_states(self):
    return self._db.collection(f"{self._prefix}{_USER_STATES_COLLECTION}")

  def _col_sessions(self):
    return self._db.collection(f"{self._prefix}{_SESSIONS_COLLECTION}")

  def _events_col(self, session_id: str):
    return (
        self._col_sessions()
        .document(session_id)
        .collection(_EVENTS_SUBCOLLECTION)
    )

  # -- state helpers -------------------------------------------------------

  async def _get_app_state(self, app_name: str) -> dict[str, Any]:
    doc = await self._col_app_states().document(app_name).get()
    if doc.exists:
      return doc.to_dict().get(_FIELD_STATE, {})
    return {}

  async def _get_user_state(
      self, app_name: str, user_id: str
  ) -> dict[str, Any]:
    doc_id = _user_state_doc_id(app_name, user_id)
    doc = await self._col_user_states().document(doc_id).get()
    if doc.exists:
      return doc.to_dict().get(_FIELD_STATE, {})
    return {}

  async def _update_app_state_transactional(
      self, app_name: str, delta: dict[str, Any]
  ) -> dict[str, Any]:
    """Atomically applies *delta* to app state inside a transaction."""
    doc_ref = self._col_app_states().document(app_name)

    @self._db.async_transactional
    async def _txn(transaction):
      snap = await doc_ref.get(transaction=transaction)
      current = snap.to_dict().get(_FIELD_STATE, {}) if snap.exists else {}
      current.update(delta)
      transaction.set(doc_ref, {_FIELD_STATE: current}, merge=True)
      return current

    transaction = self._db.transaction()
    return await _txn(transaction)

  async def _update_user_state_transactional(
      self, app_name: str, user_id: str, delta: dict[str, Any]
  ) -> dict[str, Any]:
    """Atomically applies *delta* to user state inside a transaction."""
    doc_id = _user_state_doc_id(app_name, user_id)
    doc_ref = self._col_user_states().document(doc_id)

    @self._db.async_transactional
    async def _txn(transaction):
      snap = await doc_ref.get(transaction=transaction)
      current = snap.to_dict().get(_FIELD_STATE, {}) if snap.exists else {}
      current.update(delta)
      transaction.set(
          doc_ref,
          {
              _FIELD_APP_NAME: app_name,
              _FIELD_USER_ID: user_id,
              _FIELD_STATE: current,
          },
          merge=True,
      )
      return current

    transaction = self._db.transaction()
    return await _txn(transaction)

  # -- CRUD ----------------------------------------------------------------

  @override
  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    session_id = (
        session_id.strip()
        if session_id and session_id.strip()
        else str(uuid.uuid4())
    )

    existing = await self._col_sessions().document(session_id).get()
    if existing.exists:
      raise ValueError(
          f"Session with id {session_id} already exists."
      )

    deltas = _extract_state_delta(state)
    app_state_delta = deltas["app"]
    user_state_delta = deltas["user"]
    session_state = deltas["session"]

    # Transactional state updates; reuse returned state to avoid re-read.
    app_state = (
        await self._update_app_state_transactional(app_name, app_state_delta)
        if app_state_delta
        else await self._get_app_state(app_name)
    )
    user_state = (
        await self._update_user_state_transactional(
            app_name, user_id, user_state_delta
        )
        if user_state_delta
        else await self._get_user_state(app_name, user_id)
    )

    now = time.time()
    await self._col_sessions().document(session_id).set({
        _FIELD_APP_NAME: app_name,
        _FIELD_USER_ID: user_id,
        _FIELD_STATE: session_state,
        _FIELD_CREATE_TIME: now,
        _FIELD_UPDATE_TIME: now,
    })

    merged = _merge_state(app_state, user_state, session_state)
    return Session(
        app_name=app_name,
        user_id=user_id,
        id=session_id,
        state=merged,
        last_update_time=now,
    )

  @override
  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    doc = await self._col_sessions().document(session_id).get()
    if not doc.exists:
      return None

    data = doc.to_dict()
    if data.get(_FIELD_APP_NAME) != app_name:
      return None
    if data.get(_FIELD_USER_ID) != user_id:
      return None

    session_state = data.get(_FIELD_STATE, {})

    # Build events query with server-side filtering.
    events_query = self._events_col(session_id).order_by(_FIELD_TIMESTAMP)

    if config and config.after_timestamp:
      events_query = events_query.where(
          filter=self._db.field_filter(
              _FIELD_TIMESTAMP, ">=", config.after_timestamp
          )
      )

    if config and config.num_recent_events:
      events_query = events_query.limit_to_last(
          config.num_recent_events
      )

    events: list[Event] = []
    async for event_doc in events_query.stream():
      raw = event_doc.to_dict().get(_FIELD_EVENT_DATA, {})
      if raw:
        events.append(Event.model_validate(raw))

    app_state = await self._get_app_state(app_name)
    user_state = await self._get_user_state(app_name, user_id)
    merged = _merge_state(app_state, user_state, session_state)

    return Session(
        app_name=app_name,
        user_id=user_id,
        id=session_id,
        state=merged,
        events=events,
        last_update_time=data.get(_FIELD_UPDATE_TIME, 0.0),
    )

  @override
  async def list_sessions(
      self, *, app_name: str, user_id: str
  ) -> ListSessionsResponse:
    query = self._col_sessions().where(
        filter=self._db.field_filter(_FIELD_APP_NAME, "==", app_name)
    )
    query = query.where(
        filter=self._db.field_filter(_FIELD_USER_ID, "==", user_id)
    )

    # Fetch shared state once, outside the loop.
    app_state = await self._get_app_state(app_name)
    user_state = await self._get_user_state(app_name, user_id)

    sessions: list[Session] = []
    async for doc in query.stream():
      data = doc.to_dict()
      session_state = data.get(_FIELD_STATE, {})
      merged = _merge_state(app_state, user_state, session_state)
      sessions.append(
          Session(
              app_name=app_name,
              user_id=data.get(_FIELD_USER_ID, ""),
              id=doc.id,
              state=merged,
              last_update_time=data.get(_FIELD_UPDATE_TIME, 0.0),
          )
      )

    return ListSessionsResponse(sessions=sessions)

  @override
  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    session_ref = self._col_sessions().document(session_id)
    doc = await session_ref.get()
    if not doc.exists:
      return

    # Batch-delete events in chunks of _BATCH_DELETE_LIMIT.
    events_ref = session_ref.collection(_EVENTS_SUBCOLLECTION)
    batch = self._db.batch()
    count = 0
    async for event_doc in events_ref.stream():
      batch.delete(event_doc.reference)
      count += 1
      if count >= _BATCH_DELETE_LIMIT:
        await batch.commit()
        batch = self._db.batch()
        count = 0
    if count:
      await batch.commit()

    await session_ref.delete()

  @override
  async def append_event(
      self, session: Session, event: Event
  ) -> Event:
    if event.partial:
      return event

    app_name = session.app_name
    user_id = session.user_id
    session_id = session.id

    session_ref = self._col_sessions().document(session_id)
    doc = await session_ref.get()
    if not doc.exists:
      logger.warning(
          "Cannot append event: session %s not found.", session_id
      )
      return event

    await super().append_event(session=session, event=event)
    session.last_update_time = event.timestamp

    if event.actions and event.actions.state_delta:
      deltas = _extract_state_delta(event.actions.state_delta)

      if deltas["app"]:
        await self._update_app_state_transactional(
            app_name, deltas["app"]
        )
      if deltas["user"]:
        await self._update_user_state_transactional(
            app_name, user_id, deltas["user"]
        )
      if deltas["session"]:
        stored_state = doc.to_dict().get(_FIELD_STATE, {})
        stored_state.update(deltas["session"])
        await session_ref.update({_FIELD_STATE: stored_state})

    event_data = event.model_dump(exclude_none=True, mode="json")
    await self._events_col(session_id).document(event.id).set({
        _FIELD_EVENT_DATA: event_data,
        _FIELD_TIMESTAMP: event.timestamp,
        _FIELD_INVOCATION_ID: event.invocation_id,
    })

    await session_ref.update({_FIELD_UPDATE_TIME: event.timestamp})
    return event

  async def close(self) -> None:
    """Closes the underlying Firestore client."""
    self._db.close()

  async def __aenter__(self) -> FirestoreSessionService:
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    await self.close()
