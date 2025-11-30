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
import asyncio
import logging
import time
from typing import Any
from typing import Optional
import uuid

from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.events import Event
from google.adk.sessions import _session_util
from google.adk.sessions import Session
from google.adk.sessions import State
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.base_session_service import ListSessionsResponse
from pymongo import ASCENDING
from pymongo import AsyncMongoClient
from pymongo import DESCENDING
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import DuplicateKeyError
from typing_extensions import override

logger = logging.getLogger("google_adk." + __name__)


class MongoKeys:
  """Helper to generate composite keys for Mongo-backed storage."""

  @staticmethod
  def session(app_name: str, user_id: str, session_id: str) -> str:
    return f"session::{app_name}::{user_id}::{session_id}"

  @staticmethod
  def app_state(app_name: str) -> str:
    return f"{State.APP_PREFIX}{app_name}"

  @staticmethod
  def user_state(app_name: str, user_id: str) -> str:
    return f"{State.USER_PREFIX}{app_name}:{user_id}"


class MongoSessionService(BaseSessionService):
  """Session service backed by MongoDB."""

  def __init__(
      self,
      client: Optional[AsyncMongoClient] = None,
      connection_string: Optional[str] = None,
      database_name: Optional[str] = "adk_sessions_db",
      session_collection: str = "sessions",
      state_collection: str = "session_state",
      default_app_name: Optional[str] = "adk-mongo-session-service",
  ) -> None:
    if bool(connection_string) == bool(client):
      raise ValueError(
          "Provide either 'connection_string' or 'client', but not both."
      )
    self._client = client or AsyncMongoClient(connection_string)
    self._sessions: AsyncCollection = self._client[database_name][
        session_collection
    ]
    self._kv: AsyncCollection = self._client[database_name][state_collection]
    self._default_app_name = default_app_name
    self._indexes_built = False
    self._indexes_lock = asyncio.Lock()

  @override
  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    app_name = self._resolve_app_name(app_name)
    await self._ensure_indexes()

    session_id = (
        session_id.strip()
        if session_id and session_id.strip()
        else str(uuid.uuid4())
    )
    doc_id = MongoKeys.session(app_name, user_id, session_id)

    state_deltas = _session_util.extract_state_delta(state or {})
    await self._apply_state_delta(
        app_name, user_id, state_deltas["app"], state_deltas["user"]
    )

    session_doc = {
        "_id": doc_id,
        "app_name": app_name,
        "user_id": user_id,
        "id": session_id,
        "state": state_deltas["session"] or {},
        "events": [],
        "last_update_time": time.time(),
    }

    try:
      await self._sessions.insert_one(session_doc)
    except DuplicateKeyError as exc:
      raise AlreadyExistsError(
          f"Session with id {session_id} already exists."
      ) from exc

    session = self._doc_to_session(session_doc)
    return await self._merge_state(session)

  @override
  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    app_name = self._resolve_app_name(app_name)
    await self._ensure_indexes()

    doc = await self._sessions.find_one({
        "_id": MongoKeys.session(app_name, user_id, session_id),
        "app_name": app_name,
        "user_id": user_id,
    })
    if not doc:
      return None

    session = self._doc_to_session(doc)
    session = self._apply_event_filters(session, config)
    return await self._merge_state(session)

  @override
  async def list_sessions(
      self,
      *,
      app_name: str,
      user_id: Optional[str] = None,
  ) -> ListSessionsResponse:
    app_name = self._resolve_app_name(app_name)
    await self._ensure_indexes()

    filters: dict[str, Any] = {"app_name": app_name}
    if user_id is not None:
      filters["user_id"] = user_id

    cursor = self._sessions.find(filters, projection={"events": False})
    docs = await cursor.to_list(length=None)

    sessions: list[Session] = []
    for doc in docs:
      doc.setdefault("events", [])
      session = self._doc_to_session(doc)
      merged = await self._merge_state(session)
      merged.events = []
      sessions.append(merged)

    return ListSessionsResponse(sessions=sessions)

  @override
  async def delete_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
  ) -> None:
    app_name = self._resolve_app_name(app_name)
    await self._ensure_indexes()

    await self._sessions.delete_one(
        {"_id": MongoKeys.session(app_name, user_id, session_id)}
    )

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    if event.partial:
      return event

    await self._ensure_indexes()

    event = await super().append_event(session, event)
    session.last_update_time = event.timestamp

    state_delta = event.actions.state_delta if event.actions else None
    state_deltas = _session_util.extract_state_delta(state_delta or {})

    await self._apply_state_delta(
        session.app_name,
        session.user_id,
        state_deltas["app"],
        state_deltas["user"],
    )

    updates: dict[str, Any] = {
        "$push": {"events": event.model_dump(mode="json", exclude_none=True)},
        "$set": {"last_update_time": event.timestamp},
    }

    session_state_set = {
        f"state.{key}": value
        for key, value in state_deltas["session"].items()
        if value is not None
    }
    session_state_unset = {
        f"state.{key}": ""
        for key, value in state_deltas["session"].items()
        if value is None
    }

    if session_state_set:
      updates.setdefault("$set", {}).update(session_state_set)
    if session_state_unset:
      updates["$unset"] = session_state_unset

    result = await self._sessions.update_one(
        {
            "_id": MongoKeys.session(
                session.app_name, session.user_id, session.id
            )
        },
        updates,
    )
    if result.matched_count == 0:
      logger.warning(
          "Failed to append event: session %s/%s/%s not found in storage",
          session.app_name,
          session.user_id,
          session.id,
      )

    return event

  async def _ensure_indexes(self) -> None:
    if self._indexes_built:
      return
    async with self._indexes_lock:
      if self._indexes_built:
        return
      await self._sessions.create_index(
          [("app_name", ASCENDING), ("user_id", ASCENDING), ("id", ASCENDING)],
          unique=True,
          name="session_identity_idx",
      )
      await self._sessions.create_index(
          [
              ("app_name", ASCENDING),
              ("user_id", ASCENDING),
              ("last_update_time", DESCENDING),
          ],
          name="session_last_update_idx",
      )
      self._indexes_built = True

  async def _merge_state(self, session: Session) -> Session:
    app_doc, user_doc = await asyncio.gather(
        self._kv.find_one({"_id": MongoKeys.app_state(session.app_name)}),
        self._kv.find_one(
            {"_id": MongoKeys.user_state(session.app_name, session.user_id)}
        ),
    )

    merged_state = dict(session.state)
    if app_doc and app_doc.get("state"):
      for key, value in app_doc["state"].items():
        merged_state[State.APP_PREFIX + key] = value
    if user_doc and user_doc.get("state"):
      for key, value in user_doc["state"].items():
        merged_state[State.USER_PREFIX + key] = value

    return session.model_copy(update={"state": merged_state})

  async def _apply_state_delta(
      self,
      app_name: str,
      user_id: str,
      app_state_delta: dict[str, Any],
      user_state_delta: dict[str, Any],
  ) -> None:
    if app_state_delta:
      await self._update_state_document(
          MongoKeys.app_state(app_name),
          app_state_delta,
      )
    if user_state_delta:
      await self._update_state_document(
          MongoKeys.user_state(app_name, user_id),
          user_state_delta,
      )

  async def _update_state_document(
      self,
      key: str,
      delta: dict[str, Any],
  ) -> None:
    set_ops = {
        f"state.{key}": value
        for key, value in delta.items()
        if value is not None
    }
    unset_ops = {
        f"state.{key}": "" for key, value in delta.items() if value is None
    }

    update: dict[str, Any] = {}
    if set_ops:
      update["$set"] = set_ops
    if unset_ops:
      update["$unset"] = unset_ops
    if not update:
      return

    update.setdefault("$setOnInsert", {}).update({"_id": key})
    await self._kv.update_one({"_id": key}, update, upsert=True)

  def _apply_event_filters(
      self, session: Session, config: Optional[GetSessionConfig]
  ) -> Session:
    if not config:
      return session
    events = session.events

    if config.after_timestamp is not None:
      events = [e for e in events if e.timestamp > config.after_timestamp]
    if config.num_recent_events is not None:
      events = events[-config.num_recent_events :]

    return session.model_copy(update={"events": events})

  def _doc_to_session(self, doc: dict[str, Any]) -> Session:
    events = [Event.model_validate(e) for e in doc.get("events", [])]
    return Session(
        id=doc["id"],
        app_name=doc["app_name"],
        user_id=doc["user_id"],
        state=doc.get("state", {}) or {},
        events=events,
        last_update_time=doc.get("last_update_time", 0.0),
    )

  def _resolve_app_name(self, app_name: Optional[str]) -> str:
    resolved = app_name or self._default_app_name
    if not resolved:
      raise ValueError(
          "app_name must be provided either in the call or in default_app_name."
      )
    return resolved
