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
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING
import uuid

from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.events import Event
from google.adk.sessions import _session_util
from google.adk.sessions import Session
from google.adk.sessions import State
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.base_session_service import ListSessionsResponse
import orjson
from typing_extensions import override

from .utils import _json_serializer

if TYPE_CHECKING:
  from pymongo import AsyncMongoClient
  from pymongo.asynchronous.collection import AsyncCollection

logger = logging.getLogger("google_adk." + __name__)


def _normalize_value(value: Any) -> Any:
  """Convert a value through the shared JSON serializer into BSON types."""
  return orjson.loads(orjson.dumps(value, default=_json_serializer))


class MongoKeys:
  """Helper to generate composite keys for Mongo-backed storage."""

  @staticmethod
  def session(app_name: str, user_id: str, session_id: str) -> str:
    return f"session:{app_name}:{user_id}:{session_id}"

  @staticmethod
  def app_state(app_name: str) -> str:
    return f"{State.APP_PREFIX}{app_name}"

  @staticmethod
  def user_state(app_name: str, user_id: str) -> str:
    return f"{State.USER_PREFIX}{app_name}:{user_id}"


class MongoSessionService(BaseSessionService):
  """Session service backed by MongoDB.

  The MongoDB client is bound to the event loop where it is constructed. When
  loop affinity matters, construct the client on the target loop and pass it
  through ``client=``.
  """

  def __init__(
      self,
      client: Optional[AsyncMongoClient] = None,
      connection_string: Optional[str] = None,
      database_name: str = "adk_sessions_db",
      session_collection: str = "sessions",
      state_collection: str = "session_state",
      default_app_name: Optional[str] = "adk-mongo-session-service",
  ) -> None:
    try:
      import pymongo
    except ImportError as exc:
      raise ImportError(
          "pymongo is required to use MongoSessionService. "
          "Install it with: pip install google-adk-community[mongo]"
      ) from exc

    if bool(connection_string) == bool(client):
      raise ValueError(
          "Provide either 'connection_string' or 'client', but not both."
      )
    self._client = client or pymongo.AsyncMongoClient(connection_string)
    db = self._client[database_name]
    self._sessions: AsyncCollection = db[session_collection]
    self._kv: AsyncCollection = db[state_collection]
    self._ascending = pymongo.ASCENDING
    self._descending = pymongo.DESCENDING
    self._duplicate_key_error = pymongo.errors.DuplicateKeyError
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

    session_id = (session_id or "").strip() or str(uuid.uuid4())
    doc_id = MongoKeys.session(app_name, user_id, session_id)

    state_deltas = _session_util.extract_state_delta(state or {})
    await self._apply_state_delta(
        app_name, user_id, state_deltas.get("app"), state_deltas.get("user")
    )

    session_doc = {
        "_id": doc_id,
        "app_name": app_name,
        "user_id": user_id,
        "id": session_id,
        "state": {
            key: _normalize_value(value)
            for key, value in state_deltas.get("session", {}).items()
        },
        "events": [],
        "last_update_time": time.time(),
    }

    try:
      await self._sessions.insert_one(session_doc)
    except self._duplicate_key_error as exc:
      raise AlreadyExistsError(
          f"Session with id {session_id} already exists."
      ) from exc

    session = self._doc_to_session(session_doc)
    app_doc, user_doc = await self._fetch_state(
        session.app_name, session.user_id
    )
    return self._merge_state(session, app_doc, user_doc)

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

    doc = await self._sessions.find_one(
        {"_id": MongoKeys.session(app_name, user_id, session_id)}
    )
    if not doc:
      return None

    session = self._doc_to_session(doc)
    session = self._apply_event_filters(session, config)
    app_doc, user_doc = await self._fetch_state(
        session.app_name, session.user_id
    )
    return self._merge_state(session, app_doc, user_doc)

  @override
  async def list_sessions(
      self,
      *,
      app_name: str,
      user_id: Optional[str] = None,
  ) -> ListSessionsResponse:
    """List all matching sessions, materialized because no pagination is exposed."""
    app_name = self._resolve_app_name(app_name)
    await self._ensure_indexes()

    filters: dict[str, Any] = {"app_name": app_name}
    if user_id is not None:
      filters["user_id"] = user_id

    cursor = self._sessions.find(filters, projection={"events": False})
    # NOTE: BaseSessionService expects the full list to be returned, so we have
    # to materialize the entire cursor which may load many sessions into memory.
    docs = await cursor.to_list(length=None)
    if len(docs) > 1000:
      logger.warning(
          "Loading a large number of sessions (%d) into memory for app '%s'.",
          len(docs),
          app_name,
      )

    sessions: list[Session] = [self._doc_to_session(doc) for doc in docs]
    app_doc = await self._kv.find_one({"_id": MongoKeys.app_state(app_name)})
    if user_id is not None:
      user_docs = {}
      user_doc = await self._kv.find_one(
          {"_id": MongoKeys.user_state(app_name, user_id)}
      )
      if user_doc:
        user_docs[user_doc["_id"]] = user_doc
    else:
      user_ids = {session.user_id for session in sessions}
      user_docs = {}
      if user_ids:
        cursor = self._kv.find({
            "_id": {
                "$in": [
                    MongoKeys.user_state(app_name, current_user_id)
                    for current_user_id in user_ids
                ]
            }
        })
        user_docs = {
            doc["_id"]: doc for doc in await cursor.to_list(length=None)
        }

    sessions = [
        self._merge_state(
            session,
            app_doc,
            user_docs.get(MongoKeys.user_state(app_name, session.user_id)),
        )
        for session in sessions
    ]

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
    for key in list(session.state):
      if key.startswith(State.TEMP_PREFIX):
        del session.state[key]

    state_delta = event.actions.state_delta if event.actions else None
    state_deltas = _session_util.extract_state_delta(state_delta or {})

    await self._apply_state_delta(
        session.app_name,
        session.user_id,
        state_deltas.get("app"),
        state_deltas.get("user"),
    )

    updates: dict[str, Any] = {
        "$push": {
            "events": _normalize_value(event.model_dump(exclude_none=True))
        },
        "$set": {"last_update_time": event.timestamp},
    }

    session_state_set = {
        f"state.{key}": _normalize_value(value)
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
          [
              ("app_name", self._ascending),
              ("user_id", self._ascending),
              ("id", self._ascending),
          ],
          unique=True,
          name="session_identity_idx",
      )
      await self._sessions.create_index(
          [
              ("app_name", self._ascending),
              ("user_id", self._ascending),
              ("last_update_time", self._descending),
          ],
          name="session_last_update_idx",
      )
      self._indexes_built = True

  async def _fetch_state(
      self, app_name: str, user_id: str
  ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    app_doc, user_doc = await asyncio.gather(
        self._kv.find_one({"_id": MongoKeys.app_state(app_name)}),
        self._kv.find_one({"_id": MongoKeys.user_state(app_name, user_id)}),
    )
    return app_doc, user_doc

  def _merge_state(
      self,
      session: Session,
      app_doc: Optional[dict[str, Any]],
      user_doc: Optional[dict[str, Any]],
  ) -> Session:
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
      app_state_delta: Optional[dict[str, Any]],
      user_state_delta: Optional[dict[str, Any]],
  ) -> None:
    tasks = []
    if app_state_delta:
      tasks.append(
          self._update_state_document(
              MongoKeys.app_state(app_name), app_state_delta
          )
      )
    if user_state_delta:
      tasks.append(
          self._update_state_document(
              MongoKeys.user_state(app_name, user_id), user_state_delta
          )
      )
    if tasks:
      await asyncio.gather(*tasks)

  async def _update_state_document(
      self,
      key: str,
      delta: dict[str, Any],
  ) -> None:
    set_ops = {
        f"state.{k}": _normalize_value(v)
        for k, v in delta.items()
        if v is not None
    }
    unset_ops = {f"state.{k}": "" for k, v in delta.items() if v is None}

    update: dict[str, Any] = {}
    if set_ops:
      update["$set"] = set_ops
    if unset_ops:
      update["$unset"] = unset_ops
    if not update:
      return

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
        id=doc.get("id"),
        app_name=doc.get("app_name"),
        user_id=doc.get("user_id"),
        state=doc.get("state", {}),
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
