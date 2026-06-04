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
import copy
import json
import logging
import time
from typing import Any
from typing import Optional
import uuid

from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.events.event import Event
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.base_session_service import ListSessionsResponse
from google.adk.sessions.session import Session
from google.adk.sessions.state import State
from typing_extensions import override

logger = logging.getLogger('google_adk.' + __name__)

_SEP = '||'

_SESSIONS_SCHEMA = {
    'fields': [
        {'name': 'app_name', 'type': 'string', 'facet': True},
        {'name': 'user_id', 'type': 'string', 'facet': True},
        {'name': 'session_id', 'type': 'string', 'facet': True},
        {'name': 'state', 'type': 'string', 'index': False},
        {'name': 'update_time', 'type': 'float'},
    ],
}

_EVENTS_SCHEMA = {
    'fields': [
        {'name': 'app_name', 'type': 'string', 'facet': True},
        {'name': 'user_id', 'type': 'string', 'facet': True},
        {'name': 'session_id', 'type': 'string', 'facet': True},
        {'name': 'timestamp', 'type': 'float'},
        {'name': 'event_data', 'type': 'string', 'index': False},
    ],
}

_APP_STATES_SCHEMA = {
    'fields': [
        {'name': 'app_name', 'type': 'string', 'facet': True},
        {'name': 'state', 'type': 'string', 'index': False},
        {'name': 'update_time', 'type': 'float'},
    ],
}

_USER_STATES_SCHEMA = {
    'fields': [
        {'name': 'app_name', 'type': 'string', 'facet': True},
        {'name': 'user_id', 'type': 'string', 'facet': True},
        {'name': 'state', 'type': 'string', 'index': False},
        {'name': 'update_time', 'type': 'float'},
    ],
}


def _validate_no_sep(*values: str) -> None:
  for v in values:
    if _SEP in v:
      raise ValueError(
          f'app_name, user_id, and session_id must not contain {_SEP!r}.'
      )


def _session_doc_id(app_name: str, user_id: str, session_id: str) -> str:
  return f'{app_name}{_SEP}{user_id}{_SEP}{session_id}'


def _event_doc_id(
    app_name: str, user_id: str, session_id: str, event_id: str
) -> str:
  return f'{app_name}{_SEP}{user_id}{_SEP}{session_id}{_SEP}{event_id}'


def _user_state_doc_id(app_name: str, user_id: str) -> str:
  return f'{app_name}{_SEP}{user_id}'


def _extract_state_delta(
    state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
  """Split a flat state dict into app, user, and session-scoped sub-dicts."""
  deltas: dict[str, dict[str, Any]] = {'app': {}, 'user': {}, 'session': {}}
  for key, value in state.items():
    if key.startswith(State.APP_PREFIX):
      deltas['app'][key.removeprefix(State.APP_PREFIX)] = value
    elif key.startswith(State.USER_PREFIX):
      deltas['user'][key.removeprefix(State.USER_PREFIX)] = value
    elif not key.startswith(State.TEMP_PREFIX):
      deltas['session'][key] = value
  return deltas


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


class TypesenseSessionService(BaseSessionService):
  """Persistent session storage backed by Typesense.

  Stores sessions, events, and shared app/user state in four Typesense
  collections that are created automatically on first use:

  * ``{prefix}_sessions``
  * ``{prefix}_events``
  * ``{prefix}_app_states``
  * ``{prefix}_user_states``

  **Installation**::

      pip install google-adk-community[typesense]

  **Constraints**

  * ``app_name``, ``user_id``, and ``session_id`` must not contain ``'||'``
    (used as an internal document-ID separator).
  * App and user state updates are serialized per-key within a single process.
    Multi-process deployments sharing the same Typesense instance may still
    lose concurrent state updates because Typesense has no native transactions.
  * Per-key asyncio locks are retained for the lifetime of the service
    instance and are not evicted. For workloads with a bounded set of
    apps and users this is fine; if the process sees an unbounded stream
    of unique IDs without restarts, the dicts grow monotonically.
  """

  def __init__(
      self,
      *,
      host: str = 'localhost',
      port: int = 8108,
      protocol: str = 'http',
      api_key: str,
      collection_prefix: str = 'adk',
      connection_timeout_seconds: int = 5,
  ):
    try:
      import typesense
      from typesense.exceptions import ObjectAlreadyExists
      from typesense.exceptions import ObjectNotFound

      self._typesense = typesense
      self._ObjectNotFound = ObjectNotFound
      self._ObjectAlreadyExists = ObjectAlreadyExists
    except ImportError as e:
      raise ImportError(
          'TypesenseSessionService requires the typesense package.'
          ' Install it with: pip install google-adk-community[typesense]'
      ) from e

    self._client = typesense.Client({
        'nodes': [{'host': host, 'port': str(port), 'protocol': protocol}],
        'api_key': api_key,
        'connection_timeout_seconds': connection_timeout_seconds,
    })

    self._sessions_col = f'{collection_prefix}_sessions'
    self._events_col = f'{collection_prefix}_events'
    self._app_states_col = f'{collection_prefix}_app_states'
    self._user_states_col = f'{collection_prefix}_user_states'

    self._collections_ready = False
    self._collections_lock = asyncio.Lock()
    self._app_state_locks: dict[str, asyncio.Lock] = {}
    self._user_state_locks: dict[str, asyncio.Lock] = {}

  async def _ensure_collections(self) -> None:
    if self._collections_ready:
      return
    async with self._collections_lock:
      if self._collections_ready:
        return
      await asyncio.to_thread(self._ensure_collections_sync)
      self._collections_ready = True

  def _ensure_collections_sync(self) -> None:
    for schema_template, col_name in [
        (_SESSIONS_SCHEMA, self._sessions_col),
        (_EVENTS_SCHEMA, self._events_col),
        (_APP_STATES_SCHEMA, self._app_states_col),
        (_USER_STATES_SCHEMA, self._user_states_col),
    ]:
      try:
        self._client.collections[col_name].retrieve()
      except self._ObjectNotFound:
        schema = copy.deepcopy(schema_template)
        schema['name'] = col_name
        self._client.collections.create(schema)

  def _get_doc_sync(
      self, collection: str, doc_id: str
  ) -> Optional[dict[str, Any]]:
    try:
      return self._client.collections[collection].documents[doc_id].retrieve()
    except self._ObjectNotFound:
      return None

  async def _search_all(
      self, collection: str, filter_by: str, sort_by: Optional[str] = None
  ) -> list[dict[str, Any]]:
    """Return all documents matching filter_by, handling pagination."""
    params: dict[str, Any] = {
        'q': '*',
        'query_by': 'app_name',
        'filter_by': filter_by,
        'per_page': 250,
    }
    if sort_by:
      params['sort_by'] = sort_by

    docs: list[dict[str, Any]] = []
    page = 1
    while True:
      params['page'] = page
      result = await asyncio.to_thread(
          self._client.collections[collection].documents.search, params
      )
      hits = result.get('hits', [])
      docs.extend(hit['document'] for hit in hits)
      if len(hits) < 250:
        break
      page += 1
    return docs

  async def _get_app_state(self, app_name: str) -> dict[str, Any]:
    doc = await asyncio.to_thread(
        self._get_doc_sync, self._app_states_col, app_name
    )
    return json.loads(doc['state']) if doc else {}

  async def _get_user_state(
      self, app_name: str, user_id: str
  ) -> dict[str, Any]:
    doc = await asyncio.to_thread(
        self._get_doc_sync,
        self._user_states_col,
        _user_state_doc_id(app_name, user_id),
    )
    return json.loads(doc['state']) if doc else {}

  async def _upsert_app_state(
      self, app_name: str, delta: dict[str, Any], now: float
  ) -> None:
    lock = self._app_state_locks.setdefault(app_name, asyncio.Lock())
    async with lock:
      existing = await self._get_app_state(app_name)
      merged = existing | delta
      await asyncio.to_thread(
          self._client.collections[self._app_states_col].documents.upsert,
          {
              'id': app_name,
              'app_name': app_name,
              'state': json.dumps(merged),
              'update_time': now,
          },
      )

  async def _upsert_user_state(
      self, app_name: str, user_id: str, delta: dict[str, Any], now: float
  ) -> None:
    key = _user_state_doc_id(app_name, user_id)
    lock = self._user_state_locks.setdefault(key, asyncio.Lock())
    async with lock:
      existing = await self._get_user_state(app_name, user_id)
      merged = existing | delta
      await asyncio.to_thread(
          self._client.collections[self._user_states_col].documents.upsert,
          {
              'id': key,
              'app_name': app_name,
              'user_id': user_id,
              'state': json.dumps(merged),
              'update_time': now,
          },
      )

  @override
  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    await self._ensure_collections()

    session_id = (session_id or '').strip() or str(uuid.uuid4())
    _validate_no_sep(app_name, user_id, session_id)
    now = time.time()
    doc_id = _session_doc_id(app_name, user_id, session_id)

    existing = await asyncio.to_thread(
        self._get_doc_sync, self._sessions_col, doc_id
    )
    if existing:
      raise AlreadyExistsError(f'Session with id {session_id} already exists.')

    state_deltas = _extract_state_delta(state or {})
    app_state_delta = state_deltas['app']
    user_state_delta = state_deltas['user']
    session_state = state_deltas['session']

    if app_state_delta:
      await self._upsert_app_state(app_name, app_state_delta, now)
    if user_state_delta:
      await self._upsert_user_state(app_name, user_id, user_state_delta, now)

    storage_app_state = await self._get_app_state(app_name)
    storage_user_state = await self._get_user_state(app_name, user_id)

    try:
      await asyncio.to_thread(
          self._client.collections[self._sessions_col].documents.create,
          {
              'id': doc_id,
              'app_name': app_name,
              'user_id': user_id,
              'session_id': session_id,
              'state': json.dumps(session_state),
              'update_time': now,
          },
      )
    except self._ObjectAlreadyExists:
      raise AlreadyExistsError(f'Session with id {session_id} already exists.')

    merged_state = _merge_state(
        storage_app_state, storage_user_state, session_state
    )
    return Session(
        app_name=app_name,
        user_id=user_id,
        id=session_id,
        state=merged_state,
        events=[],
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
    await self._ensure_collections()
    _validate_no_sep(app_name, user_id, session_id)

    doc_id = _session_doc_id(app_name, user_id, session_id)
    session_doc = await asyncio.to_thread(
        self._get_doc_sync, self._sessions_col, doc_id
    )
    if session_doc is None:
      return None

    session_state = json.loads(session_doc['state'])
    last_update_time = session_doc['update_time']

    events: list[Event] = []
    if not (config and config.num_recent_events == 0):
      filter_by = (
          f'app_name:={app_name}'
          f' && user_id:={user_id}'
          f' && session_id:={session_id}'
      )
      if config and config.after_timestamp:
        filter_by += f' && timestamp:>={config.after_timestamp}'

      if config and config.num_recent_events is not None:
        event_docs = []
        page = 1
        remaining = config.num_recent_events
        while remaining > 0:
          params: dict[str, Any] = {
              'q': '*',
              'query_by': 'app_name',
              'filter_by': filter_by,
              'sort_by': 'timestamp:desc',
              'per_page': min(remaining, 250),
              'page': page,
          }
          result = await asyncio.to_thread(
              self._client.collections[self._events_col].documents.search,
              params,
          )
          hits = result.get('hits', [])
          event_docs.extend(hit['document'] for hit in hits)
          if len(hits) < min(remaining, 250):
            break
          remaining -= len(hits)
          page += 1
      else:
        event_docs = await self._search_all(
            self._events_col,
            filter_by,
            sort_by='timestamp:desc',
        )

      events = [
          Event.model_validate_json(doc['event_data'])
          for doc in reversed(event_docs)
      ]

    app_state = await self._get_app_state(app_name)
    user_state = await self._get_user_state(app_name, user_id)
    merged_state = _merge_state(app_state, user_state, session_state)

    return Session(
        app_name=app_name,
        user_id=user_id,
        id=session_id,
        state=merged_state,
        events=events,
        last_update_time=last_update_time,
    )

  @override
  async def list_sessions(
      self, *, app_name: str, user_id: Optional[str] = None
  ) -> ListSessionsResponse:
    await self._ensure_collections()
    _validate_no_sep(app_name)
    if user_id:
      _validate_no_sep(user_id)

    filter_by = f'app_name:={app_name}'
    if user_id:
      filter_by += f' && user_id:={user_id}'

    session_docs = await self._search_all(self._sessions_col, filter_by)
    app_state = await self._get_app_state(app_name)

    user_states_map: dict[str, dict[str, Any]] = {}
    unique_user_ids = {doc['user_id'] for doc in session_docs}
    for uid in unique_user_ids:
      user_states_map[uid] = await self._get_user_state(app_name, uid)

    sessions = []
    for doc in session_docs:
      uid = doc['user_id']
      merged_state = _merge_state(
          app_state,
          user_states_map.get(uid, {}),
          json.loads(doc['state']),
      )
      sessions.append(
          Session(
              app_name=app_name,
              user_id=uid,
              id=doc['session_id'],
              state=merged_state,
              events=[],
              last_update_time=doc['update_time'],
          )
      )
    return ListSessionsResponse(sessions=sessions)

  @override
  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    await self._ensure_collections()
    _validate_no_sep(app_name, user_id, session_id)

    events_filter = (
        f'app_name:={app_name}'
        f' && user_id:={user_id}'
        f' && session_id:={session_id}'
    )
    await asyncio.to_thread(
        self._client.collections[self._events_col].documents.delete,
        {'filter_by': events_filter},
    )

    doc_id = _session_doc_id(app_name, user_id, session_id)
    try:
      await asyncio.to_thread(
          self._client.collections[self._sessions_col].documents[doc_id].delete
      )
    except self._ObjectNotFound:
      pass

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    if event.partial:
      return event

    await self._ensure_collections()

    self._apply_temp_state(session, event)
    event = self._trim_temp_delta_state(event)
    event_timestamp = event.timestamp

    doc_id = _session_doc_id(session.app_name, session.user_id, session.id)
    session_doc = await asyncio.to_thread(
        self._get_doc_sync, self._sessions_col, doc_id
    )
    if session_doc is None:
      raise ValueError(f'Session {session.id} not found.')

    if session_doc['update_time'] > session.last_update_time:
      raise ValueError(
          'The session has been modified in storage since it was loaded.'
          ' Please reload the session before appending more events.'
      )

    session_state_update: dict[str, Any] = {'update_time': event_timestamp}

    if event.actions and event.actions.state_delta:
      state_deltas = _extract_state_delta(event.actions.state_delta)
      if state_deltas['app']:
        await self._upsert_app_state(
            session.app_name, state_deltas['app'], event_timestamp
        )
      if state_deltas['user']:
        await self._upsert_user_state(
            session.app_name,
            session.user_id,
            state_deltas['user'],
            event_timestamp,
        )
      if state_deltas['session']:
        current_state = json.loads(session_doc['state'])
        session_state_update['state'] = json.dumps(
            current_state | state_deltas['session']
        )

    await asyncio.to_thread(
        self._client.collections[self._sessions_col].documents[doc_id].update,
        session_state_update,
    )

    await asyncio.to_thread(
        self._client.collections[self._events_col].documents.create,
        {
            'id': _event_doc_id(
                session.app_name, session.user_id, session.id, event.id
            ),
            'app_name': session.app_name,
            'user_id': session.user_id,
            'session_id': session.id,
            'timestamp': event.timestamp,
            'event_data': event.model_dump_json(exclude_none=True),
        },
    )

    session.last_update_time = event_timestamp
    await super().append_event(session=session, event=event)
    return event
