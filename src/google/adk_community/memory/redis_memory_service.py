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

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import hashlib
import json
import re
from typing import Any
from typing import TYPE_CHECKING
from urllib.parse import quote

from google.adk.memory import _utils
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types
import redis.asyncio as redis
from typing_extensions import override

from .utils import extract_text_from_event

if TYPE_CHECKING:
  from google.adk.events.event import Event
  from google.adk.sessions.session import Session

_UNKNOWN_SESSION_ID = '__unknown_session_id__'


def _key_part(value: str) -> str:
  return quote(value, safe='')


def _decode(value: Any) -> str:
  if isinstance(value, bytes):
    return value.decode('utf-8')
  return str(value)


def _extract_words_lower(text: str) -> set[str]:
  return set(word.lower() for word in re.findall(r'[A-Za-z]+', text))


def _content_from_text(text: str) -> types.Content:
  return types.Content(parts=[types.Part(text=text)])


def _event_id(event: Event, content_text: str) -> str:
  if event.id:
    return event.id
  digest = hashlib.sha256(
      f'{event.author}:{event.timestamp}:{content_text}'.encode('utf-8')
  ).hexdigest()
  return f'generated-{digest}'


def _event_to_payload(
    event: Event,
    *,
    session_id: str,
    content_text: str,
    custom_metadata: Mapping[str, object] | None = None,
) -> dict[str, Any]:
  metadata = dict(custom_metadata or {})
  metadata.setdefault('session_id', session_id)
  return {
      'id': _event_id(event, content_text),
      'author': event.author,
      'timestamp': (
          _utils.format_timestamp(event.timestamp)
          if event.timestamp is not None
          else None
      ),
      'content': (
          _content_from_text(content_text).model_dump(
              mode='json', by_alias=True, exclude_none=True
          )
      ),
      'text': content_text,
      'custom_metadata': metadata,
  }


class RedisMemoryService(BaseMemoryService):
  """Redis-backed memory service for ADK community integrations.

  This service mirrors InMemoryMemoryService's keyword search behavior while
  keeping memory entries in Redis so they survive process restarts.
  """

  def __init__(
      self,
      host: str = 'localhost',
      port: int = 6379,
      db: int = 0,
      uri: str | None = None,
      cluster_uri: str | None = None,
      *,
      key_prefix: str = 'adk:memory:',
      client: Any | None = None,
      **kwargs: Any,
  ):
    """Initializes the Redis memory service.

    Args:
      host: Redis host used when uri, cluster_uri, and client are not supplied.
      port: Redis port used when uri, cluster_uri, and client are not supplied.
      db: Redis database used when uri, cluster_uri, and client are not supplied.
      uri: Redis URL used to create a standalone Redis client.
      cluster_uri: Redis Cluster URL used to create a Redis Cluster client.
      key_prefix: Prefix for all Redis keys written by this service.
      client: Optional async Redis-compatible client, mainly for tests.
      **kwargs: Extra keyword arguments forwarded to the Redis client factory.
    """
    if client is not None:
      self.cache = client
    elif cluster_uri:
      self.cache = redis.RedisCluster.from_url(cluster_uri, **kwargs)
    elif uri:
      self.cache = redis.Redis.from_url(uri, **kwargs)
    else:
      self.cache = redis.Redis(host=host, port=port, db=db, **kwargs)

    self._key_prefix = key_prefix

  def _scope_prefix(self, app_name: str, user_id: str) -> str:
    return f'{self._key_prefix}{_key_part(app_name)}:{_key_part(user_id)}'

  def _sessions_key(self, app_name: str, user_id: str) -> str:
    return f'{self._scope_prefix(app_name, user_id)}:sessions'

  def _session_keys(
      self, app_name: str, user_id: str, session_id: str
  ) -> tuple[str, str]:
    session_prefix = (
        f'{self._scope_prefix(app_name, user_id)}:{_key_part(session_id)}'
    )
    return f'{session_prefix}:order', f'{session_prefix}:entries'

  async def _append_events(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      events: Sequence[Event],
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    await self.cache.sadd(self._sessions_key(app_name, user_id), session_id)
    order_key, entries_key = self._session_keys(app_name, user_id, session_id)

    for event in events:
      content_text = extract_text_from_event(event)
      if not content_text:
        continue

      event_id = _event_id(event, content_text)
      payload = _event_to_payload(
          event,
          session_id=session_id,
          content_text=content_text,
          custom_metadata=custom_metadata,
      )
      was_added = await self.cache.hsetnx(
          entries_key, event_id, json.dumps(payload)
      )
      if was_added:
        await self.cache.rpush(order_key, event_id)

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    session_id = session.id or _UNKNOWN_SESSION_ID
    order_key, entries_key = self._session_keys(
        session.app_name, session.user_id, session_id
    )
    await self.cache.delete(order_key)
    await self.cache.delete(entries_key)
    await self._append_events(
        app_name=session.app_name,
        user_id=session.user_id,
        session_id=session_id,
        events=session.events,
    )

  @override
  async def add_events_to_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      events: Sequence[Event],
      session_id: str | None = None,
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    await self._append_events(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id or _UNKNOWN_SESSION_ID,
        events=events,
        custom_metadata=custom_metadata,
    )

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    sessions_key = self._sessions_key(app_name, user_id)
    session_ids = sorted(
        [_decode(value) for value in await self.cache.smembers(sessions_key)]
    )
    words_in_query = _extract_words_lower(query)
    response = SearchMemoryResponse()

    for session_id in session_ids:
      order_key, entries_key = self._session_keys(app_name, user_id, session_id)
      event_ids = [
          _decode(value) for value in await self.cache.lrange(order_key, 0, -1)
      ]
      for event_id in event_ids:
        raw_payload = await self.cache.hget(entries_key, event_id)
        if raw_payload is None:
          continue
        payload = json.loads(_decode(raw_payload))
        words_in_memory = _extract_words_lower(payload.get('text', ''))
        if not words_in_memory:
          continue
        if any(query_word in words_in_memory for query_word in words_in_query):
          response.memories.append(
              MemoryEntry(
                  id=payload['id'],
                  content=types.Content.model_validate(payload['content']),
                  author=payload.get('author'),
                  timestamp=payload.get('timestamp'),
                  custom_metadata=payload.get('custom_metadata') or {},
              )
          )

    return response

  async def close(self) -> None:
    """Closes the Redis client if it exposes a close method."""
    close = getattr(self.cache, 'aclose', None)
    if close is None:
      close = getattr(self.cache, 'close', None)
    if close is not None:
      result = close()
      if hasattr(result, '__await__'):
        await result

  async def __aenter__(self) -> RedisMemoryService:
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    await self.close()
