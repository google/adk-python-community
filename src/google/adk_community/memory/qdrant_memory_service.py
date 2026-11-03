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

"""Memory service implementation backed by Qdrant."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import logging
from typing import Any
from typing import Literal
from typing import Optional
from typing import TYPE_CHECKING
import uuid

from google.adk.memory import _utils
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

if TYPE_CHECKING:
  from google.adk.events.event import Event
  from google.adk.sessions.session import Session
  from qdrant_client import AsyncQdrantClient

logger = logging.getLogger('google_adk_community.' + __name__)

EmbeddingFunction = Callable[[list[str]], Awaitable[list[list[float]]]]
"""Async callable mapping a batch of texts to one embedding vector each."""

_ID_NAMESPACE = uuid.UUID('6f1d5c2e-8f4b-4a57-9d3e-2b7c1a0e9f41')


class QdrantMemoryServiceConfig(BaseModel):
  """Configuration for QdrantMemoryService.

  Attributes:
      collection_name: Name of the Qdrant collection that stores memories.
      vector_size: Embedding dimension. If None, it is inferred from the first
        batch of embeddings when the collection is auto-created.
      distance: Distance metric used when the collection is auto-created.
      create_collection: Create the collection and payload indexes on first
        write if they do not exist. Set to False to manage them yourself.
      search_top_k: Maximum number of memories returned per search.
      score_threshold: Optional minimum similarity score for search results.
  """

  collection_name: str = 'adk_memory'
  vector_size: Optional[int] = Field(default=None, ge=1)
  distance: Literal['Cosine', 'Dot', 'Euclid', 'Manhattan'] = 'Cosine'
  create_collection: bool = True
  search_top_k: int = Field(default=10, ge=1)
  score_threshold: Optional[float] = None


class QdrantMemoryService(BaseMemoryService):
  """A memory service that stores and searches memories in Qdrant.

  Each text-bearing event becomes one Qdrant point. Points carry `app_name`
  and `user_id` in their payload, and every read, write and delete is scoped
  by a payload filter on both fields.

  Embeddings are computed by a caller-provided async function, so the service
  does not depend on any embedding provider.
  """

  def __init__(
      self,
      client: AsyncQdrantClient,
      embedding_function: EmbeddingFunction,
      config: Optional[QdrantMemoryServiceConfig] = None,
  ):
    """Initializes the Qdrant memory service.

    Args:
        client: An `AsyncQdrantClient`. The caller owns its lifecycle.
        embedding_function: Async callable that takes a list of texts and
          returns a list of embedding vectors, one per text.
        config: Optional QdrantMemoryServiceConfig. Defaults are used if None.
    """
    try:
      from qdrant_client import models
    except ImportError as exc:
      raise ImportError(
          'qdrant-client is required to use QdrantMemoryService. '
          'Install it with: pip install google-adk-community[qdrant]'
      ) from exc
    if client is None:
      raise ValueError('client is required.')
    if embedding_function is None:
      raise ValueError('embedding_function is required.')
    self._models = models
    self._client = client
    self._embedding_function = embedding_function
    self._config = config or QdrantMemoryServiceConfig()
    self._collection_ready = False
    self._collection_lock = asyncio.Lock()

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    await self._upsert_events(
        app_name=session.app_name,
        user_id=session.user_id,
        events=session.events,
        session_id=session.id,
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
    await self._upsert_events(
        app_name=app_name,
        user_id=user_id,
        events=events,
        session_id=session_id,
        custom_metadata=custom_metadata,
    )

  @override
  async def add_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      memories: Sequence[MemoryEntry],
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    records = []
    for memory in memories:
      text = _extract_text(memory.content)
      if not text:
        continue
      key = memory.id or str(uuid.uuid4())
      metadata = {**memory.custom_metadata, **(custom_metadata or {})}
      records.append((
          _point_id(app_name, user_id, key),
          text,
          {
              'app_name': app_name,
              'user_id': user_id,
              'text': text,
              'content': (
                  _visible_content(memory.content).model_dump(
                      mode='json', exclude_none=True
                  )
              ),
              'author': memory.author,
              'timestamp': memory.timestamp,
              'custom_metadata': metadata,
          },
      ))
    await self._upsert(records)

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    if not await self._collection_exists():
      return SearchMemoryResponse()
    [query_vector] = await self._embedding_function([query])
    response = await self._client.query_points(
        collection_name=self._config.collection_name,
        query=query_vector,
        query_filter=self._scope_filter(app_name, user_id),
        limit=self._config.search_top_k,
        score_threshold=self._config.score_threshold,
        with_payload=True,
    )
    memories = []
    for point in response.points:
      payload = point.payload or {}
      memories.append(
          MemoryEntry(
              id=str(point.id),
              content=types.Content.model_validate(payload['content']),
              author=payload.get('author'),
              timestamp=payload.get('timestamp'),
              custom_metadata=payload.get('custom_metadata') or {},
          )
      )
    return SearchMemoryResponse(memories=memories)

  @override
  async def delete_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      memory_id: str | None = None,
  ) -> None:
    if not memory_id or not await self._collection_exists():
      return
    try:
      uuid.UUID(memory_id)
    except ValueError:
      # Ids returned by search_memory are always UUIDs; anything else cannot
      # match a point in this collection.
      return
    models = self._models
    scoped = self._scope_filter(app_name, user_id)
    await self._client.delete(
        collection_name=self._config.collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.HasIdCondition(has_id=[memory_id]), *scoped.must]
            )
        ),
    )

  async def _upsert_events(
      self,
      *,
      app_name: str,
      user_id: str,
      events: Sequence[Event],
      session_id: str | None,
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    records = []
    for event in events:
      text = _extract_text(event.content)
      if not text:
        continue
      records.append((
          _point_id(app_name, user_id, event.id),
          text,
          {
              'app_name': app_name,
              'user_id': user_id,
              'session_id': session_id,
              'event_id': event.id,
              'text': text,
              'content': (
                  _visible_content(event.content).model_dump(
                      mode='json', exclude_none=True
                  )
              ),
              'author': event.author,
              'timestamp': _utils.format_timestamp(event.timestamp),
              'custom_metadata': dict(custom_metadata or {}),
          },
      ))
    await self._upsert(records)

  async def _upsert(self, records: list[tuple[str, str, dict[str, Any]]]):
    """Embeds and upserts (point_id, text, payload) records."""
    if not records:
      return
    vectors = await self._embedding_function([text for _, text, _ in records])
    if len(vectors) != len(records):
      raise ValueError(
          f'embedding_function returned {len(vectors)} vectors for'
          f' {len(records)} texts.'
      )
    await self._ensure_collection(len(vectors[0]))
    await self._client.upsert(
        collection_name=self._config.collection_name,
        points=[
            self._models.PointStruct(
                id=point_id, vector=vector, payload=payload
            )
            for (point_id, _, payload), vector in zip(records, vectors)
        ],
    )
    logger.debug(
        'Upserted %d memories into %s',
        len(records),
        self._config.collection_name,
    )

  async def _collection_exists(self) -> bool:
    if self._collection_ready:
      return True
    return await self._client.collection_exists(self._config.collection_name)

  async def _ensure_collection(self, vector_size: int) -> None:
    if self._collection_ready or not self._config.create_collection:
      return
    async with self._collection_lock:
      if self._collection_ready:
        return
      name = self._config.collection_name
      if not await self._client.collection_exists(name):
        models = self._models
        await self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=self._config.vector_size or vector_size,
                distance=models.Distance(self._config.distance),
            ),
        )
        for field in ('app_name', 'user_id'):
          await self._client.create_payload_index(
              collection_name=name,
              field_name=field,
              field_schema=models.PayloadSchemaType.KEYWORD,
          )
      self._collection_ready = True

  def _scope_filter(self, app_name: str, user_id: str):
    models = self._models
    return models.Filter(
        must=[
            models.FieldCondition(
                key='app_name', match=models.MatchValue(value=app_name)
            ),
            models.FieldCondition(
                key='user_id', match=models.MatchValue(value=user_id)
            ),
        ]
    )


def _point_id(app_name: str, user_id: str, key: str) -> str:
  """Deterministic point id so re-ingesting a session overwrites, not dupes."""
  return str(uuid.uuid5(_ID_NAMESPACE, f'{app_name}\x00{user_id}\x00{key}'))


def _visible_content(content: types.Content) -> types.Content:
  """Returns a copy of content with only non-thought text parts."""
  return types.Content(
      role=content.role,
      parts=[
          types.Part(text=part.text)
          for part in content.parts or []
          if part.text and not part.thought
      ],
  )


def _extract_text(content: Optional[types.Content]) -> str:
  if not content or not content.parts:
    return ''
  return ' '.join(
      part.text for part in content.parts if part.text and not part.thought
  )
