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

"""Milvus-backed memory service for ADK."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import hashlib
import inspect
import json
import logging
import os
import re
from typing import Optional
from typing import TYPE_CHECKING

from google.adk.memory import _utils
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from .utils import extract_text_from_event

if TYPE_CHECKING:
  from google.adk.events.event import Event
  from google.adk.sessions.session import Session

logger = logging.getLogger("google_adk_community." + __name__)

EmbeddingFunction = Callable[
    [Sequence[str]],
    Sequence[Sequence[float]] | Awaitable[Sequence[Sequence[float]]],
]

_DEFAULT_COLLECTION_NAME = "adk_memory"
_DEFAULT_LITE_URI = "./adk_milvus_memory.db"
_UNKNOWN_SESSION_ID = "__unknown_session_id__"
_EVENT_SOURCE = "adk_event"
_DIRECT_MEMORY_SOURCE = "adk_memory"
_FIELD_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _load_pymilvus():
  try:
    from pymilvus import DataType
    from pymilvus import MilvusClient
  except ImportError as exc:
    raise ImportError(
        "pymilvus is required to use MilvusMemoryService. "
        "Install it with: pip install google-adk-community[milvus]"
    ) from exc
  return MilvusClient, DataType


def _env_value(name: str) -> Optional[str]:
  value = os.getenv(name)
  return value if value else None


def _json_safe(value: Mapping[str, object] | None) -> dict[str, object]:
  if not value:
    return {}
  return json.loads(json.dumps(dict(value), default=str))


def _quote_filter_value(value: str) -> str:
  return json.dumps(value)


def _hash_id(*parts: object) -> str:
  digest = hashlib.sha256()
  for part in parts:
    digest.update(str(part).encode("utf-8"))
    digest.update(b"\0")
  return "adk-" + digest.hexdigest()


def _content_to_text(content: types.Content | None) -> str:
  if not content or not content.parts:
    return ""
  text_parts = [
      part.text for part in content.parts if part.text and not part.thought
  ]
  return " ".join(text_parts)


def _timestamp_from_event(event: Event) -> str:
  if event.timestamp is None:
    return ""
  return _utils.format_timestamp(event.timestamp)


def _field_name(field: Mapping[str, object]) -> str | None:
  raw_name = field.get("name", field.get("field_name"))
  return str(raw_name) if raw_name is not None else None


def _field_dim(field: Mapping[str, object]) -> int | None:
  params = field.get("params", {})
  candidates = []
  if isinstance(params, Mapping):
    candidates.append(params.get("dim"))
    nested_params = params.get("params")
    if isinstance(nested_params, Mapping):
      candidates.append(nested_params.get("dim"))
  candidates.append(field.get("dim"))
  for candidate in candidates:
    if candidate is not None:
      return int(candidate)
  return None


def _field_type(field: Mapping[str, object]) -> object | None:
  return field.get("type", field.get("data_type", field.get("datatype")))


def _type_label(data_type: object) -> str:
  return str(getattr(data_type, "name", data_type))


def _field_type_matches(actual: object, expected: object) -> bool:
  if actual == expected:
    return True
  actual_value = getattr(actual, "value", actual)
  expected_value = getattr(expected, "value", expected)
  if actual_value == expected_value:
    return True
  return _type_label(actual) == _type_label(expected)


def _validate_field_names(config: "MilvusMemoryServiceConfig") -> None:
  field_names = [
      config.id_field,
      config.vector_field,
      config.text_field,
      config.app_name_field,
      config.user_id_field,
      config.session_id_field,
      config.event_id_field,
      config.author_field,
      config.timestamp_field,
      config.source_field,
      config.metadata_field,
  ]
  invalid_names = [
      name for name in field_names if not _FIELD_NAME_PATTERN.fullmatch(name)
  ]
  if invalid_names:
    raise ValueError(
        "Milvus field names must be valid identifiers: "
        + ", ".join(sorted(set(invalid_names)))
    )
  duplicate_names = sorted(
      {name for name in field_names if field_names.count(name) > 1}
  )
  if duplicate_names:
    raise ValueError(
        "Milvus field names must be unique: " + ", ".join(duplicate_names)
    )


class MilvusMemoryServiceConfig(BaseModel):
  """Configuration for Milvus memory storage."""

  uri: str = Field(
      default_factory=lambda: _env_value("MILVUS_URI") or _DEFAULT_LITE_URI
  )
  token: Optional[str] = Field(
      default_factory=lambda: _env_value("MILVUS_TOKEN")
  )
  db_name: Optional[str] = Field(
      default_factory=lambda: _env_value("MILVUS_DB_NAME")
  )
  collection_name: str = Field(default=_DEFAULT_COLLECTION_NAME, min_length=1)
  dimension: int = Field(gt=0)
  search_top_k: int = Field(default=10, ge=1, le=100)
  metric_type: str = Field(default="COSINE")
  index_type: str = Field(default="AUTOINDEX")
  consistency_level: Optional[str] = Field(default="Session")
  timeout: Optional[float] = Field(default=None, gt=0.0)
  id_field: str = Field(default="id")
  vector_field: str = Field(default="embedding")
  text_field: str = Field(default="text")
  app_name_field: str = Field(default="app_name")
  user_id_field: str = Field(default="user_id")
  session_id_field: str = Field(default="session_id")
  event_id_field: str = Field(default="event_id")
  author_field: str = Field(default="author")
  timestamp_field: str = Field(default="timestamp")
  source_field: str = Field(default="source")
  metadata_field: str = Field(default="metadata")
  id_max_length: int = Field(default=512, gt=0)
  text_max_length: int = Field(default=65535, gt=0)
  scalar_max_length: int = Field(default=1024, gt=0)


class MilvusMemoryService(BaseMemoryService):
  """A Milvus-backed implementation of ADK's BaseMemoryService."""

  def __init__(
      self,
      *,
      embedding_function: EmbeddingFunction,
      dimension: int | None = None,
      config: MilvusMemoryServiceConfig | None = None,
      uri: str | None = None,
      token: str | None = None,
      db_name: str | None = None,
      collection_name: str | None = None,
      search_top_k: int | None = None,
      consistency_level: str | None = None,
  ):
    """Initializes the Milvus memory service.

    Args:
      embedding_function: Function that embeds a batch of texts.
      dimension: Embedding vector dimension. Required unless provided by config.
      config: Optional MilvusMemoryServiceConfig.
      uri: Optional Milvus URI override. Defaults to MILVUS_URI or local Lite.
      token: Optional token override. Defaults to MILVUS_TOKEN.
      db_name: Optional Milvus database name override. Defaults to
        MILVUS_DB_NAME.
      collection_name: Optional collection name override.
      search_top_k: Optional search result limit override.
      consistency_level: Optional collection consistency level override.
    """
    if config is None:
      if dimension is None:
        raise ValueError("dimension is required when config is not provided.")
      config = MilvusMemoryServiceConfig(dimension=dimension)
    elif dimension is not None:
      config = config.model_copy(update={"dimension": dimension})

    updates: dict[str, object] = {}
    if uri is not None:
      updates["uri"] = uri
    if token is not None:
      updates["token"] = token
    if db_name is not None:
      updates["db_name"] = db_name
    if collection_name is not None:
      updates["collection_name"] = collection_name
    if search_top_k is not None:
      updates["search_top_k"] = search_top_k
    if consistency_level is not None:
      updates["consistency_level"] = consistency_level
    if updates:
      config = config.model_copy(update=updates)

    _validate_field_names(config)

    self._embedding_function = embedding_function
    self._config = config
    milvus_client, data_type = _load_pymilvus()
    self._data_type = data_type
    client_kwargs: dict[str, object] = {
        "uri": self._config.uri,
    }
    if self._config.token:
      client_kwargs["token"] = self._config.token
    if self._config.db_name:
      client_kwargs["db_name"] = self._config.db_name
    if self._config.timeout is not None:
      client_kwargs["timeout"] = self._config.timeout
    self._client = milvus_client(**client_kwargs)
    self._ensure_collection()

  def _ensure_collection(self) -> None:
    if self._client.has_collection(
        collection_name=self._config.collection_name,
        timeout=self._config.timeout,
    ):
      self._validate_existing_collection()
      return

    schema = self._client.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(
        field_name=self._config.id_field,
        datatype=self._data_type.VARCHAR,
        is_primary=True,
        max_length=self._config.id_max_length,
    )
    schema.add_field(
        field_name=self._config.vector_field,
        datatype=self._data_type.FLOAT_VECTOR,
        dim=self._config.dimension,
    )
    for field_name, max_length in [
        (self._config.text_field, self._config.text_max_length),
        (self._config.app_name_field, self._config.scalar_max_length),
        (self._config.user_id_field, self._config.scalar_max_length),
        (self._config.session_id_field, self._config.scalar_max_length),
        (self._config.event_id_field, self._config.scalar_max_length),
        (self._config.author_field, self._config.scalar_max_length),
        (self._config.timestamp_field, self._config.scalar_max_length),
        (self._config.source_field, self._config.scalar_max_length),
    ]:
      schema.add_field(
          field_name=field_name,
          datatype=self._data_type.VARCHAR,
          max_length=max_length,
      )
    schema.add_field(
        field_name=self._config.metadata_field,
        datatype=self._data_type.JSON,
    )

    index_params = self._client.prepare_index_params()
    index_params.add_index(
        field_name=self._config.vector_field,
        index_type=self._config.index_type,
        metric_type=self._config.metric_type,
    )
    create_kwargs: dict[str, object] = {
        "collection_name": self._config.collection_name,
        "schema": schema,
        "index_params": index_params,
        "timeout": self._config.timeout,
    }
    if self._config.consistency_level:
      create_kwargs["consistency_level"] = self._config.consistency_level
    self._client.create_collection(**create_kwargs)

  def _validate_existing_collection(self) -> None:
    description = self._client.describe_collection(
        collection_name=self._config.collection_name,
        timeout=self._config.timeout,
    )
    fields = {
        name: field
        for field in description.get("fields", [])
        if isinstance(field, Mapping) and (name := _field_name(field))
    }
    required_fields = [
        self._config.id_field,
        self._config.vector_field,
        self._config.text_field,
        self._config.app_name_field,
        self._config.user_id_field,
    ]
    missing_fields = [field for field in required_fields if field not in fields]
    if missing_fields:
      raise ValueError(
          "Milvus collection "
          f"{self._config.collection_name!r} is missing required fields: "
          + ", ".join(missing_fields)
      )
    if description.get("auto_id") is True:
      raise ValueError(
          "Milvus collection "
          f"{self._config.collection_name!r} must use auto_id=False."
      )
    id_field = fields[self._config.id_field]
    if id_field.get("is_primary") is not None and not id_field.get(
        "is_primary"
    ):
      raise ValueError(
          "Milvus collection "
          f"{self._config.collection_name!r} field "
          f"{self._config.id_field!r} must be the primary key."
      )
    self._validate_field_type(
        id_field, self._data_type.VARCHAR, self._config.id_field
    )
    self._validate_field_type(
        fields[self._config.vector_field],
        self._data_type.FLOAT_VECTOR,
        self._config.vector_field,
    )
    for field_name in [
        self._config.text_field,
        self._config.app_name_field,
        self._config.user_id_field,
    ]:
      self._validate_field_type(
          fields[field_name], self._data_type.VARCHAR, field_name
      )
    vector_dim = _field_dim(fields[self._config.vector_field])
    if vector_dim is not None and vector_dim != self._config.dimension:
      raise ValueError(
          "Milvus collection "
          f"{self._config.collection_name!r} has vector dimension "
          f"{vector_dim}, expected {self._config.dimension}."
      )

  def _validate_field_type(
      self, field: Mapping[str, object], expected_type: object, field_name: str
  ) -> None:
    actual_type = _field_type(field)
    if actual_type is not None and not _field_type_matches(
        actual_type, expected_type
    ):
      raise ValueError(
          "Milvus collection "
          f"{self._config.collection_name!r} field {field_name!r} has type "
          f"{_type_label(actual_type)}, expected {_type_label(expected_type)}."
      )

  async def _embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
    if not texts:
      return []
    embeddings = self._embedding_function(texts)
    if inspect.isawaitable(embeddings):
      embeddings = await embeddings

    vectors = [list(vector) for vector in embeddings]
    if len(vectors) != len(texts):
      raise ValueError(
          "embedding_function returned "
          f"{len(vectors)} vectors for {len(texts)} texts."
      )
    for vector in vectors:
      if len(vector) != self._config.dimension:
        raise ValueError(
            "embedding_function returned vector dimension "
            f"{len(vector)}, expected {self._config.dimension}."
        )
    return vectors

  def _scope_filter(self, *, app_name: str, user_id: str) -> str:
    return (
        f"{self._config.app_name_field} == {_quote_filter_value(app_name)} "
        f"and {self._config.user_id_field} == {_quote_filter_value(user_id)}"
    )

  def _event_to_record(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str | None,
      event: Event,
      text: str,
      embedding: Sequence[float],
      custom_metadata: Mapping[str, object] | None = None,
  ) -> dict[str, object]:
    scoped_session_id = session_id or _UNKNOWN_SESSION_ID
    event_id = event.id or _hash_id(
        app_name,
        user_id,
        scoped_session_id,
        event.author,
        event.timestamp,
        text,
    )
    record_id = _hash_id(app_name, user_id, scoped_session_id, event_id)
    metadata = _json_safe(custom_metadata)
    metadata.update({
        "invocation_id": event.invocation_id,
        "source": _EVENT_SOURCE,
    })
    return self._record(
        record_id=record_id,
        app_name=app_name,
        user_id=user_id,
        session_id=scoped_session_id,
        event_id=event_id,
        author=event.author or "",
        timestamp=_timestamp_from_event(event),
        text=text,
        embedding=embedding,
        source=_EVENT_SOURCE,
        metadata=metadata,
    )

  def _memory_to_record(
      self,
      *,
      app_name: str,
      user_id: str,
      memory: MemoryEntry,
      text: str,
      embedding: Sequence[float],
      index: int,
      custom_metadata: Mapping[str, object] | None = None,
  ) -> dict[str, object]:
    record_id = memory.id or _hash_id(app_name, user_id, index, text)
    metadata = _json_safe(custom_metadata)
    metadata.update(_json_safe(memory.custom_metadata))
    metadata["source"] = _DIRECT_MEMORY_SOURCE
    return self._record(
        record_id=record_id,
        app_name=app_name,
        user_id=user_id,
        session_id="",
        event_id="",
        author=memory.author or "",
        timestamp=memory.timestamp or "",
        text=text,
        embedding=embedding,
        source=_DIRECT_MEMORY_SOURCE,
        metadata=metadata,
    )

  def _record(
      self,
      *,
      record_id: str,
      app_name: str,
      user_id: str,
      session_id: str,
      event_id: str,
      author: str,
      timestamp: str,
      text: str,
      embedding: Sequence[float],
      source: str,
      metadata: Mapping[str, object],
  ) -> dict[str, object]:
    return {
        self._config.id_field: record_id[: self._config.id_max_length],
        self._config.vector_field: list(embedding),
        self._config.text_field: text[: self._config.text_max_length],
        self._config.app_name_field: app_name[: self._config.scalar_max_length],
        self._config.user_id_field: user_id[: self._config.scalar_max_length],
        self._config.session_id_field: session_id[
            : self._config.scalar_max_length
        ],
        self._config.event_id_field: event_id[: self._config.scalar_max_length],
        self._config.author_field: author[: self._config.scalar_max_length],
        self._config.timestamp_field: timestamp[
            : self._config.scalar_max_length
        ],
        self._config.source_field: source[: self._config.scalar_max_length],
        self._config.metadata_field: dict(metadata),
    }

  async def _upsert_records(self, records: Sequence[dict[str, object]]) -> None:
    if not records:
      return
    await asyncio.to_thread(
        self._client.upsert,
        collection_name=self._config.collection_name,
        data=list(records),
        timeout=self._config.timeout,
    )

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    await self.add_events_to_memory(
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
    items = [
        (event, text)
        for event in events
        if (text := extract_text_from_event(event))
    ]
    embeddings = await self._embed_texts([text for _, text in items])
    records = [
        self._event_to_record(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            event=event,
            text=text,
            embedding=embedding,
            custom_metadata=custom_metadata,
        )
        for (event, text), embedding in zip(items, embeddings)
    ]
    await self._upsert_records(records)
    logger.info("Added %d memories to Milvus.", len(records))

  @override
  async def add_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      memories: Sequence[MemoryEntry],
      custom_metadata: Mapping[str, object] | None = None,
  ) -> None:
    items = [
        (index, memory, text)
        for index, memory in enumerate(memories)
        if (text := _content_to_text(memory.content))
    ]
    embeddings = await self._embed_texts([text for _, _, text in items])
    records = [
        self._memory_to_record(
            app_name=app_name,
            user_id=user_id,
            memory=memory,
            text=text,
            embedding=embedding,
            index=index,
            custom_metadata=custom_metadata,
        )
        for (index, memory, text), embedding in zip(items, embeddings)
    ]
    await self._upsert_records(records)
    logger.info("Added %d direct memories to Milvus.", len(records))

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    query_embedding = (await self._embed_texts([query]))[0]
    results = await asyncio.to_thread(
        self._client.search,
        collection_name=self._config.collection_name,
        data=[query_embedding],
        filter=self._scope_filter(app_name=app_name, user_id=user_id),
        limit=self._config.search_top_k,
        output_fields=[
            self._config.text_field,
            self._config.author_field,
            self._config.timestamp_field,
            self._config.metadata_field,
        ],
        anns_field=self._config.vector_field,
        search_params={"metric_type": self._config.metric_type},
        timeout=self._config.timeout,
    )
    memories: list[MemoryEntry] = []
    for hit in results[0] if results else []:
      entity = hit.get("entity", {})
      text = entity.get(self._config.text_field, "")
      if not text:
        continue
      memories.append(
          MemoryEntry(
              content=types.Content(parts=[types.Part(text=text)]),
              author=entity.get(self._config.author_field) or None,
              timestamp=entity.get(self._config.timestamp_field) or None,
              custom_metadata=entity.get(self._config.metadata_field) or {},
          )
      )
    return SearchMemoryResponse(memories=memories)

  async def close(self) -> None:
    await asyncio.to_thread(self._client.close)
