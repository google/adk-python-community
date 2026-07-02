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

"""Milvus vector store and retrieval toolset for ADK."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import hashlib
import inspect
import json
import os
import re
from typing import Any
from typing import Optional

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.base_toolset import ToolPredicate
from google.adk.tools.retrieval import BaseRetrievalTool
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

EmbeddingFunction = Callable[
    [Sequence[str]],
    Sequence[Sequence[float]] | Awaitable[Sequence[Sequence[float]]],
]

DEFAULT_MILVUS_TOOL_NAME_PREFIX = "milvus"
_DEFAULT_COLLECTION_NAME = "adk_rag"
_DEFAULT_LITE_URI = "./adk_milvus_rag.db"
_FIELD_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _load_pymilvus():
  try:
    from pymilvus import DataType
    from pymilvus import MilvusClient
  except ImportError as exc:
    raise ImportError(
        "pymilvus is required to use MilvusToolset. "
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


def _hash_id(*parts: object) -> str:
  digest = hashlib.sha256()
  for part in parts:
    digest.update(str(part).encode("utf-8"))
    digest.update(b"\0")
  return "adk-rag-" + digest.hexdigest()


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


def _validate_field_names(settings: "MilvusVectorStoreSettings") -> None:
  field_names = [
      settings.id_field,
      settings.vector_field,
      settings.content_field,
      settings.source_field,
      settings.metadata_field,
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


class MilvusVectorStoreSettings(BaseModel):
  """Settings for the Milvus vector store used by retrieval tools."""

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
  search_top_k: int = Field(default=4, ge=1, le=100)
  metric_type: str = Field(default="COSINE")
  index_type: str = Field(default="AUTOINDEX")
  consistency_level: Optional[str] = Field(default="Session")
  timeout: Optional[float] = Field(default=None, gt=0.0)
  auto_create: bool = True
  id_field: str = Field(default="id")
  vector_field: str = Field(default="embedding")
  content_field: str = Field(default="content")
  source_field: str = Field(default="source")
  metadata_field: str = Field(default="metadata")
  id_max_length: int = Field(default=512, gt=0)
  content_max_length: int = Field(default=65535, gt=0)
  scalar_max_length: int = Field(default=1024, gt=0)


class MilvusToolSettings(BaseModel):
  """Settings for Milvus tools."""

  vector_store_settings: MilvusVectorStoreSettings | None = None
  similarity_search_description: str = (
      "Performs semantic similarity search over a Milvus vector store and "
      "returns relevant context for the user's query."
  )


class MilvusVectorStore:
  """Utility class for Milvus collection setup, ingestion, and search."""

  def __init__(
      self,
      *,
      embedding_function: EmbeddingFunction,
      settings: MilvusVectorStoreSettings,
  ):
    _validate_field_names(settings)
    self._embedding_function = embedding_function
    self._settings = settings
    milvus_client, data_type = _load_pymilvus()
    self._data_type = data_type
    client_kwargs: dict[str, object] = {"uri": self._settings.uri}
    if self._settings.token:
      client_kwargs["token"] = self._settings.token
    if self._settings.db_name:
      client_kwargs["db_name"] = self._settings.db_name
    if self._settings.timeout is not None:
      client_kwargs["timeout"] = self._settings.timeout
    self._client = milvus_client(**client_kwargs)
    if self._settings.auto_create:
      self.create_vector_store()

  def create_vector_store(self) -> None:
    """Create the Milvus collection if it does not already exist."""
    if self._client.has_collection(
        collection_name=self._settings.collection_name,
        timeout=self._settings.timeout,
    ):
      self._validate_existing_collection()
      return

    schema = self._client.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(
        field_name=self._settings.id_field,
        datatype=self._data_type.VARCHAR,
        is_primary=True,
        max_length=self._settings.id_max_length,
    )
    schema.add_field(
        field_name=self._settings.vector_field,
        datatype=self._data_type.FLOAT_VECTOR,
        dim=self._settings.dimension,
    )
    schema.add_field(
        field_name=self._settings.content_field,
        datatype=self._data_type.VARCHAR,
        max_length=self._settings.content_max_length,
    )
    schema.add_field(
        field_name=self._settings.source_field,
        datatype=self._data_type.VARCHAR,
        max_length=self._settings.scalar_max_length,
    )
    schema.add_field(
        field_name=self._settings.metadata_field,
        datatype=self._data_type.JSON,
    )

    index_params = self._client.prepare_index_params()
    index_params.add_index(
        field_name=self._settings.vector_field,
        index_type=self._settings.index_type,
        metric_type=self._settings.metric_type,
    )
    create_kwargs: dict[str, object] = {
        "collection_name": self._settings.collection_name,
        "schema": schema,
        "index_params": index_params,
        "timeout": self._settings.timeout,
    }
    if self._settings.consistency_level:
      create_kwargs["consistency_level"] = self._settings.consistency_level
    self._client.create_collection(**create_kwargs)

  async def create_vector_store_async(self) -> None:
    """Asynchronously create the Milvus collection if needed."""
    await asyncio.to_thread(self.create_vector_store)

  def _validate_existing_collection(self) -> None:
    description = self._client.describe_collection(
        collection_name=self._settings.collection_name,
        timeout=self._settings.timeout,
    )
    fields = {
        name: field
        for field in description.get("fields", [])
        if isinstance(field, Mapping) and (name := _field_name(field))
    }
    required_fields = [
        self._settings.id_field,
        self._settings.vector_field,
        self._settings.content_field,
        self._settings.source_field,
        self._settings.metadata_field,
    ]
    missing_fields = [field for field in required_fields if field not in fields]
    if missing_fields:
      raise ValueError(
          "Milvus collection "
          f"{self._settings.collection_name!r} is missing required fields: "
          + ", ".join(missing_fields)
      )
    if description.get("auto_id") is True:
      raise ValueError(
          "Milvus collection "
          f"{self._settings.collection_name!r} must use auto_id=False."
      )
    id_field = fields[self._settings.id_field]
    if id_field.get("is_primary") is not None and not id_field.get(
        "is_primary"
    ):
      raise ValueError(
          "Milvus collection "
          f"{self._settings.collection_name!r} field "
          f"{self._settings.id_field!r} must be the primary key."
      )
    self._validate_field_type(
        id_field, self._data_type.VARCHAR, self._settings.id_field
    )
    self._validate_field_type(
        fields[self._settings.vector_field],
        self._data_type.FLOAT_VECTOR,
        self._settings.vector_field,
    )
    self._validate_field_type(
        fields[self._settings.content_field],
        self._data_type.VARCHAR,
        self._settings.content_field,
    )
    self._validate_field_type(
        fields[self._settings.source_field],
        self._data_type.VARCHAR,
        self._settings.source_field,
    )
    self._validate_field_type(
        fields[self._settings.metadata_field],
        self._data_type.JSON,
        self._settings.metadata_field,
    )
    vector_dim = _field_dim(fields[self._settings.vector_field])
    if vector_dim is not None and vector_dim != self._settings.dimension:
      raise ValueError(
          "Milvus collection "
          f"{self._settings.collection_name!r} has vector dimension "
          f"{vector_dim}, expected {self._settings.dimension}."
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
          f"{self._settings.collection_name!r} field {field_name!r} has type "
          f"{_type_label(actual_type)}, expected {_type_label(expected_type)}."
      )

  def _embed_texts_sync(self, texts: Sequence[str]) -> list[list[float]]:
    embeddings = self._embedding_function(texts)
    if inspect.isawaitable(embeddings):
      raise ValueError(
          "embedding_function returned an awaitable. "
          "Use add_texts_async() or similarity_search_async() instead."
      )
    return self._validate_embeddings(texts, embeddings)

  async def _embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
    embeddings = self._embedding_function(texts)
    if inspect.isawaitable(embeddings):
      embeddings = await embeddings
    return self._validate_embeddings(texts, embeddings)

  def _validate_embeddings(
      self, texts: Sequence[str], embeddings: Sequence[Sequence[float]]
  ) -> list[list[float]]:
    vectors = [list(vector) for vector in embeddings]
    if len(vectors) != len(texts):
      raise ValueError(
          "embedding_function returned "
          f"{len(vectors)} vectors for {len(texts)} texts."
      )
    for vector in vectors:
      if len(vector) != self._settings.dimension:
        raise ValueError(
            "embedding_function returned vector dimension "
            f"{len(vector)}, expected {self._settings.dimension}."
        )
    return vectors

  def _records(
      self,
      *,
      contents: Sequence[str],
      embeddings: Sequence[Sequence[float]],
      metadatas: Sequence[Mapping[str, object]],
      ids: Sequence[str | None],
  ) -> list[dict[str, object]]:
    records = []
    for index, (content, embedding, metadata, record_id) in enumerate(
        zip(contents, embeddings, metadatas, ids)
    ):
      safe_metadata = _json_safe(metadata)
      source = safe_metadata.get("source", "")
      if not isinstance(source, str):
        source = str(source)
      records.append({
          self._settings.id_field: (
              record_id or _hash_id(index, content, safe_metadata)
          )[: self._settings.id_max_length],
          self._settings.vector_field: list(embedding),
          self._settings.content_field: content[
              : self._settings.content_max_length
          ],
          self._settings.source_field: source[
              : self._settings.scalar_max_length
          ],
          self._settings.metadata_field: safe_metadata,
      })
    return records

  def _prepare_inputs(
      self,
      contents: Iterable[str],
      metadatas: Iterable[Mapping[str, object]] | None,
      ids: Iterable[str] | None,
  ) -> tuple[list[str], list[Mapping[str, object]], list[str | None]]:
    content_list = list(contents)
    metadata_list = (
        list(metadatas) if metadatas is not None else [{} for _ in content_list]
    )
    id_list = list(ids) if ids is not None else [None for _ in content_list]
    if len(metadata_list) != len(content_list):
      raise ValueError(
          "metadatas must contain one item per content item. "
          f"Got {len(metadata_list)} metadata items for "
          f"{len(content_list)} contents."
      )
    if len(id_list) != len(content_list):
      raise ValueError(
          "ids must contain one item per content item. "
          f"Got {len(id_list)} ids for {len(content_list)} contents."
      )
    return content_list, metadata_list, id_list

  def add_texts(
      self,
      contents: Iterable[str],
      *,
      metadatas: Iterable[Mapping[str, object]] | None = None,
      ids: Iterable[str] | None = None,
  ) -> dict[str, object]:
    """Embed and upsert text content into the Milvus vector store."""
    content_list, metadata_list, id_list = self._prepare_inputs(
        contents, metadatas, ids
    )
    if not content_list:
      return {"status": "SUCCESS", "inserted_count": 0}
    embeddings = self._embed_texts_sync(content_list)
    records = self._records(
        contents=content_list,
        embeddings=embeddings,
        metadatas=metadata_list,
        ids=id_list,
    )
    self._client.upsert(
        collection_name=self._settings.collection_name,
        data=records,
        timeout=self._settings.timeout,
    )
    return {"status": "SUCCESS", "inserted_count": len(records)}

  async def add_texts_async(
      self,
      contents: Iterable[str],
      *,
      metadatas: Iterable[Mapping[str, object]] | None = None,
      ids: Iterable[str] | None = None,
  ) -> dict[str, object]:
    """Asynchronously embed and upsert text content into Milvus."""
    content_list, metadata_list, id_list = self._prepare_inputs(
        contents, metadatas, ids
    )
    if not content_list:
      return {"status": "SUCCESS", "inserted_count": 0}
    embeddings = await self._embed_texts(content_list)
    records = self._records(
        contents=content_list,
        embeddings=embeddings,
        metadatas=metadata_list,
        ids=id_list,
    )
    await asyncio.to_thread(
        self._client.upsert,
        collection_name=self._settings.collection_name,
        data=records,
        timeout=self._settings.timeout,
    )
    return {"status": "SUCCESS", "inserted_count": len(records)}

  def _search_result(self, hits: Sequence[Mapping[str, object]]) -> dict:
    rows = []
    for hit in hits:
      entity = hit.get("entity", {})
      if not isinstance(entity, Mapping):
        entity = {}
      rows.append({
          "id": hit.get("id") or entity.get(self._settings.id_field),
          "content": entity.get(self._settings.content_field, ""),
          "source": entity.get(self._settings.source_field, ""),
          "metadata": entity.get(self._settings.metadata_field, {}),
          "distance": hit.get("distance"),
      })
    return {"status": "SUCCESS", "rows": rows}

  def similarity_search(
      self,
      query: str,
      *,
      top_k: int | None = None,
      filter_expr: str | None = None,
  ) -> dict:
    """Perform semantic similarity search over the Milvus vector store."""
    query_embedding = self._embed_texts_sync([query])[0]
    search_kwargs = self._search_kwargs(query_embedding, top_k, filter_expr)
    results = self._client.search(**search_kwargs)
    return self._search_result(results[0] if results else [])

  def _search_kwargs(
      self,
      query_embedding: Sequence[float],
      top_k: int | None,
      filter_expr: str | None,
  ) -> dict[str, object]:
    search_kwargs: dict[str, object] = {
        "collection_name": self._settings.collection_name,
        "data": [list(query_embedding)],
        "limit": top_k or self._settings.search_top_k,
        "output_fields": [
            self._settings.id_field,
            self._settings.content_field,
            self._settings.source_field,
            self._settings.metadata_field,
        ],
        "anns_field": self._settings.vector_field,
        "search_params": {"metric_type": self._settings.metric_type},
        "timeout": self._settings.timeout,
    }
    if filter_expr:
      search_kwargs["filter"] = filter_expr
    return search_kwargs

  async def similarity_search_async(
      self,
      query: str,
      *,
      top_k: int | None = None,
      filter_expr: str | None = None,
  ) -> dict:
    """Asynchronously perform semantic similarity search over Milvus."""
    query_embedding = (await self._embed_texts([query]))[0]
    search_kwargs = self._search_kwargs(query_embedding, top_k, filter_expr)
    results = await asyncio.to_thread(
        self._client.search,
        **search_kwargs,
    )
    return self._search_result(results[0] if results else [])

  async def close(self) -> None:
    await asyncio.to_thread(self._client.close)


class MilvusSimilaritySearchTool(BaseRetrievalTool):
  """Retrieval tool that performs similarity search over Milvus."""

  def __init__(
      self,
      *,
      vector_store: MilvusVectorStore,
      description: str,
      name: str = "similarity_search",
  ):
    super().__init__(name=name, description=description)
    self._vector_store = vector_store

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    _ = tool_context
    return await self._vector_store.similarity_search_async(args["query"])


class MilvusToolset(BaseToolset):
  """Toolset for retrieving context from a Milvus vector store."""

  def __init__(
      self,
      *,
      embedding_function: EmbeddingFunction | None = None,
      vector_store: MilvusVectorStore | None = None,
      milvus_tool_settings: MilvusToolSettings | None = None,
      tool_filter: ToolPredicate | list[str] | None = None,
  ):
    super().__init__(
        tool_filter=tool_filter,
        tool_name_prefix=DEFAULT_MILVUS_TOOL_NAME_PREFIX,
    )
    self._tool_settings = milvus_tool_settings
    if vector_store is None:
      if (
          milvus_tool_settings is None
          or milvus_tool_settings.vector_store_settings is None
      ):
        raise ValueError(
            "milvus_tool_settings.vector_store_settings is required when "
            "vector_store is not provided."
        )
      if embedding_function is None:
        raise ValueError(
            "embedding_function is required when vector_store is not provided."
        )
      vector_store = MilvusVectorStore(
          embedding_function=embedding_function,
          settings=milvus_tool_settings.vector_store_settings,
      )
    elif (
        milvus_tool_settings is None
        or milvus_tool_settings.vector_store_settings is None
    ):
      existing_settings = (
          milvus_tool_settings.model_dump() if milvus_tool_settings else {}
      )
      existing_settings["vector_store_settings"] = (
          vector_store._settings  # pylint: disable=protected-access
      )
      milvus_tool_settings = MilvusToolSettings(**existing_settings)
    self._vector_store = vector_store
    self._tool_settings = milvus_tool_settings

  @override
  async def get_tools(
      self, readonly_context: ReadonlyContext | None = None
  ) -> list[BaseTool]:
    """Get Milvus tools from the toolset."""
    all_tools: list[BaseTool] = [
        MilvusSimilaritySearchTool(
            vector_store=self._vector_store,
            description=self._tool_settings.similarity_search_description,
        )
    ]
    return [
        tool
        for tool in all_tools
        if self._is_tool_selected(tool, readonly_context)
    ]

  @override
  async def close(self):
    await self._vector_store.close()
