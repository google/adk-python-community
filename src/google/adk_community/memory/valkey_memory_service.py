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

"""Valkey-backed memory service for ADK using valkey-glide client.

Uses the Valkey Search module with vector similarity search (HNSW)
for semantic memory retrieval, analogous to VertexAiRagMemoryService.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
import logging
import struct
import time
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
import uuid

from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from typing_extensions import override

from .utils import extract_text_from_event

if TYPE_CHECKING:
  from glide import GlideClient
  from glide import GlideClusterClient
  from google.adk.sessions.session import Session

logger = logging.getLogger("google_adk." + __name__)

# Type alias for the embedding function.
# It takes a list of text strings and returns a list of float vectors.
EmbeddingFunction = Callable[[list[str]], Awaitable[list[list[float]]]]


class ValkeyMemoryServiceConfig(BaseModel):
  """Configuration for ValkeyMemoryService.

  Attributes:
      similarity_top_k: Maximum number of memories to retrieve per
          search (KNN parameter).
      vector_distance_threshold: Maximum distance threshold for
          filtering results. Results with distance greater than this
          are excluded. None means no threshold filtering.
      embedding_dimensions: Dimensionality of the embedding vectors.
      key_prefix: Prefix for all Valkey keys to avoid collisions.
      index_name: Name of the Valkey Search index.
      distance_metric: Distance metric for vector similarity.
          One of 'COSINE', 'L2', or 'IP' (inner product).
      ttl_seconds: Optional TTL for memory entries in seconds.
          None means no expiration.
  """

  similarity_top_k: int = Field(default=10, ge=1, le=1000)
  vector_distance_threshold: Optional[float] = Field(default=None, ge=0.0)
  embedding_dimensions: int = Field(default=768, ge=1)
  key_prefix: str = Field(default="adk:memory")
  index_name: str = Field(default="adk_memory_idx")
  distance_metric: str = Field(default="COSINE")
  ttl_seconds: Optional[int] = Field(default=None, ge=1)

  @field_validator("distance_metric")
  @classmethod
  def _validate_distance_metric(cls, v):
    """Validate distance_metric is one of the allowed values."""
    allowed = {"COSINE", "L2", "IP"}
    if v.upper() not in allowed:
      raise ValueError(f"distance_metric must be one of {allowed}, got '{v}'")
    return v.upper()


class ValkeyMemoryService(BaseMemoryService):
  """Memory service using Valkey Search module with vector similarity.

  Uses valkey-glide client for communication with Valkey server and the
  Valkey Search module for vector-based semantic search over stored
  memories. This provides functionality analogous to
  VertexAiRagMemoryService but backed by Valkey infrastructure.

  Memories are stored as Valkey Hash keys with fields: content, author,
  timestamp, session_id, event_id, app_name, user_id, created_at, and
  an embedding vector field. A vector search index (HNSW) is created
  for approximate nearest neighbor retrieval, with TAG fields for
  app_name and user_id to enable scoped queries.

  Example usage:

      from glide import GlideClientConfiguration, NodeAddress, GlideClient

      # IMPORTANT: Set client_name for observability in CLIENT LIST,
      # monitoring dashboards, and CloudWatch metrics.
      config = GlideClientConfiguration(
          addresses=[NodeAddress(host="localhost", port=6379)],
          client_name="adk_memory_client",
      )
      client = await GlideClient.create(config)

      async def my_embed_fn(texts: list[str]) -> list[list[float]]:
          # Your embedding logic here (OpenAI, Gemini, etc.)
          ...

      service = ValkeyMemoryService(
          client=client,
          embedding_function=my_embed_fn,
      )
      await service.create_index()

  """

  def __init__(
      self,
      client: Union["GlideClient", "GlideClusterClient"],
      embedding_function: EmbeddingFunction,
      config: Optional[ValkeyMemoryServiceConfig] = None,
  ):
    """Initializes the Valkey memory service.

    Args:
        client: A connected valkey-glide GlideClient or
            GlideClusterClient instance. The caller is responsible
            for creating and managing the client lifecycle.
        embedding_function: An async callable that takes a list of
            text strings and returns a list of embedding vectors
            (list of floats). Users provide their own embedding
            model (e.g., OpenAI, Google Gemini, sentence-transformers).
        config: Optional ValkeyMemoryServiceConfig instance.
            If None, uses defaults.
    """
    if client is None:
      raise ValueError(
          "client is required. Provide a connected valkey-glide "
          "GlideClient or GlideClusterClient instance."
      )
    if embedding_function is None:
      raise ValueError(
          "embedding_function is required. Provide an async callable "
          "that takes list[str] and returns list[list[float]]."
      )
    self._client = client
    self._embedding_function = embedding_function
    self._config = config or ValkeyMemoryServiceConfig()
    self._index_created = False
    self._index_lock = asyncio.Lock()

  async def create_index(self):
    """Create the Valkey Search index if it does not already exist.

    Creates a vector search index (HNSW) with:
    - embedding: VECTOR field (HNSW, FLOAT32) for similarity search
    - content: TEXT field for optional full-text filtering
    - app_name: TAG field for filtering by application
    - user_id: TAG field for filtering by user
    - author: TAG field for filtering by author

    This method is idempotent — if the index already exists, it
    will log a debug message and return without error.
    """
    from glide import DataType
    from glide import DistanceMetricType
    from glide import ft
    from glide import FtCreateOptions
    from glide import TagField
    from glide import TextField
    from glide import VectorAlgorithm
    from glide import VectorField
    from glide import VectorFieldAttributesHnsw
    from glide import VectorType

    distance_map = {
        "COSINE": DistanceMetricType.COSINE,
        "L2": DistanceMetricType.L2,
        "IP": DistanceMetricType.IP,
    }
    distance_metric = distance_map.get(
        self._config.distance_metric.upper(), DistanceMetricType.COSINE
    )

    schema = [
        VectorField(
            "embedding",
            algorithm=VectorAlgorithm.HNSW,
            attributes=VectorFieldAttributesHnsw(
                dimensions=self._config.embedding_dimensions,
                distance_metric=distance_metric,
                type=VectorType.FLOAT32,
            ),
        ),
        TextField("content"),
        TagField("app_name"),
        TagField("user_id"),
        TagField("author"),
    ]

    options = FtCreateOptions(
        data_type=DataType.HASH,
        prefixes=[f"{self._config.key_prefix}:"],
    )

    try:
      await ft.create(
          self._client,
          self._config.index_name,
          schema,
          options,
      )
      self._index_created = True
      logger.info("Created search index: %s", self._config.index_name)
    except Exception as e:
      error_msg = str(e).lower()
      if "index already exists" in error_msg or "exists" in error_msg:
        self._index_created = True
        logger.debug(
            "Search index already exists: %s",
            self._config.index_name,
        )
      else:
        raise

  def _memory_hash_key(self) -> str:
    """Generate a unique Valkey hash key for a memory entry."""
    unique_id = uuid.uuid4().hex[:12]
    return f"{self._config.key_prefix}:{unique_id}"

  async def _ensure_index(self):
    """Ensure the search index exists, using double-check locking."""
    if self._index_created:
      return
    async with self._index_lock:
      if not self._index_created:
        await self.create_index()

  @staticmethod
  def _vector_to_bytes(vector: list[float]) -> bytes:
    """Convert a list of floats to a binary blob for Valkey storage."""
    return struct.pack(f"<{len(vector)}f", *vector)

  @override
  async def add_session_to_memory(self, session: Session):
    """Add a session's events to Valkey memory storage.

    Extracts text from session events, generates embeddings using the
    configured embedding function, and stores each event as a Valkey
    Hash with the embedding vector for later similarity search.
    """
    await self._ensure_index()
    await self._ingest_events(
        events=session.events,
        app_name=session.app_name,
        user_id=session.user_id,
        session_id=session.id,
    )

  @override
  async def add_events_to_memory(
      self,
      *,
      app_name: str,
      user_id: str,
      events: Sequence,
      session_id: str | None = None,
      custom_metadata=None,
  ) -> None:
    """Adds an incremental list of events to memory.

    Generates embeddings and stores each event with text content.
    This is useful for persisting only a subset of events (e.g.,
    the latest turn) without re-ingesting the full session.

    Args:
        app_name: The application name for memory scope.
        user_id: The user ID for memory scope.
        events: The events to add to memory.
        session_id: Optional session ID for partitioning.
        custom_metadata: Optional metadata (unused currently).
    """
    await self._ensure_index()
    await self._ingest_events(
        events=events,
        app_name=app_name,
        user_id=user_id,
        session_id=session_id or "",
    )

  async def _ingest_events(
      self,
      events: Sequence,
      app_name: str,
      user_id: str,
      session_id: str,
  ) -> None:
    """Shared ingestion logic for add_session_to_memory and add_events_to_memory.

    Extracts text from events, generates embeddings in batch, and
    stores each event as a Valkey Hash using pipelined Batch commands
    for efficiency.

    Args:
        events: The events to ingest.
        app_name: The application name for memory scope.
        user_id: The user ID for memory scope.
        session_id: The session ID for partitioning.
    """
    from glide import Batch

    # Collect texts and their corresponding events
    texts = []
    valid_events = []
    for event in events:
      content_text = extract_text_from_event(event)
      if content_text:
        texts.append(content_text)
        valid_events.append(event)

    if not texts:
      logger.debug("No text events to ingest")
      return

    # Generate embeddings for all texts in one batch
    try:
      embeddings = await self._embedding_function(texts)
    except Exception as e:
      logger.error("Failed to generate embeddings: %s", e)
      return

    if len(embeddings) != len(texts):
      logger.error(
          "Embedding function returned %d vectors for %d texts",
          len(embeddings),
          len(texts),
      )
      return

    # Build a Batch to pipeline all hset + expire calls
    batch = Batch(is_atomic=False)
    hash_keys = []
    for event, content_text, embedding in zip(valid_events, texts, embeddings):
      hash_key = self._memory_hash_key()
      hash_keys.append(hash_key)
      field_values = {
          "content": content_text,
          "author": event.author or "",
          "timestamp": str(event.timestamp) if event.timestamp else "0",
          "session_id": session_id,
          "event_id": event.id or "",
          "app_name": app_name,
          "user_id": user_id,
          "created_at": str(time.time()),
          "embedding": self._vector_to_bytes(embedding),
      }

      batch.hset(hash_key, field_values)
      if self._config.ttl_seconds is not None:
        batch.expire(hash_key, self._config.ttl_seconds)

    try:
      await self._client.exec(batch, raise_on_error=True)
      logger.info("Added %d memories via batch pipeline", len(hash_keys))
    except Exception as e:
      logger.error("Failed to execute batch pipeline: %s", e)
      raise RuntimeError(f"Memory ingestion failed: {e}") from e

  # Characters that must be escaped in Valkey Search TAG field values.
  # Includes '?' which is a single-character wildcard glob in TAG queries.
  _TAG_SPECIAL_CHARS = set(r',.<>{}[]"' + r"':;!@#$%^&*()-+=~|/\\ ?")

  @staticmethod
  def _escape_tag_value(value: str) -> str:
    """Escape special characters for Valkey Search TAG field queries.

    Per the Valkey Search query syntax, TAG values must have
    metacharacters escaped with a backslash.
    """
    escaped = []
    for ch in value:
      if ch in ValkeyMemoryService._TAG_SPECIAL_CHARS:
        escaped.append(f"\\{ch}")
      else:
        escaped.append(ch)
    return "".join(escaped)

  def _build_knn_query(self, app_name: str, user_id: str, top_k: int) -> str:
    """Build a KNN search query with TAG pre-filters.

    Args:
        app_name: Application name filter.
        user_id: User ID filter.
        top_k: Number of nearest neighbors to retrieve.

    Returns:
        A Valkey Search KNN query string.
    """
    escaped_app = self._escape_tag_value(app_name)
    escaped_user = self._escape_tag_value(user_id)

    # KNN query with pre-filter: filter first, then KNN on results
    return (
        f"(@app_name:{{{escaped_app}}} "
        f"@user_id:{{{escaped_user}}})"
        f"=>[KNN {top_k} @embedding $query_vec]"
    )

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Search for memories using vector similarity (KNN).

    Generates an embedding for the query text, then performs a KNN
    search using FT.SEARCH with pre-filtering by app_name and
    user_id. Results are ranked by vector distance (lower = more
    similar for COSINE).

    Args:
        app_name: The application name to scope the search.
        user_id: The user ID to scope the search.
        query: The search query string.

    Returns:
        SearchMemoryResponse containing matching MemoryEntry objects,
        ordered by similarity.
    """
    from glide import ft
    from glide import FtSearchOptions

    await self._ensure_index()

    # Generate embedding for the query
    try:
      query_embeddings = await self._embedding_function([query])
      query_embedding = query_embeddings[0]
    except Exception as e:
      logger.error("Failed to generate query embedding: %s", e)
      return SearchMemoryResponse(memories=[])

    query_vec_bytes = self._vector_to_bytes(query_embedding)

    try:
      search_query = self._build_knn_query(
          app_name, user_id, self._config.similarity_top_k
      )
      options = FtSearchOptions(
          params={"query_vec": query_vec_bytes},
      )

      result = await ft.search(
          self._client,
          self._config.index_name,
          search_query,
          options,
      )

      if not result or len(result) < 2:
        return SearchMemoryResponse(memories=[])

      doc_count = result[0]
      if doc_count == 0:
        return SearchMemoryResponse(memories=[])

      memories = []
      doc_map = result[1]

      for doc_id, fields in doc_map.items():
        try:
          # Check distance threshold if configured
          if self._config.vector_distance_threshold is not None:
            score_raw = self._decode(fields.get(b"__embedding_score", b""))
            if score_raw:
              distance = float(score_raw)
              if distance > self._config.vector_distance_threshold:
                continue

          content_text = self._decode(fields.get(b"content", b""))
          if not content_text:
            continue

          author = self._decode(fields.get(b"author", b"")) or None
          timestamp_raw = self._decode(fields.get(b"timestamp", b"0"))
          if timestamp_raw and timestamp_raw != "0":
            try:
              timestamp = str(int(float(timestamp_raw)))
            except (ValueError, TypeError):
              timestamp = timestamp_raw
          else:
            timestamp = None

          content = types.Content(parts=[types.Part(text=content_text)])
          entry = MemoryEntry(
              content=content,
              author=author,
              timestamp=timestamp,
          )
          memories.append(entry)
        except Exception as e:
          logger.debug("Failed to parse search result: %s", e)
          continue

      logger.info("Found %d memories for query: '%s'", len(memories), query)
      return SearchMemoryResponse(memories=memories)

    except Exception as e:
      logger.error("Failed to search memories: %s", e)
      return SearchMemoryResponse(memories=[])

  @staticmethod
  def _decode(value) -> str:
    """Decode bytes to string if needed."""
    if isinstance(value, bytes):
      return value.decode("utf-8")
    return str(value) if value is not None else ""

  async def close(self):
    """Close the memory service.

    Note: This does NOT close the underlying Valkey client, as
    the client lifecycle is managed by the caller.
    """
    pass
