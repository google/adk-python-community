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

Uses the Valkey Search module (FT.CREATE / FT.SEARCH) for full-text
search over stored memories.
"""

from __future__ import annotations

import logging
import time
from typing import Optional
from typing import TYPE_CHECKING
import uuid

from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from .utils import extract_text_from_event

if TYPE_CHECKING:
  from google.adk.sessions.session import Session

logger = logging.getLogger("google_adk." + __name__)


class ValkeyMemoryServiceConfig(BaseModel):
  """Configuration for ValkeyMemoryService.

  Attributes:
      search_top_k: Maximum number of memories to retrieve per search.
      key_prefix: Prefix for all Valkey keys to avoid collisions.
      index_name: Name of the Valkey Search index.
      ttl_seconds: Optional TTL for memory entries in seconds. None means
          no expiration.
  """

  search_top_k: int = Field(default=10, ge=1, le=100)
  key_prefix: str = Field(default="adk:memory")
  index_name: str = Field(default="adk_memory_idx")
  ttl_seconds: Optional[int] = Field(default=None, ge=1)


class ValkeyMemoryService(BaseMemoryService):
  """Memory service implementation using Valkey with the Search module.

  Uses valkey-glide client for communication with Valkey server and the
  Valkey Search module (FT.CREATE / FT.SEARCH) for full-text search
  over stored memories.

  Memories are stored as Valkey Hash keys with fields: content, author,
  timestamp, session_id, event_id, app_name, user_id, created_at. A
  full-text search index is created over the content field, with TAG
  fields for app_name and user_id to enable scoped queries.

  Example usage:

      from glide import GlideClientConfiguration, NodeAddress, GlideClient

      config = GlideClientConfiguration(
          addresses=[NodeAddress(host="localhost", port=6379)],
          client_name="adk_memory_client",
      )
      client = await GlideClient.create(config)
      service = ValkeyMemoryService(client=client)
      await service.create_index()

  """

  def __init__(
      self,
      client,
      config: Optional[ValkeyMemoryServiceConfig] = None,
  ):
    """Initializes the Valkey memory service.

    Args:
        client: A connected valkey-glide GlideClient or
            GlideClusterClient instance. The caller is responsible
            for creating and managing the client lifecycle.
        config: Optional ValkeyMemoryServiceConfig instance.
            If None, uses defaults.
    """
    if client is None:
      raise ValueError(
          "client is required. Provide a connected valkey-glide "
          "GlideClient or GlideClusterClient instance."
      )
    self._client = client
    self._config = config or ValkeyMemoryServiceConfig()
    self._index_created = False

  async def create_index(self):
    """Create the Valkey Search index if it does not already exist.

    Creates a full-text search index with:
    - content: TEXT field for full-text search
    - app_name: TAG field for filtering by application
    - user_id: TAG field for filtering by user
    - author: TAG field for filtering by author
    - timestamp: NUMERIC field for sorting

    This method is idempotent — if the index already exists, it
    will log a debug message and return without error.
    """
    from glide import DataType
    from glide import ft
    from glide import FtCreateOptions
    from glide import NumericField
    from glide import TagField
    from glide import TextField

    schema = [
        TextField("content"),
        TagField("app_name"),
        TagField("user_id"),
        TagField("author"),
        NumericField("timestamp", sortable=True),
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

  @override
  async def add_session_to_memory(self, session: Session):
    """Add a session's events to Valkey memory storage.

    Each event with text content is stored as a separate Valkey Hash
    key with the configured prefix, making it automatically indexed
    by the search module.
    """
    if not self._index_created:
      await self.create_index()

    memories_added = 0

    for event in session.events:
      content_text = extract_text_from_event(event)
      if not content_text:
        continue

      hash_key = self._memory_hash_key()
      field_values = {
          "content": content_text,
          "author": event.author or "",
          "timestamp": str(event.timestamp) if event.timestamp else "0",
          "session_id": session.id,
          "event_id": event.id or "",
          "app_name": session.app_name,
          "user_id": session.user_id,
          "created_at": str(time.time()),
      }

      try:
        await self._client.hset(hash_key, field_values)
        memories_added += 1
        logger.debug("Added memory for event %s at key %s", event.id, hash_key)

        if self._config.ttl_seconds:
          await self._client.expire(hash_key, self._config.ttl_seconds)
      except Exception as e:
        logger.error("Failed to add memory for event %s: %s", event.id, e)

    logger.info("Added %d memories from session %s", memories_added, session.id)

  def _build_search_query(self, app_name: str, user_id: str, query: str) -> str:
    """Build an FT.SEARCH query string with filters.

    Constructs a query that:
    - Filters by app_name and user_id using TAG filters
    - Searches content using full-text search

    Args:
        app_name: Application name filter.
        user_id: User ID filter.
        query: The user's search query text.

    Returns:
        A Valkey Search query string.
    """
    # Escape special characters in TAG values
    escaped_app = app_name.replace("-", "\\-")
    escaped_user = user_id.replace("-", "\\-")

    # Build full-text query with TAG filters
    # Use @field:{value} for TAG filtering and plain text for content
    tag_filter = f"@app_name:{{{escaped_app}}} @user_id:{{{escaped_user}}}"

    # Escape special FT.SEARCH characters in the query text
    search_chars = r'@!{}()|-=><~*:;$["\]^'
    escaped_query = query
    for ch in search_chars:
      escaped_query = escaped_query.replace(ch, f"\\{ch}")

    return f"{tag_filter} {escaped_query}"

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Search for memories matching the query using Valkey Search.

    Uses FT.SEARCH with the Valkey Search module for full-text search.
    Results are filtered by app_name and user_id, and the query is
    matched against the content field.

    Args:
        app_name: The application name to scope the search.
        user_id: The user ID to scope the search.
        query: The search query string.

    Returns:
        SearchMemoryResponse containing matching MemoryEntry objects.
    """
    from glide import ft
    from glide import FtSearchLimit
    from glide import FtSearchOptions

    if not self._index_created:
      await self.create_index()

    try:
      search_query = self._build_search_query(app_name, user_id, query)
      options = FtSearchOptions(
          limit=FtSearchLimit(0, self._config.search_top_k),
      )

      result = await ft.search(
          self._client,
          self._config.index_name,
          search_query,
          options,
      )

      if not result or len(result) < 2:
        return SearchMemoryResponse(memories=[])

      # result is [count, {doc_id: {field: value, ...}, ...}]
      doc_count = result[0]
      if doc_count == 0:
        return SearchMemoryResponse(memories=[])

      memories = []
      doc_map = result[1] if len(result) > 1 else {}

      for doc_id, fields in doc_map.items():
        try:
          content_text = self._decode(fields.get(b"content", b""))
          if not content_text:
            continue

          author = self._decode(fields.get(b"author", b"")) or None
          timestamp_raw = self._decode(fields.get(b"timestamp", b"0"))
          # Numeric fields may return "12345.0"; normalize to int string
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
