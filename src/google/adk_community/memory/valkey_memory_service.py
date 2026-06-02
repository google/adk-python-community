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

"""Valkey-backed memory service for ADK using valkey-glide client."""

from __future__ import annotations

import json
import logging
import time
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry

from .utils import extract_text_from_event

if TYPE_CHECKING:
  from google.adk.sessions.session import Session

logger = logging.getLogger('google_adk.' + __name__)


class ValkeyMemoryServiceConfig(BaseModel):
  """Configuration for ValkeyMemoryService.

  Attributes:
      search_top_k: Maximum number of memories to retrieve per search.
      key_prefix: Prefix for all Valkey keys to avoid collisions.
      ttl_seconds: Optional TTL for memory entries in seconds. None means
          no expiration.
  """

  search_top_k: int = Field(default=10, ge=1, le=100)
  key_prefix: str = Field(default="adk:memory")
  ttl_seconds: Optional[int] = Field(default=None, ge=1)


class ValkeyMemoryService(BaseMemoryService):
  """Memory service implementation using Valkey as the backend.

  Uses valkey-glide client for communication with Valkey server.
  Memories are stored as JSON strings in Valkey lists, indexed by
  app_name and user_id for efficient retrieval.

  Example usage:

      from glide import GlideClientConfiguration, NodeAddress, GlideClient

      config = GlideClientConfiguration(
          addresses=[NodeAddress(host="localhost", port=6379)],
          client_name="adk_memory_client",
      )
      client = await GlideClient.create(config)
      service = ValkeyMemoryService(client=client)

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

  def _memory_list_key(self, app_name: str, user_id: str) -> str:
    """Generate the Valkey key for a user's memory list."""
    return f"{self._config.key_prefix}:{app_name}:{user_id}:entries"

  def _serialize_memory(
      self, event, content_text: str, session
  ) -> str:
    """Serialize an event into a JSON string for storage."""
    memory_data = {
        "content": content_text,
        "author": event.author,
        "timestamp": event.timestamp,
        "session_id": session.id,
        "event_id": event.id,
        "app_name": session.app_name,
        "user_id": session.user_id,
        "created_at": time.time(),
    }
    return json.dumps(memory_data)

  @override
  async def add_session_to_memory(self, session: Session):
    """Add a session's events to Valkey memory storage."""
    memories_added = 0
    list_key = self._memory_list_key(session.app_name, session.user_id)

    for event in session.events:
      content_text = extract_text_from_event(event)
      if not content_text:
        continue

      try:
        serialized = self._serialize_memory(event, content_text, session)
        await self._client.rpush(list_key, [serialized])
        memories_added += 1
        logger.debug("Added memory for event %s", event.id)
      except Exception as e:
        logger.error(
            "Failed to add memory for event %s: %s", event.id, e
        )

    if self._config.ttl_seconds and memories_added > 0:
      try:
        await self._client.expire(list_key, self._config.ttl_seconds)
      except Exception as e:
        logger.error("Failed to set TTL on key %s: %s", list_key, e)

    logger.info(
        "Added %d memories from session %s", memories_added, session.id
    )

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Search for memories matching the query.

    Performs a simple text-based search over stored memories for
    the given app and user. Retrieves all stored memories and
    filters them by checking if the query terms appear in the
    content.

    Args:
        app_name: The application name to scope the search.
        user_id: The user ID to scope the search.
        query: The search query string.

    Returns:
        SearchMemoryResponse containing matching MemoryEntry objects.
    """
    list_key = self._memory_list_key(app_name, user_id)

    try:
      # Retrieve all memories for this user/app
      raw_memories = await self._client.lrange(list_key, 0, -1)

      if not raw_memories:
        return SearchMemoryResponse(memories=[])

      memories = []
      query_lower = query.lower()
      query_terms = query_lower.split()

      for raw in raw_memories:
        try:
          raw_str = (
              raw.decode("utf-8") if isinstance(raw, bytes) else raw
          )
          memory_data = json.loads(raw_str)
          content_text = memory_data.get("content", "")

          # Simple term-matching search
          content_lower = content_text.lower()
          if any(term in content_lower for term in query_terms):
            content = types.Content(
                parts=[types.Part(text=content_text)]
            )
            timestamp = memory_data.get("timestamp")
            if timestamp is not None:
              timestamp = str(timestamp)
            entry = MemoryEntry(
                content=content,
                author=memory_data.get("author"),
                timestamp=timestamp,
            )
            memories.append(entry)

            if len(memories) >= self._config.search_top_k:
              break
        except (json.JSONDecodeError, KeyError) as e:
          logger.debug("Failed to parse memory entry: %s", e)
          continue

      logger.info(
          "Found %d memories for query: '%s'", len(memories), query
      )
      return SearchMemoryResponse(memories=memories)

    except Exception as e:
      logger.error("Failed to search memories: %s", e)
      return SearchMemoryResponse(memories=[])

  async def close(self):
    """Close the memory service.

    Note: This does NOT close the underlying Valkey client, as
    the client lifecycle is managed by the caller.
    """
    pass
