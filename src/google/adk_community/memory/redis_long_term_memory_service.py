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

"""Redis Long-Term Memory Service for ADK.

This module provides integration with the Redis Agent Memory Server,
offering production-grade long-term memory with automatic summarization,
topic/entity extraction, and recency-boosted search.

Note: The classes were renamed from RedisAgentMemoryService to
RedisLongTermMemoryService to better reflect their purpose of managing
long-term memory via the Agent Memory Server.
"""

from __future__ import annotations

from functools import cached_property
import logging
from typing import Literal
from typing import Optional
from typing import TYPE_CHECKING

from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from google.adk_community.memory.utils import extract_text_from_event

if TYPE_CHECKING:
  from google.adk.sessions.session import Session

logger = logging.getLogger("google_adk." + __name__)


class RedisLongTermMemoryServiceConfig(BaseModel):
  """Configuration for Redis Long-Term Memory Service.

  Attributes:
      api_base_url: Base URL of the Agent Memory Server.
      timeout: HTTP request timeout in seconds.
      default_namespace: Default namespace for memory operations.
      search_top_k: Maximum number of memories to retrieve per search.
      distance_threshold: Maximum distance threshold for search results (0.0-1.0).
      recency_boost: Enable recency-aware re-ranking of search results.
      semantic_weight: Weight for semantic similarity in recency boosting (0.0-1.0).
      recency_weight: Weight for recency score in recency boosting (0.0-1.0).
      freshness_weight: Weight for freshness component within recency score.
      novelty_weight: Weight for novelty component within recency score.
      half_life_last_access_days: Half-life in days for last_accessed decay.
      half_life_created_days: Half-life in days for created_at decay.
      extraction_strategy: Memory extraction strategy (discrete, summary, preferences, custom).
      extraction_strategy_config: Additional configuration for the extraction strategy.
      model_name: Model name for context window management and summarization.
      context_window_max: Maximum context window tokens (overrides model default).
  """

  api_base_url: str = Field(default="http://localhost:8000")
  timeout: float = Field(default=30.0, gt=0.0)
  default_namespace: Optional[str] = None
  search_top_k: int = Field(default=10, ge=1)
  distance_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
  recency_boost: bool = True
  semantic_weight: float = Field(default=0.8, ge=0.0, le=1.0)
  recency_weight: float = Field(default=0.2, ge=0.0, le=1.0)
  freshness_weight: float = Field(default=0.6, ge=0.0, le=1.0)
  novelty_weight: float = Field(default=0.4, ge=0.0, le=1.0)
  half_life_last_access_days: float = Field(default=7.0, gt=0.0)
  half_life_created_days: float = Field(default=30.0, gt=0.0)
  extraction_strategy: Literal[
      "discrete", "summary", "preferences", "custom"
  ] = "discrete"
  extraction_strategy_config: dict = Field(default_factory=dict)
  model_name: Optional[str] = None
  context_window_max: Optional[int] = Field(default=None, ge=1)


class RedisLongTermMemoryService(BaseMemoryService):
  """Long-term memory service implementation using Redis Agent Memory Server.

  This service provides production-grade memory capabilities including:
  - Two-tier memory architecture (working memory + long-term memory)
  - Automatic memory extraction (semantic facts, episodic events, preferences)
  - Topic and entity extraction
  - Auto-summarization when context window is exceeded
  - Recency-boosted semantic search
  - Deduplication and memory compaction
  - https://github.com/redis/agent-memory-server
  Requires the `agent-memory-client` package to be installed.

  Example:
      ```python
      from google.adk_community.memory import (
          RedisLongTermMemoryService,
          RedisLongTermMemoryServiceConfig,
      )

      config = RedisLongTermMemoryServiceConfig(
          api_base_url="http://localhost:8000",
          default_namespace="my_app",
          recency_boost=True,
      )
      memory_service = RedisLongTermMemoryService(config=config)

      # Use with ADK agent
      agent = Agent(
          name="my_agent",
          memory_service=memory_service,
      )
      ```
  """

  def __init__(self, config: Optional[RedisLongTermMemoryServiceConfig] = None):
    """Initialize the Redis Long-Term Memory Service.

    Args:
        config: Configuration for the service. If None, uses defaults.

    Raises:
        ImportError: If agent-memory-client package is not installed.
    """
    self._config = config or RedisLongTermMemoryServiceConfig()

  @cached_property
  def _client(self):
    """Lazily initialize and return the MemoryAPIClient."""
    try:
      from agent_memory_client import MemoryAPIClient
      from agent_memory_client import MemoryClientConfig
    except ImportError as e:
      raise ImportError(
          "agent-memory-client package is required for"
          " RedisLongTermMemoryService. Install it with: pip install"
          " agent-memory-client"
      ) from e

    client_config = MemoryClientConfig(
        base_url=self._config.api_base_url,
        timeout=self._config.timeout,
        default_namespace=self._config.default_namespace,
        default_model_name=self._config.model_name,
        default_context_window_max=self._config.context_window_max,
    )
    return MemoryAPIClient(client_config)

  def _build_working_memory(self, session: "Session"):
    """Convert ADK Session to WorkingMemory for the Agent Memory Server."""
    from agent_memory_client.models import MemoryMessage
    from agent_memory_client.models import MemoryStrategyConfig
    from agent_memory_client.models import WorkingMemory

    messages = []
    for event in session.events:
      text = extract_text_from_event(event)
      if not text:
        continue
      role = "user" if event.author == "user" else "assistant"
      messages.append(MemoryMessage(role=role, content=text))

    strategy_config = MemoryStrategyConfig(
        strategy=self._config.extraction_strategy,
        config=self._config.extraction_strategy_config,
    )

    return WorkingMemory(
        session_id=session.id,
        namespace=self._config.default_namespace or session.app_name,
        user_id=session.user_id,
        messages=messages,
        long_term_memory_strategy=strategy_config,
    )

  @override
  async def add_session_to_memory(self, session: "Session"):
    """Add a session's events to the Agent Memory Server.

    Converts ADK Session events to WorkingMemory messages and stores them
    in the Agent Memory Server. The server will automatically:
    - Extract semantic and episodic memories based on the configured strategy
    - Perform topic and entity extraction
    - Summarize context when the token limit is exceeded
    - Promote memories to long-term storage via background tasks

    Args:
        session: The ADK Session containing events to store.
    """
    try:
      working_memory = self._build_working_memory(session)

      if not working_memory.messages:
        logger.debug("No messages to store for session %s", session.id)
        return

      response = await self._client.put_working_memory(
          session_id=session.id,
          memory=working_memory,
          user_id=session.user_id,
      )

      logger.info(
          "Stored %d messages for session %s (context: %.1f%% used)",
          len(working_memory.messages),
          session.id,
          response.context_percentage_total_used or 0,
      )

    except Exception as e:
      logger.error(
          "Failed to add session %s to memory: %s",
          session.id,
          e,
      )

  def _build_recency_config(self):
    """Build RecencyConfig from service configuration."""
    from agent_memory_client.models import RecencyConfig

    return RecencyConfig(
        recency_boost=self._config.recency_boost,
        semantic_weight=self._config.semantic_weight,
        recency_weight=self._config.recency_weight,
        freshness_weight=self._config.freshness_weight,
        novelty_weight=self._config.novelty_weight,
        half_life_last_access_days=self._config.half_life_last_access_days,
        half_life_created_days=self._config.half_life_created_days,
    )

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Search for memories using the Agent Memory Server.

    Performs semantic search against long-term memory with optional
    recency boosting. Results are filtered by namespace (derived from
    app_name) and user_id.

    Args:
        app_name: The application name (used as namespace if not configured).
        user_id: The user ID to filter memories.
        query: The search query for semantic matching.

    Returns:
        SearchMemoryResponse containing matching MemoryEntry objects.
    """
    try:
      recency_config = (
          self._build_recency_config() if self._config.recency_boost else None
      )

      namespace = self._config.default_namespace or app_name

      results = await self._client.search_long_term_memory(
          text=query,
          namespace={"eq": namespace},
          user_id={"eq": user_id},
          distance_threshold=self._config.distance_threshold,
          recency=recency_config,
          limit=self._config.search_top_k,
      )

      memories = []
      for record in results.memories:
        content = types.Content(parts=[types.Part(text=record.text)])
        memory_entry = MemoryEntry(content=content)
        memories.append(memory_entry)

      logger.info(
          "Found %d memories for query '%s' (namespace=%s, user=%s)",
          len(memories),
          query[:50],
          namespace,
          user_id,
      )
      return SearchMemoryResponse(memories=memories)

    except Exception as e:
      logger.error("Failed to search memories: %s", e)
      return SearchMemoryResponse(memories=[])

  async def close(self):
    """Close the memory service and cleanup resources."""
    if (
        "_client" in self.__dict__
    ):  # Check for initialized client without triggering cached_property
      await self._client.close()
      # Clear the cached property
      del self._client
