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

"""Redis Working Memory Session Service for ADK.

This module provides session management using the Redis Agent Memory Server's
Working Memory API, offering automatic context summarization and background
memory extraction.
"""

from __future__ import annotations

from functools import cached_property
import logging
import time
from typing import Any
from typing import Literal
from typing import Optional
import uuid

from google.adk.events.event import Event
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.base_session_service import ListSessionsResponse
from google.adk.sessions.session import Session
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from google.adk_community.memory.utils import extract_text_from_event

logger = logging.getLogger("google_adk." + __name__)


class RedisWorkingMemorySessionServiceConfig(BaseModel):
  """Configuration for Redis Working Memory Session Service.

  Attributes:
      api_base_url: Base URL of the Agent Memory Server.
      timeout: HTTP request timeout in seconds.
      default_namespace: Default namespace for session operations.
      model_name: Model name for context window management and summarization.
      context_window_max: Maximum context window tokens.
      extraction_strategy: Memory extraction strategy.
      extraction_strategy_config: Additional config for extraction strategy.
      session_ttl_seconds: Optional TTL for session expiration.
  """

  api_base_url: str = Field(default="http://localhost:8000")
  timeout: float = Field(default=30.0, gt=0.0)
  default_namespace: Optional[str] = None
  model_name: Optional[str] = None
  context_window_max: Optional[int] = Field(default=None, ge=1)
  extraction_strategy: Literal[
      "discrete", "summary", "preferences", "custom"
  ] = "discrete"
  extraction_strategy_config: dict = Field(default_factory=dict)
  session_ttl_seconds: Optional[int] = Field(default=None, ge=1)


class RedisWorkingMemorySessionService(BaseSessionService):
  """Session service using Redis Agent Memory Server's Working Memory API.

  This service provides session management backed by Agent Memory Server:
  - Session storage in Working Memory
  - Automatic context summarization when token limit exceeded
  - Background memory extraction to Long-Term Memory
  - Incremental message appending
  - https://github.com/redis/agent-memory-server

  Requires the `agent-memory-client` package to be installed.

  Example:
      ```python
      from google.adk_community.sessions import (
          RedisWorkingMemorySessionService,
          RedisWorkingMemorySessionServiceConfig,
      )

      config = RedisWorkingMemorySessionServiceConfig(
          api_base_url="http://localhost:8000",
          default_namespace="my_app",
      )
      session_service = RedisWorkingMemorySessionService(config=config)

      # Use with ADK runner
      runner = Runner(
          agent=agent,
          session_service=session_service,
      )
      ```
  """

  def __init__(
      self, config: Optional[RedisWorkingMemorySessionServiceConfig] = None
  ):
    """Initialize the Redis Working Memory Session Service.

    Args:
        config: Configuration for the service. If None, uses defaults.
    """
    self._config = config or RedisWorkingMemorySessionServiceConfig()

  @cached_property
  def _client(self):
    """Lazily initialize and return the MemoryAPIClient."""
    try:
      from agent_memory_client import MemoryAPIClient
      from agent_memory_client import MemoryClientConfig
    except ImportError as e:
      raise ImportError(
          "agent-memory-client package is required for "
          "RedisWorkingMemorySessionService. "
          "Install it with: pip install agent-memory-client"
      ) from e

    client_config = MemoryClientConfig(
        base_url=self._config.api_base_url,
        timeout=self._config.timeout,
        default_namespace=self._config.default_namespace,
        default_model_name=self._config.model_name,
        default_context_window_max=self._config.context_window_max,
    )
    return MemoryAPIClient(client_config)

  def _get_namespace(self, app_name: str) -> str:
    """Get namespace from config or app_name."""
    return self._config.default_namespace or app_name

  def _event_to_message(self, event: Event):
    """Convert ADK Event to MemoryMessage."""
    from datetime import datetime
    from datetime import timezone

    from agent_memory_client.models import MemoryMessage

    text = extract_text_from_event(event)
    if not text:
      return None

    role = "user" if event.author == "user" else "assistant"
    # Convert event timestamp (float) to datetime for MemoryMessage
    created_at = datetime.fromtimestamp(event.timestamp, tz=timezone.utc)
    return MemoryMessage(role=role, content=text, created_at=created_at)

  def _working_memory_response_to_session(
      self,
      response,
      app_name: str,
      user_id: str,
  ) -> Session:
    """Convert WorkingMemoryResponse to ADK Session."""
    events = []
    for msg in response.messages or []:
      author = "user" if msg.role == "user" else response.session_id
      content = types.Content(parts=[types.Part(text=msg.content)])
      # Preserve original message timestamp if available
      timestamp = (
          msg.created_at.timestamp()
          if hasattr(msg, "created_at") and msg.created_at
          else time.time()
      )
      event = Event(
          author=author,
          content=content,
          timestamp=timestamp,
      )
      events.append(event)

    return Session(
        id=response.session_id,
        app_name=app_name,
        user_id=user_id,
        events=events,
        state=response.data or {},
        last_update_time=time.time(),
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
    """Create a new session in Working Memory.

    Uses get_or_create_working_memory to prevent accidental overwrites
    of existing sessions.

    Args:
        app_name: Application name (used as namespace if not configured).
        user_id: User identifier.
        state: Initial session state.
        session_id: Optional session ID (generated if not provided).

    Returns:
        The created Session.
    """
    from agent_memory_client.models import MemoryStrategyConfig

    session_id = (
        session_id.strip()
        if session_id and session_id.strip()
        else str(uuid.uuid4())
    )
    namespace = self._get_namespace(app_name)

    strategy_config = MemoryStrategyConfig(
        strategy=self._config.extraction_strategy,
        config=self._config.extraction_strategy_config,
    )

    # Use get_or_create to prevent accidental overwrites
    created, working_memory = await self._client.get_or_create_working_memory(
        session_id=session_id,
        namespace=namespace,
        user_id=user_id,
        long_term_memory_strategy=strategy_config,
    )

    if not created:
      logger.warning(
          "Session %s already exists in namespace %s, returning existing",
          session_id,
          namespace,
      )
      # Return existing session data
      return self._working_memory_response_to_session(
          working_memory, app_name, user_id
      )

    # Update with initial state and TTL if provided
    if state or self._config.session_ttl_seconds:
      if state:
        working_memory.data = state
      if self._config.session_ttl_seconds:
        working_memory.ttl_seconds = self._config.session_ttl_seconds
      await self._client.put_working_memory(
          session_id=session_id,
          memory=working_memory,
          user_id=user_id,
      )

    logger.info("Created session %s in namespace %s", session_id, namespace)

    return Session(
        id=session_id,
        app_name=app_name,
        user_id=user_id,
        state=state or {},
        events=[],
        last_update_time=time.time(),
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
    """Retrieve a session from Working Memory.

    Uses get_or_create_working_memory and checks if session was newly created
    to determine if it exists. Passes model_name and context_window_max to
    enable automatic context summarization when token limit is exceeded.

    Args:
        app_name: Application name.
        user_id: User identifier.
        session_id: Session ID to retrieve.
        config: Optional configuration for filtering events.

    Returns:
        The Session if found, None otherwise.
    """
    from agent_memory_client.exceptions import MemoryNotFoundError

    try:
      namespace = self._get_namespace(app_name)
      # Use get_or_create to avoid deprecated get_working_memory
      created, response = await self._client.get_or_create_working_memory(
          session_id=session_id,
          namespace=namespace,
          user_id=user_id,
          model_name=self._config.model_name,
          context_window_max=self._config.context_window_max,
      )

      # If session was just created, it means it didn't exist before
      # Delete it and return None to maintain get_session semantics
      if created:
        await self._client.delete_working_memory(
            session_id=session_id,
            namespace=namespace,
            user_id=user_id,
        )
        return None

      session = self._working_memory_response_to_session(
          response, app_name, user_id
      )

      if config:
        if config.num_recent_events:
          session.events = session.events[-config.num_recent_events :]
        if config.after_timestamp:
          session.events = [
              e for e in session.events if e.timestamp > config.after_timestamp
          ]

      return session

    except MemoryNotFoundError:
      return None
    except Exception as e:
      logger.error("Failed to get session %s: %s", session_id, e)
      return None

  @override
  async def list_sessions(
      self, *, app_name: str, user_id: str
  ) -> ListSessionsResponse:
    """List all sessions for a user from Working Memory.

    Args:
        app_name: Application name.
        user_id: User identifier.

    Returns:
        ListSessionsResponse containing sessions (without events).
    """
    try:
      namespace = self._get_namespace(app_name)

      # SDK method: list_sessions returns SessionListResponse
      # with sessions: list[str] (session IDs only)
      response = await self._client.list_sessions(
          namespace=namespace,
          user_id=user_id,
      )

      sessions = []
      for session_id in response.sessions:
        session = Session(
            id=session_id,
            app_name=app_name,
            user_id=user_id,
            state={},
            events=[],
            last_update_time=time.time(),
        )
        sessions.append(session)

      return ListSessionsResponse(sessions=sessions)

    except Exception as e:
      logger.error("Failed to list sessions: %s", e)
      return ListSessionsResponse(sessions=[])

  @override
  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    """Delete a session from Working Memory.

    Args:
        app_name: Application name.
        user_id: User identifier.
        session_id: Session ID to delete.
    """
    try:
      namespace = self._get_namespace(app_name)
      await self._client.delete_working_memory(
          session_id=session_id,
          namespace=namespace,
          user_id=user_id,
      )
      logger.info("Deleted session %s", session_id)
    except Exception as e:
      logger.error("Failed to delete session %s: %s", session_id, e)

  @override
  async def append_event(self, session: Session, event: Event) -> Event:
    """Append an event to the session in Working Memory.

    Uses the incremental append API to add a single message without
    resending the full conversation history.

    Args:
        session: The session to append to.
        event: The event to append.

    Returns:
        The appended event.
    """
    await super().append_event(session=session, event=event)
    session.last_update_time = event.timestamp

    try:
      message = self._event_to_message(event)
      if message:
        namespace = self._get_namespace(session.app_name)
        await self._client.append_messages_to_working_memory(
            session_id=session.id,
            messages=[message],
            namespace=namespace,
            user_id=session.user_id,
        )
        logger.debug("Appended message to session %s", session.id)
    except Exception as e:
      logger.error("Failed to append event to session %s: %s", session.id, e)

    return event

  async def close(self):
    """Close the session service and cleanup resources."""
    if "_client" in self.__dict__:
      await self._client.close()
      del self.__dict__["_client"]
