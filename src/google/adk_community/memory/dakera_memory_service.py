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
import httpx
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from .utils import extract_text_from_event

if TYPE_CHECKING:
  from google.adk.sessions.session import Session

logger = logging.getLogger('google_adk.' + __name__)


class DakeraMemoryService(BaseMemoryService):
  """Memory service implementation using Dakera.

  Dakera (https://dakera.ai) is a self-hosted memory server that persists agent
  memories across sessions and ranks recall by an access-weighted importance
  decay model, so frequently-used memories survive while stale ones fade.

  Session events are stored to ``POST /v1/memory/store`` and retrieved with
  semantic recall via ``POST /v1/memory/recall``. Memories are namespaced by a
  Dakera ``agent_id`` derived from the ADK ``app_name`` and ``user_id`` so each
  app/user keeps an isolated memory space; ``session_id`` is preserved for
  provenance.

  Self-host the server (with its object store) using the public compose file::

      git clone https://github.com/dakera-ai/dakera-deploy
      cd dakera-deploy && docker compose up -d   # server on :3000 + MinIO

  Then initialise the service with your server URL and API key::

      from google.adk_community.memory import DakeraMemoryService

      memory_service = DakeraMemoryService(
          base_url="http://localhost:3000",  # or set DAKERA_API_URL
          api_key="dk-...",                  # or set DAKERA_API_KEY
      )
  """

  def __init__(
      self,
      base_url: Optional[str] = None,
      api_key: Optional[str] = None,
      config: Optional[DakeraMemoryServiceConfig] = None,
  ):
    """Initializes the Dakera memory service.

    Args:
        base_url: Base URL of the Dakera server. Defaults to the
            ``DAKERA_API_URL`` environment variable, falling back to
            ``http://localhost:3000``.
        api_key: API key for authentication (a ``dk-...`` token). Defaults to
            the ``DAKERA_API_KEY`` environment variable. **Required** — must be
            provided directly or via the environment.
        config: DakeraMemoryServiceConfig instance. If None, uses defaults.

    Raises:
        ValueError: If no API key is provided or resolvable from the
            environment.
    """
    resolved_key = api_key or os.environ.get('DAKERA_API_KEY', '')
    if not resolved_key:
      raise ValueError(
          'api_key is required for Dakera. Provide an API key when '
          'initializing DakeraMemoryService or set DAKERA_API_KEY.'
      )
    resolved_url = base_url or os.environ.get(
        'DAKERA_API_URL', 'http://localhost:3000'
    )
    self._base_url = resolved_url.rstrip('/')
    self._api_key = resolved_key
    self._config = config or DakeraMemoryServiceConfig()

  def _namespace(self, app_name: str, user_id: str) -> str:
    """Derive the Dakera agent_id (namespace) for an app/user pair."""
    return f'{app_name}:{user_id}'

  def _determine_importance(self, author: Optional[str]) -> float:
    """Determine importance based on the content author.

    User-authored content is weighted highest, then model output, so Dakera's
    decay engine retains the most task-relevant memories longest.
    """
    if not author:
      return self._config.default_importance

    author_lower = author.lower()
    if author_lower == 'user':
      return self._config.user_content_importance
    elif author_lower == 'model':
      return self._config.model_content_importance
    else:
      return self._config.default_importance

  def _prepare_memory_data(self, event, content_text: str, session) -> dict:
    """Prepare the ``/v1/memory/store`` request payload for one event."""
    timestamp_str = None
    if event.timestamp:
      timestamp_str = _utils.format_timestamp(event.timestamp)

    # Embed author and timestamp in the content so they can be recovered on
    # recall (Dakera returns stored content verbatim).
    # Format: [Author: user, Time: 2025-11-04T10:32:01] Content text
    enriched_content = content_text
    metadata_parts = []
    if event.author:
      metadata_parts.append(f'Author: {event.author}')
    if timestamp_str:
      metadata_parts.append(f'Time: {timestamp_str}')

    if metadata_parts:
      metadata_prefix = '[' + ', '.join(metadata_parts) + '] '
      enriched_content = metadata_prefix + content_text

    memory_data = {
        'content': enriched_content,
        'agent_id': self._namespace(session.app_name, session.user_id),
        'memory_type': self._config.memory_type,
        'session_id': session.id,
        'importance': self._determine_importance(event.author),
        'metadata': {
            'app_name': session.app_name,
            'user_id': session.user_id,
            'session_id': session.id,
            'event_id': event.id,
            'invocation_id': event.invocation_id,
            'author': event.author,
            'timestamp': event.timestamp,
            'source': 'adk_session',
        },
    }

    if self._config.enable_metadata_tags:
      tags = [
          f'session:{session.id}',
          f'app:{session.app_name}',
      ]
      if event.author:
        tags.append(f'author:{event.author}')
      memory_data['tags'] = tags

    return memory_data

  @override
  async def add_session_to_memory(self, session: Session):
    """Add a session's events to Dakera memory."""
    memories_added = 0

    async with httpx.AsyncClient(timeout=self._config.timeout) as http_client:
      headers = {
          'Content-Type': 'application/json',
          'Authorization': f'Bearer {self._api_key}',
      }

      for event in session.events:
        content_text = extract_text_from_event(event)
        if not content_text:
          continue

        payload = self._prepare_memory_data(event, content_text, session)

        try:
          response = await http_client.post(
              f'{self._base_url}/v1/memory/store',
              json=payload,
              headers=headers,
          )
          response.raise_for_status()
          memories_added += 1
          logger.debug('Added memory for event %s', event.id)
        except httpx.HTTPStatusError as e:
          logger.error(
              'Failed to add memory for event %s due to HTTP error: %s - %s',
              event.id,
              e.response.status_code,
              e.response.text,
          )
        except httpx.RequestError as e:
          logger.error(
              'Failed to add memory for event %s due to request error: %s',
              event.id,
              e,
          )
        except Exception as e:
          logger.error(
              'Failed to add memory for event %s due to unexpected error: %s',
              event.id,
              e,
          )

    logger.info('Added %d memories from session %s', memories_added, session.id)

  def _build_recall_payload(
      self, app_name: str, user_id: str, query: str
  ) -> dict:
    """Build the ``/v1/memory/recall`` request payload."""
    payload = {
        'query': query,
        'agent_id': self._namespace(app_name, user_id),
        'top_k': self._config.search_top_k,
    }
    if self._config.min_importance is not None:
      payload['min_importance'] = self._config.min_importance
    return payload

  def _convert_to_memory_entry(self, memory: dict) -> Optional[MemoryEntry]:
    """Convert a Dakera memory record to a ``MemoryEntry``.

    Extracts author and timestamp from the enriched content prefix
    ``[Author: user, Time: 2025-11-04T10:32:01] Content text``.
    """
    try:
      raw_content = memory['content']
      author = None
      timestamp = None
      clean_content = raw_content

      match = re.match(r'^\[([^\]]+)\]\s+(.*)', raw_content, re.DOTALL)
      if match:
        metadata_str = match.group(1)
        clean_content = match.group(2)

        author_match = re.search(r'Author:\s*([^,\]]+)', metadata_str)
        if author_match:
          author = author_match.group(1).strip()

        time_match = re.search(r'Time:\s*([^,\]]+)', metadata_str)
        if time_match:
          timestamp = time_match.group(1).strip()

      content = types.Content(parts=[types.Part(text=clean_content)])
      return MemoryEntry(content=content, author=author, timestamp=timestamp)
    except (KeyError, ValueError) as e:
      logger.debug('Failed to convert result to MemoryEntry: %s', e)
      return None

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Search Dakera for memories relevant to *query* within the namespace."""
    try:
      recall_payload = self._build_recall_payload(app_name, user_id, query)
      memories = []

      async with httpx.AsyncClient(timeout=self._config.timeout) as http_client:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}',
        }

        logger.debug('Recall payload: %s', recall_payload)

        response = await http_client.post(
            f'{self._base_url}/v1/memory/recall',
            json=recall_payload,
            headers=headers,
        )
        response.raise_for_status()
        result = response.json()

      results = result.get('memories', [])
      logger.debug('Recall returned %d matches', len(results))

      for item in results:
        # Each result wraps the stored record under "memory".
        memory = item.get('memory') if isinstance(item, dict) else None
        if not memory:
          continue
        memory_entry = self._convert_to_memory_entry(memory)
        if memory_entry:
          memories.append(memory_entry)

      logger.info("Found %d memories for query: '%s'", len(memories), query)
      return SearchMemoryResponse(memories=memories)

    except httpx.HTTPStatusError as e:
      logger.error(
          'Failed to search memories due to HTTP error: %s - %s',
          e.response.status_code,
          e.response.text,
      )
      return SearchMemoryResponse(memories=[])
    except httpx.RequestError as e:
      logger.error('Failed to search memories due to request error: %s', e)
      return SearchMemoryResponse(memories=[])
    except Exception as e:
      logger.error('Failed to search memories due to unexpected error: %s', e)
      return SearchMemoryResponse(memories=[])

  async def close(self):
    """Close the memory service and cleanup resources."""
    pass


class DakeraMemoryServiceConfig(BaseModel):
  """Configuration for Dakera memory service behavior.

  Attributes:
      search_top_k: Maximum number of memories to retrieve per recall.
      timeout: Request timeout in seconds.
      user_content_importance: Importance for user-authored content (0.0-1.0).
      model_content_importance: Importance for model-generated content
          (0.0-1.0).
      default_importance: Default importance for other authors (0.0-1.0).
      min_importance: Optional lower bound on importance for recall results.
      memory_type: Dakera memory type for stored session events.
      enable_metadata_tags: Include session/app/author tags on stored memories.
  """

  search_top_k: int = Field(default=10, ge=1, le=100)
  timeout: float = Field(default=30.0, gt=0.0)
  user_content_importance: float = Field(default=0.8, ge=0.0, le=1.0)
  model_content_importance: float = Field(default=0.7, ge=0.0, le=1.0)
  default_importance: float = Field(default=0.6, ge=0.0, le=1.0)
  min_importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
  memory_type: str = Field(default='episodic')
  enable_metadata_tags: bool = Field(default=True)
