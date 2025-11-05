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
import re
from typing import Optional
from typing import TYPE_CHECKING

import httpx
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from google.adk.memory import _utils
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry

from .utils import extract_text_from_event

if TYPE_CHECKING:
  from google.adk.sessions.session import Session

logger = logging.getLogger('google_adk.' + __name__)

class OpenMemoryService(BaseMemoryService):
  """Memory service implementation using OpenMemory.
  
  See https://openmemory.cavira.app/ for more information.
  """

  def __init__(
      self,
      base_url: str = "http://localhost:3000",
      api_key: str = "",  # Required parameter (empty string triggers validation)
      config: Optional[OpenMemoryServiceConfig] = None,
  ):
    """Initializes the OpenMemory service.

    Args:
        base_url: Base URL of the OpenMemory instance (default: http://localhost:3000).
        api_key: API key for authentication. **Required** - must be provided.
        config: OpenMemoryServiceConfig instance. If None, uses defaults.
    
    Raises:
        ValueError: If api_key is not provided or is empty.
    """
    if not api_key:
      raise ValueError(
          "api_key is required for OpenMemory. "
          "Provide an API key when initializing OpenMemoryService."
      )
    self._base_url = base_url.rstrip('/')
    self._api_key = api_key
    self._config = config or OpenMemoryServiceConfig()

  def _determine_salience(self, author: Optional[str]) -> float:
    """Determine salience value based on content author."""
    if not author:
      return self._config.default_salience

    author_lower = author.lower()
    if author_lower == "user":
      return self._config.user_content_salience
    elif author_lower == "model":
      return self._config.model_content_salience
    else:
      return self._config.default_salience

  def _prepare_memory_data(
      self, event, content_text: str, session
  ) -> dict:
    """Prepare memory data structure for OpenMemory API."""
    timestamp_str = None
    if event.timestamp:
      timestamp_str = _utils.format_timestamp(event.timestamp)
    
    # Embed author and timestamp in content for search retrieval
    # Format: [Author: user, Time: 2025-11-04T10:32:01] Content text
    enriched_content = content_text
    metadata_parts = []
    if event.author:
      metadata_parts.append(f"Author: {event.author}")
    if timestamp_str:
      metadata_parts.append(f"Time: {timestamp_str}")
    
    if metadata_parts:
      metadata_prefix = "[" + ", ".join(metadata_parts) + "] "
      enriched_content = metadata_prefix + content_text
    
    metadata = {
        "app_name": session.app_name,
        "user_id": session.user_id,
        "session_id": session.id,
        "event_id": event.id,
        "invocation_id": event.invocation_id,
        "author": event.author,
        "timestamp": event.timestamp,
        "source": "adk_session"
    }
    
    memory_data = {
        "content": enriched_content,
        "metadata": metadata,
        "salience": self._determine_salience(event.author)
    }

    if self._config.enable_metadata_tags:
      tags = [
          f"session:{session.id}",
          f"app:{session.app_name}",
      ]
      if event.author:
        tags.append(f"author:{event.author}")
      memory_data["tags"] = tags

    return memory_data

  @override
  async def add_session_to_memory(self, session: Session):
    """Add a session's events to OpenMemory."""
    memories_added = 0

    async with httpx.AsyncClient(timeout=self._config.timeout) as http_client:
      headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {self._api_key}"
      }

      for event in session.events:
        content_text = extract_text_from_event(event)
        if not content_text:
          continue

        memory_data = self._prepare_memory_data(event, content_text, session)

        try:
          # user_id is passed as top-level field for server-side filtering
          payload = {
              "content": memory_data["content"],
              "tags": memory_data.get("tags", []),
              "metadata": memory_data.get("metadata", {}),
              "salience": memory_data.get("salience", 0.5),
              "user_id": session.user_id
          }
          
          response = await http_client.post(
              f"{self._base_url}/memory/add",
              json=payload,
              headers=headers
          )
          response.raise_for_status()
          
          memories_added += 1
          logger.debug("Added memory for event %s", event.id)
        except httpx.HTTPStatusError as e:
          logger.error(
              "Failed to add memory for event %s due to HTTP error: %s - %s",
              event.id,
              e.response.status_code,
              e.response.text,
          )
        except httpx.RequestError as e:
          logger.error(
              "Failed to add memory for event %s due to request error: %s", event.id, e
          )
        except Exception as e:
          logger.error("Failed to add memory for event %s due to unexpected error: %s", event.id, e)

    logger.info(
        "Added %d memories from session %s", memories_added, session.id
    )

  def _build_search_payload(
      self, app_name: str, user_id: str, query: str
  ) -> dict:
    """Build search payload for OpenMemory query API."""
    payload = {
        "query": query,
        "k": self._config.search_top_k,
        "filter": {}
    }

    payload["filter"]["user_id"] = user_id

    if self._config.enable_metadata_tags:
      payload["filter"]["tags"] = [f"app:{app_name}"]

    return payload

  def _convert_to_memory_entry(self, result: dict) -> Optional[MemoryEntry]:
    """Convert OpenMemory result to MemoryEntry.
    
    Extracts author and timestamp from enriched content format:
    [Author: user, Time: 2025-11-04T10:32:01] Content text
    """
    try:
      raw_content = result["content"]
      author = None
      timestamp = None
      clean_content = raw_content
      
      # Parse enriched content format to extract metadata
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

      return MemoryEntry(
          content=content,
          author=author,
          timestamp=timestamp
      )
    except (KeyError, ValueError) as e:
      logger.debug("Failed to convert result to MemoryEntry: %s", e)
      return None

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Search for memories using OpenMemory's query API."""
    try:
      search_payload = self._build_search_payload(app_name, user_id, query)
      memories = []
      
      async with httpx.AsyncClient(timeout=self._config.timeout) as http_client:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}"
        }
        
        logger.debug("Query payload: %s", search_payload)
        
        response = await http_client.post(
            f"{self._base_url}/memory/query",
            json=search_payload,
            headers=headers
        )
        response.raise_for_status()
        result = response.json()
        
        logger.debug("Query returned %d matches", len(result.get("matches", [])))
        
        for match in result.get("matches", []):
          memory_entry = self._convert_to_memory_entry(match)
          if memory_entry:
            memories.append(memory_entry)

      logger.info("Found %d memories for query: '%s'", len(memories), query)
      return SearchMemoryResponse(memories=memories)

    except httpx.HTTPStatusError as e:
      logger.error(
          "Failed to search memories due to HTTP error: %s - %s",
          e.response.status_code,
          e.response.text,
      )
      return SearchMemoryResponse(memories=[])
    except httpx.RequestError as e:
      logger.error("Failed to search memories due to request error: %s", e)
      return SearchMemoryResponse(memories=[])
    except Exception as e:
      logger.error("Failed to search memories due to unexpected error: %s", e)
      return SearchMemoryResponse(memories=[])

  async def close(self):
    """Close the memory service and cleanup resources."""
    pass


class OpenMemoryServiceConfig(BaseModel):
  """Configuration for OpenMemory service behavior.

  Attributes:
      search_top_k: Maximum number of memories to retrieve per search.
      timeout: Request timeout in seconds.
      user_content_salience: Salience for user-authored content (0.0-1.0).
      model_content_salience: Salience for model-generated content (0.0-1.0).
      default_salience: Default salience value for memories (0.0-1.0).
      enable_metadata_tags: Include session/app tags in memories.
  """

  search_top_k: int = Field(default=10, ge=1, le=100)
  timeout: float = Field(default=30.0, gt=0.0)
  user_content_salience: float = Field(default=0.8, ge=0.0, le=1.0)
  model_content_salience: float = Field(default=0.7, ge=0.0, le=1.0)
  default_salience: float = Field(default=0.6, ge=0.0, le=1.0)
  enable_metadata_tags: bool = Field(default=True)
