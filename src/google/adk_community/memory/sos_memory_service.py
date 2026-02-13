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

"""SOS (Sovereign Operating System) Memory Service for ADK.

This module provides integration with SOS Mirror API, a multi-tier semantic
memory system with FRC (Frequency-Recency-Context) physics for intelligent
retrieval and consolidation.

Features:
- Semantic search with vector embeddings
- FRC-weighted retrieval (frequency, recency, context relevance)
- Lineage tracking for memory provenance
- Multi-agent memory isolation
- Automatic memory consolidation

See https://github.com/servathadi/sos for more information.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
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


class SOSMemoryService(BaseMemoryService):
    """Memory service implementation using SOS Mirror API.

    SOS Mirror provides semantic memory with FRC physics - memories are
    retrieved based on frequency (how often accessed), recency (how recent),
    and context (semantic relevance to query).

    Example usage:
        ```python
        from google.adk_community.memory import SOSMemoryService

        memory_service = SOSMemoryService(
            base_url="https://mirror.mumega.com",
            api_key="your-api-key",
            agent_id="my-agent"
        )

        # Use with ADK agent
        agent = Agent(
            model='gemini-2.0-flash',
            name='my_agent',
            memory_service=memory_service,
            tools=[load_memory, preload_memory],
        )
        ```
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8844",
        api_key: str = "",
        agent_id: str = "adk_agent",
        config: Optional[SOSMemoryServiceConfig] = None,
    ):
        """Initialize the SOS Memory service.

        Args:
            base_url: Base URL of the SOS Mirror API.
            api_key: API key for authentication.
            agent_id: Unique identifier for this agent's memory namespace.
            config: SOSMemoryServiceConfig instance for advanced options.

        Raises:
            ValueError: If api_key is not provided.
        """
        if not api_key:
            raise ValueError(
                "api_key is required for SOS Mirror. "
                "Provide an API key when initializing SOSMemoryService."
            )
        self._base_url = base_url.rstrip('/')
        self._api_key = api_key
        self._agent_id = agent_id
        self._config = config or SOSMemoryServiceConfig()
        self._lineage_chain: List[str] = []

    def _compute_lineage_hash(self, content: str, context: str = "") -> str:
        """Compute a lineage hash for memory provenance tracking."""
        prev_hash = self._lineage_chain[-1] if self._lineage_chain else "genesis"
        data = f"{prev_hash}:{self._agent_id}:{content}:{context}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _determine_salience(self, author: Optional[str]) -> float:
        """Determine salience based on content author."""
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
    ) -> Dict[str, Any]:
        """Prepare memory data for SOS Mirror API."""
        timestamp_str = None
        if event.timestamp:
            timestamp_str = _utils.format_timestamp(event.timestamp)

        # Compute lineage hash for provenance
        lineage_hash = self._compute_lineage_hash(content_text, session.id)
        self._lineage_chain.append(lineage_hash)

        # Build enriched content with metadata prefix
        enriched_content = content_text
        metadata_parts = []
        if event.author:
            metadata_parts.append(f"Author: {event.author}")
        if timestamp_str:
            metadata_parts.append(f"Time: {timestamp_str}")

        if metadata_parts:
            metadata_prefix = "[" + ", ".join(metadata_parts) + "] "
            enriched_content = metadata_prefix + content_text

        return {
            "text": enriched_content,
            "agent": self._agent_id,
            "context_id": session.id,
            "metadata": {
                "app_name": session.app_name,
                "user_id": session.user_id,
                "session_id": session.id,
                "event_id": event.id,
                "invocation_id": event.invocation_id,
                "author": event.author,
                "timestamp": event.timestamp,
                "lineage_hash": lineage_hash,
                "salience": self._determine_salience(event.author),
                "source": "adk_session"
            }
        }

    @override
    async def add_session_to_memory(self, session: Session):
        """Add a session's events to SOS Mirror memory."""
        memories_added = 0

        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
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
                    response = await client.post(
                        f"{self._base_url}/store",
                        json=memory_data,
                        headers=headers
                    )
                    response.raise_for_status()

                    memories_added += 1
                    logger.debug("Added memory for event %s (lineage: %s)",
                                event.id, memory_data["metadata"]["lineage_hash"])
                except httpx.HTTPStatusError as e:
                    logger.error(
                        "Failed to add memory for event %s: HTTP %s - %s",
                        event.id, e.response.status_code, e.response.text
                    )
                except httpx.RequestError as e:
                    logger.error(
                        "Failed to add memory for event %s: %s", event.id, e
                    )
                except Exception as e:
                    logger.error(
                        "Unexpected error adding memory for event %s: %s", event.id, e
                    )

        logger.info(
            "Added %d memories from session %s to agent %s",
            memories_added, session.id, self._agent_id
        )

    def _build_search_payload(
        self, app_name: str, user_id: str, query: str
    ) -> Dict[str, Any]:
        """Build search payload for SOS Mirror query API."""
        return {
            "query": query,
            "agent": self._agent_id,
            "limit": self._config.search_top_k,
            "filters": {
                "user_id": user_id,
                "app_name": app_name,
            } if self._config.enable_user_filtering else {}
        }

    def _convert_to_memory_entry(self, result: Dict[str, Any]) -> Optional[MemoryEntry]:
        """Convert SOS Mirror result to ADK MemoryEntry."""
        try:
            raw_content = result.get("text", result.get("content", ""))
            author = None
            timestamp = None
            clean_content = raw_content

            # Parse enriched content format: [Author: user, Time: ...] Content
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

            # Also check metadata dict
            metadata = result.get("metadata", {})
            if not author and metadata.get("author"):
                author = metadata["author"]
            if not timestamp and metadata.get("timestamp"):
                timestamp = metadata["timestamp"]

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
        """Search memories using SOS Mirror's semantic search with FRC physics."""
        try:
            search_payload = self._build_search_payload(app_name, user_id, query)
            memories = []

            async with httpx.AsyncClient(timeout=self._config.timeout) as client:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}"
                }

                logger.debug("SOS search payload: %s", search_payload)

                response = await client.post(
                    f"{self._base_url}/search",
                    json=search_payload,
                    headers=headers
                )
                response.raise_for_status()
                result = response.json()

                # Handle both list and dict response formats
                results = result if isinstance(result, list) else result.get("results", [])

                logger.debug("SOS search returned %d results", len(results))

                for match in results:
                    memory_entry = self._convert_to_memory_entry(match)
                    if memory_entry:
                        memories.append(memory_entry)

            logger.info(
                "Found %d memories for query '%s' (agent: %s)",
                len(memories), query[:50], self._agent_id
            )
            return SearchMemoryResponse(memories=memories)

        except httpx.HTTPStatusError as e:
            logger.error(
                "SOS search failed: HTTP %s - %s",
                e.response.status_code, e.response.text
            )
            return SearchMemoryResponse(memories=[])
        except httpx.RequestError as e:
            logger.error("SOS search request failed: %s", e)
            return SearchMemoryResponse(memories=[])
        except Exception as e:
            logger.error("Unexpected error in SOS search: %s", e)
            return SearchMemoryResponse(memories=[])

    async def get_lineage(self) -> Dict[str, Any]:
        """Get the current lineage chain for this agent session.

        Returns:
            Dict with lineage information including chain and latest hash.
        """
        return {
            "agent_id": self._agent_id,
            "chain_length": len(self._lineage_chain),
            "latest_hash": self._lineage_chain[-1] if self._lineage_chain else None,
            "chain": self._lineage_chain[-10:],  # Last 10 hashes
        }

    async def close(self):
        """Close the memory service and cleanup resources."""
        self._lineage_chain.clear()


class SOSMemoryServiceConfig(BaseModel):
    """Configuration for SOS Memory service behavior.

    Attributes:
        search_top_k: Maximum memories to retrieve per search.
        timeout: Request timeout in seconds.
        user_content_salience: Salience for user content (0.0-1.0).
        model_content_salience: Salience for model content (0.0-1.0).
        default_salience: Default salience value (0.0-1.0).
        enable_user_filtering: Filter results by user_id.
        enable_lineage_tracking: Track memory provenance with hashes.
    """

    search_top_k: int = Field(default=10, ge=1, le=100)
    timeout: float = Field(default=30.0, gt=0.0)
    user_content_salience: float = Field(default=0.8, ge=0.0, le=1.0)
    model_content_salience: float = Field(default=0.7, ge=0.0, le=1.0)
    default_salience: float = Field(default=0.6, ge=0.0, le=1.0)
    enable_user_filtering: bool = Field(default=True)
    enable_lineage_tracking: bool = Field(default=True)
