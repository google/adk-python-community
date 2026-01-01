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
from typing import TYPE_CHECKING, Optional, Any
from typing_extensions import override

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from google.genai import types
from google.adk.memory import _utils
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
import google.auth

if TYPE_CHECKING:
    from google.adk.sessions.session import Session
logger = logging.getLogger(__name__)


def _extract_words_lower(text: str) -> set[str]:
    """Extracts words from a string and converts them to lowercase."""
    return {word.lower() for word in re.findall(r"\w+", text)}


class FirestoreWordMemoryService(BaseMemoryService):
    """A Firestore-based memory service for Google ADK.

    Uses Google Cloud Firestore to store and retrieve memories.
    Matches the keyword-based search behavior of InMemoryMemoryService.
    """

    def __init__(
        self,
        collection_name: str = "agent_memories",
        database: Optional[
            str
        ] = "(default)",  # use generous free tier default by default
    ):
        """Initializes the FirestoreWordMemoryService.

        Args:
            collection_name: The root collection name in Firestore.
            database: The Firestore database to use. (Uses free tier by default)
        """
        credentials, project_id = google.auth.default()
        self.db = firestore.AsyncClient(
            credentials=credentials, project=project_id, database=database
        )
        self.collection_name = collection_name

    def _serialize_content(self, content: types.Content) -> dict[str, Any]:
        if not content or not content.parts:
            return {"parts": []}
        return {"parts": [{"text": part.text} for part in content.parts if part.text]}

    def _deserialize_content(self, data: dict[str, Any]) -> types.Content:
        parts = [types.Part(text=p["text"]) for p in data.get("parts", [])]
        return types.Content(parts=parts)

    @override
    async def add_session_to_memory(self, session: Session):
        """Adds all events from a session to Firestore memory."""
        # We store events in a subcollection under a user document
        # Structure: {collection_name}/{app_name}:{user_id}/events/{event_id}

        user_key = f"{session.app_name}:{session.user_id}"
        user_ref = self.db.collection(self.collection_name).document(user_key)
        events_ref = user_ref.collection("events")

        # batch up additions for efficiency
        batch = self.db.batch()
        has_events_to_add = False

        for event in session.events:
            if event.content and event.content.parts:
                has_events_to_add = True
                event_data = {
                    "session_id": session.id,
                    "content": self._serialize_content(event.content),
                    "author": event.author,
                    "timestamp": event.timestamp,
                    # We pre-calculate words for easier filtering if needed,
                    # though we still do filtering in search_memory to match InMemory behavior
                    "words": list(
                        _extract_words_lower(
                            " ".join(
                                [part.text for part in event.content.parts if part.text]
                            )
                        )
                    ),
                }
                # Add to batch with firestore-generated ID
                batch.set(events_ref.document(), event_data)
        if has_events_to_add:
            await batch.commit()

    @override
    async def search_memory(
        self, *, app_name: str, user_id: str, query: str, limit: int = 100
    ) -> SearchMemoryResponse:
        """Searches memory in Firestore based on keyword matching."""
        user_key = f"{app_name}:{user_id}"
        words_in_query = _extract_words_lower(query)
        response = SearchMemoryResponse()

        if not words_in_query:
            return response

        # Structure for events to make searching easier while keeping user context.
        # Structure: {collection_name}/{user_key}/events/{event_id}

        events_query = (
            self.db.collection(self.collection_name)
            .document(user_key)
            .collection("events")
        )

        # We can attempt to filter by query words using array_contains_any
        # This matches the 'any(query_word in words_in_event for query_word in words_in_query)' logic
        query_words_list = list(words_in_query)

        # Firestore array_contains_any handles up to 30 elements
        # If query is longer, we might need multiple queries or client-side filtering
        # warn that this is a simple implementation and may not scale for very large queries
        if len(query_words_list) > 30:
            logger.warning(
                "Query contains more than 30 unique words; truncating for Firestore query."
            )
            query_words_list = query_words_list[:30]

        docs = (
            events_query.where(
                filter=FieldFilter("words", "array_contains_any", query_words_list)
            )
            .limit(limit)
            .stream()
        )

        async for doc in docs:
            data = doc.to_dict()
            if (
                data
                and data.get("content")
                and data.get("author")
                and data.get("timestamp")
            ):

                response.memories.append(
                    MemoryEntry(
                        content=self._deserialize_content(data["content"]),
                        author=data["author"],
                        timestamp=_utils.format_timestamp(data["timestamp"]),
                    )
                )

        return response
