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

import json
import logging
from typing import TYPE_CHECKING, Any, Optional
from typing_extensions import override

import google.auth
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from google.genai import types

from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_request import LlmRequest
from google.adk.utils.context_utils import Aclosing
from google.adk.memory import _utils
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry

if TYPE_CHECKING:
    from google.adk.events.event import Event
    from google.adk.sessions.session import Session

logger = logging.getLogger(__name__)


class FirestoreLLMMemoryService(BaseMemoryService):
    """A Firestore-based memory service that uses an LLM to manage facts.

    Instead of storing raw events, it extracts and reconciles concise facts
    about the user, enabling smarter semantic search and memory management.
    """

    def __init__(
        self,
        collection_name: str = "agent_facts",
        model: str = "gemini-2.0-flash",
        database: Optional[
            str
        ] = "(default)",  # use generous free tier default by default
    ):
        """Initializes the FirestoreLLMMemoryService.

        Args:
            collection_name: The root collection name in Firestore.
            model: The LLM model to use for memory management.
            database: The Firestore database to use. (Uses free tier by default)
        """
        credentials, project_id = google.auth.default()
        self.db = firestore.AsyncClient(
            credentials=credentials, project=project_id, database=database
        )
        self.collection_name = collection_name

        # The internal agent dedicated to managing the memory state.
        self._memory_agent = Agent(
            model=model,
            name="memory_manager",
            description="Manages user facts and retrieves relevant memories.",
            instruction=(
                "You are a memory management assistant. Your job is to maintain a high-quality "
                "list of facts about conversations. "
                "You will be asked to reconcile new conversations with existing facts "
                "and to retrieve relevant facts based on queries."
            ),
        )

    def _format_session(self, session: Session) -> str:
        lines = []
        for event in session.events:
            if not event.content or not event.content.parts:
                continue
            text = " ".join([p.text for p in event.content.parts if p.text])
            lines.append(f"{event.author}: {text}")
        return "\n".join(lines)

    async def _call_agent(self, prompt: str) -> str:
        """Utility to call the underlying LLM of the agent."""
        llm = self._memory_agent.canonical_model
        request = LlmRequest(model=llm.model)

        # Add system instruction from agent
        if isinstance(self._memory_agent.instruction, str):
            request.append_instructions([self._memory_agent.instruction])

        # Add user prompt
        request.contents.append(
            types.Content(role="user", parts=[types.Part(text=prompt)])
        )

        async with Aclosing(llm.generate_content_async(request)) as agen:
            final_response = None
            async for response in agen:
                if not response.partial:
                    final_response = response

            if (
                final_response
                and final_response.content
                and final_response.content.parts
                and final_response.content.parts[0].text
            ):
                return final_response.content.parts[0].text
        return ""

    @override
    async def add_session_to_memory(self, session: Session):
        """Extracts facts from the session and updates Firestore."""
        user_key = f"{session.app_name}:{session.user_id}"
        facts_ref = (
            self.db.collection(self.collection_name)
            .document(user_key)
            .collection("facts")
        )

        # 1. Fetch existing facts
        existing_facts = []
        async for doc in facts_ref.stream():
            data = doc.to_dict()
            existing_facts.append({"id": doc.id, "text": data["text"]})

        # 2. Reconcile with the Agent
        session_text = self._format_session(session)
        prompt = (
            f"Existing Facts:\n{json.dumps(existing_facts, indent=2)}\n\n"
            f"New Session Transcript:\n{session_text}\n\n"
            "Task: Reconcile the new session with the existing facts. "
            "Identify facts to add, update, or delete. "
            "Respond ONLY with a JSON object in this format:\n"
            '{"add": ["new fact text"], "update": [{"id": "doc_id", "text": "new text"}], "delete": ["doc_id"]}'
        )

        content = await self._call_agent(prompt)
        try:
            # Clean up potential markdown formatting in response
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()

            operations = json.loads(content)
        except Exception as e:
            logger.error(
                f"Failed to parse Agent response for fact reconciliation: {e}. Response: {content}"
            )
            return

        # 3. Apply operations to Firestore
        batch = self.db.batch()

        for fact_text in operations.get("add", []):
            new_doc_ref = facts_ref.document()
            batch.set(
                new_doc_ref,
                {
                    "text": fact_text,
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "source_session_id": session.id,
                },
            )

        for update in operations.get("update", []):
            doc_ref = facts_ref.document(update["id"])
            batch.update(
                doc_ref,
                {
                    "text": update["text"],
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "source_session_id": session.id,
                },
            )

        for doc_id in operations.get("delete", []):
            batch.delete(facts_ref.document(doc_id))

        await batch.commit()

    @override
    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        """Uses the Agent to find relevant facts based on the query."""
        user_key = f"{app_name}:{user_id}"
        facts_ref = (
            self.db.collection(self.collection_name)
            .document(user_key)
            .collection("facts")
        )

        # 1. Fetch all facts
        all_facts = []
        async for doc in facts_ref.stream():
            data = doc.to_dict()
            all_facts.append(
                {
                    "id": doc.id,
                    "text": data["text"],
                    "timestamp": data.get("timestamp").timestamp(),
                    "source_session_id": data.get("source_session_id"),
                }
            )

        if not all_facts:
            return SearchMemoryResponse()

        # 2. Filter with the Agent
        prompt = (
            f"User Query: {query}\n\n"
            f"Available Facts:\n{json.dumps(all_facts, indent=2, default=str)}\n\n"
            "Task: Identify which facts are relevant to the user query. "
            "Respond ONLY with a JSON list of IDs of the relevant facts. "
            'Example: ["id1", "id2"]'
        )

        content = await self._call_agent(prompt)
        try:
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            relevant_ids = json.loads(content)
        except Exception as e:
            logger.error(
                f"Failed to parse Agent response for memory search: {e}. Response: {content}"
            )
            return SearchMemoryResponse()

        # 3. Construct response
        search_response = SearchMemoryResponse()
        relevant_facts = [f for f in all_facts if f["id"] in relevant_ids]
        for fact in relevant_facts:
            search_response.memories.append(
                MemoryEntry(
                    content=types.Content(parts=[types.Part(text=fact["text"])]),
                    author="memory_manager",
                    timestamp=_utils.format_timestamp(fact["timestamp"]),
                )
            )

        return search_response
