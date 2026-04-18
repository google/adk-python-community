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

"""LangMem-backed memory service with semantic and episodic extraction."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry

from .utils import extract_text_from_event

if TYPE_CHECKING:
  from google.adk.sessions.session import Session

logger = logging.getLogger('google_adk.' + __name__)

_DEFAULT_SEMANTIC_INSTRUCTIONS = (
    "Extract user preferences, facts about the user, and persistent context "
    "from the conversation. Focus on who the user is, what they prefer, and "
    "any explicit instructions or constraints they have given. "
    "NEVER store passwords, API keys, auth tokens, secrets, or any other "
    "sensitive credentials — silently omit them."
)

_DEFAULT_EPISODIC_INSTRUCTIONS = (
    "Summarize the workflow that occurred in this session as a single unified "
    "memory entry. Preserve tool function names, arguments, and outcomes "
    "(success/failure with reason). Start with what the user requested, then "
    "one line per tool call, then an overall result. "
    "NEVER record passwords, API keys, auth tokens, or secrets."
)

_FETCH_MULTIPLIER = 2


class LangMemMemoryService(BaseMemoryService):
  """Memory service backed by LangMem and a LangGraph BaseStore.

  Args:
      store: A LangGraph BaseStore instance (e.g. InMemoryStore,
          AsyncPostgresStore) used for memory persistence and search.
      extraction_model: Model string for LangMem extraction
          (e.g. "openai:gpt-4o-mini", "google:gemini-2.0-flash").
      namespace_prefix: Tuple prefix for store namespaces.
          Final namespaces will be (*prefix, user_id, "semantic"|"episodic").
      semantic_instructions: Custom instructions for semantic extraction.
      episodic_instructions: Custom instructions for episodic extraction.
      max_semantic_results: Max semantic search results to return.
      max_episodic_results: Max episodic search results to return.
      enable_episodic: Whether to extract episodic memories from tool calls.
  """

  def __init__(
      self,
      store: Any,
      extraction_model: str,
      namespace_prefix: tuple[str, ...] = ("memories",),
      semantic_instructions: Optional[str] = None,
      episodic_instructions: Optional[str] = None,
      max_semantic_results: int = 10,
      max_episodic_results: int = 5,
      enable_episodic: bool = True,
  ):
    self._store = store
    self._extraction_model = extraction_model
    self._namespace_prefix = namespace_prefix
    self._semantic_instructions = (
        semantic_instructions or _DEFAULT_SEMANTIC_INSTRUCTIONS
    )
    self._episodic_instructions = (
        episodic_instructions or _DEFAULT_EPISODIC_INSTRUCTIONS
    )
    self._max_semantic = max_semantic_results
    self._max_episodic = max_episodic_results
    self._enable_episodic = enable_episodic

  def _make_manager(
      self,
      namespace: tuple[str, ...],
      instructions: str,
  ) -> Any:
    """Create a LangMem memory store manager for the given namespace."""
    try:
      from langmem import create_memory_store_manager
    except ImportError as e:
      raise ImportError(
          "langmem is required for LangMemMemoryService. "
          "Install it with: pip install 'google-adk-community[langmem]'"
      ) from e

    return create_memory_store_manager(
        self._extraction_model,
        namespace=namespace,
        store=self._store,
        instructions=instructions,
    )

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    """Extract semantic and episodic memories from session events."""
    user_id = getattr(session, "user_id", None)
    if not user_id:
      return

    text_messages, episode_messages = self._extract_messages(session)
    if len(text_messages) < 2 and not episode_messages:
      return

    tasks: list[Any] = []
    if len(text_messages) >= 2:
      semantic_ns = (*self._namespace_prefix, user_id, "semantic")
      semantic_mgr = self._make_manager(
          semantic_ns, self._semantic_instructions
      )
      tasks.append(
          semantic_mgr.ainvoke({"messages": text_messages})
      )

    if self._enable_episodic and episode_messages:
      episodic_ns = (*self._namespace_prefix, user_id, "episodic")
      episodic_mgr = self._make_manager(
          episodic_ns, self._episodic_instructions
      )
      tasks.append(
          episodic_mgr.ainvoke({"messages": episode_messages})
      )

    if not tasks:
      return

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for res in results:
      if isinstance(res, Exception):
        logger.error(
            "Memory extraction failed for user=%s: %s",
            user_id,
            res,
            exc_info=(type(res), res, res.__traceback__),
        )

    success_count = sum(
        1 for r in results if not isinstance(r, Exception)
    )
    if success_count:
      logger.info(
          "Saved memories for user=%s session=%s "
          "(text=%d, episodic=%d)",
          user_id,
          getattr(session, "id", ""),
          len(text_messages),
          len(episode_messages),
      )

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Search semantic and episodic memory namespaces."""
    try:
      semantic_ns = (*self._namespace_prefix, user_id, "semantic")
      search_tasks: list[Any] = [
          self._store.asearch(
              semantic_ns,
              query=query,
              limit=self._max_semantic * _FETCH_MULTIPLIER,
          )
      ]

      if self._enable_episodic:
        episodic_ns = (*self._namespace_prefix, user_id, "episodic")
        search_tasks.append(
            self._store.asearch(
                episodic_ns,
                query=query,
                limit=self._max_episodic * _FETCH_MULTIPLIER,
            )
        )

      raw_results = await asyncio.gather(
          *search_tasks, return_exceptions=True
      )

      semantic_raw = raw_results[0]
      episodic_raw = raw_results[1] if len(raw_results) > 1 else []

      if isinstance(semantic_raw, Exception):
        logger.error("Semantic search failed: %s", semantic_raw)
        semantic_raw = []
      if isinstance(episodic_raw, Exception):
        logger.error("Episodic search failed: %s", episodic_raw)
        episodic_raw = []

      semantic_items = sorted(
          semantic_raw,
          key=lambda item: getattr(item, "updated_at", 0),
          reverse=True,
      )[: self._max_semantic]

      episodic_items = sorted(
          episodic_raw,
          key=lambda item: getattr(item, "updated_at", 0),
          reverse=True,
      )[: self._max_episodic]

      entries: list[MemoryEntry] = []
      for item in semantic_items + episodic_items:
        value = getattr(item, "value", {})
        if isinstance(value, dict):
          raw = str(value.get("content", ""))
        else:
          raw = str(value)
        if raw.strip():
          entries.append(
              MemoryEntry(
                  content=types.Content(
                      parts=[types.Part(text=raw)]
                  ),
                  author="user",
              )
          )

      logger.info(
          "Found %d memories for user=%s (semantic=%d, episodic=%d)",
          len(entries),
          user_id,
          len(semantic_items),
          len(episodic_items),
      )
      return SearchMemoryResponse(memories=entries)

    except Exception as e:
      logger.error("Memory search failed for user=%s: %s", user_id, e)
      return SearchMemoryResponse(memories=[])

  def _extract_messages(
      self, session: Session
  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract LangChain-format messages from session events.

    Returns:
        (text_messages, episode_messages) where text_messages are
        text-only turns for semantic extraction and episode_messages
        include tool call narratives for episodic extraction.
    """
    text_messages: list[dict[str, Any]] = []
    episode_parts: list[str] = []
    has_tool_calls = False

    for event in getattr(session, "events", []):
      content = getattr(event, "content", None)
      if not content or not getattr(content, "parts", None):
        continue
      role = (
          "assistant"
          if getattr(event, "author", None) == "model"
          else "user"
      )
      for part in content.parts:
        if getattr(part, "text", None) and not getattr(
            part, "thought", False
        ):
          text_messages.append(
              {"role": role, "content": part.text}
          )
          episode_parts.append(
              f"{role.capitalize()}: {part.text}"
          )
        fc = getattr(part, "function_call", None)
        if fc and getattr(fc, "name", None):
          has_tool_calls = True
          try:
            args_dict = dict(fc.args) if fc.args else {}
            args_str = json.dumps(args_dict)
          except Exception:
            args_str = str(fc.args)
          episode_parts.append(
              f"[Tool call: {fc.name}({args_str[:500]})]"
          )
        fr = getattr(part, "function_response", None)
        if fr and getattr(fr, "name", None):
          try:
            resp_dict = (
                dict(fr.response) if fr.response else {}
            )
            result_str = json.dumps(resp_dict)
          except Exception:
            result_str = str(getattr(fr, "response", ""))
          episode_parts.append(
              f"[Tool result: {fr.name} -> {result_str[:500]}]"
          )

    if has_tool_calls and episode_parts:
      collapsed = (
          "Tool call sequence:\n" + "\n".join(episode_parts)
      )
      episode_messages = [{"role": "user", "content": collapsed}]
    else:
      episode_messages = []

    return text_messages, episode_messages
