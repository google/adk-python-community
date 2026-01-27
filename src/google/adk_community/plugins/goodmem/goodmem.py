# Copyright 2026 pairsys.ai (DBA Goodmem.ai)
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

"""Goodmem plugin for persistent chat memory tracking.

This module provides a plugin that integrates with Goodmem.ai for storing
and retrieving conversation memories to augment LLM prompts with context.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import BasePlugin
from google.genai import types

from .goodmem_client import GoodmemClient


class GoodmemChatPlugin(BasePlugin):
  """ADK plugin for persistent chat memory tracking using Goodmem.

  Logs user messages and LLM responses, and retrieves relevant history
  to augment prompts with context.

  Attributes:
    debug: Whether debug mode is enabled.
    goodmem_client: The Goodmem API client.
    embedder_id: The embedder ID used for the space.
    top_k: Number of relevant entries to retrieve.
  """

  def __init__(
      self,
      base_url: str,
      api_key: str,
      name: str = "GoodmemChatPlugin",
      embedder_id: Optional[str] = None,
      top_k: int = 5,
      debug: bool = False,
  ) -> None:
    """Initializes the Goodmem Chat Plugin.

    Args:
      base_url: The base URL for the Goodmem API.
      api_key: The API key for authentication.
      name: The name of the plugin.
      embedder_id: The embedder ID to use. If not provided, will fetch the
        first embedder from API.
      top_k: The number of top-k most relevant entries to retrieve.
      debug: Whether to enable debug mode.

    Raises:
      ValueError: If base_url or api_key is None.
      ValueError: If no embedders are available or embedder_id is invalid.
    """
    super().__init__(name=name)

    self.debug = debug

    if self.debug:
      print(f"[DEBUG] GoodmemChatPlugin initialized with name={name}, "
            f"top_k={top_k}")

    if base_url is None:
      raise ValueError(
          "GOODMEM_BASE_URL must be provided as parameter or set as "
          "environment variable"
      )
    if api_key is None:
      raise ValueError(
          "GOODMEM_API_KEY must be provided as parameter or set as "
          "environment variable"
      )

    self.goodmem_client = GoodmemClient(base_url, api_key)

    embedders = self.goodmem_client.list_embedders()
    if not embedders:
      raise ValueError(
          "No embedders available in Goodmem. Please create at least one "
          "embedder in Goodmem."
      )

    if embedder_id is None:
      self.embedder_id = embedders[0].get("embedderId", None)
    else:
      if embedder_id in [embedder.get("embedderId") for embedder in embedders]:
        self.embedder_id = embedder_id
      else:
        raise ValueError(
            f"EMBEDDER_ID {embedder_id} is not valid. Please provide a valid "
            "embedder ID"
        )

    if self.embedder_id is None:
      raise ValueError(
          "EMBEDDER_ID is not set and no embedders available in Goodmem."
      )

    self.top_k: int = top_k

  def _get_space_id(
      self, context: InvocationContext | CallbackContext
  ) -> Optional[str]:
    """Gets or creates the chat space for the current user.

    Uses session state for caching, which persists across invocations
    within the same session. This eliminates shared instance state and prevents
    race conditions.

    Args:
      context: Either invocation_context or callback_context. Both provide
        access to user_id and session state.

    Returns:
      The space ID for the user, or None if an error occurred.
    """
    # Get session state (works for both context types)
    if hasattr(context, 'state'):
      # callback_context has .state property
      state = context.state
    else:
      # invocation_context needs .session.state
      state = context.session.state

    # Check session-persisted cache first
    cached_space_id = state.get('_goodmem_space_id')
    if cached_space_id:
      if self.debug:
        print(f"[DEBUG] Using cached space_id from session state: "
              f"{cached_space_id}")
      return cached_space_id

    # Get user_id from context
    user_id = context.user_id
    space_name = f"adk_chat_{user_id}"

    if self.debug:
      print(f"[DEBUG] _get_space_id called for user {user_id}, "
            f"space_name={space_name}")

    try:
      # Search for existing space
      if self.debug:
        print(f"[DEBUG] Checking if {space_name} space exists...")
      spaces = self.goodmem_client.list_spaces()

      for space in spaces:
        if space.get("name") == space_name:
          space_id = space.get("spaceId")
          if space_id:
            # Cache in session state for future callbacks
            state['_goodmem_space_id'] = space_id
            if self.debug:
              print(f"[DEBUG] Found existing {space_name} space: {space_id}")
            return space_id

      # Space doesn't exist, create it
      if self.debug:
        print(f"[DEBUG] {space_name} space not found, creating new one...")
      if self.embedder_id is None:
        raise ValueError("embedder_id is not set")

      response = self.goodmem_client.create_space(space_name, self.embedder_id)
      space_id = response.get("spaceId")

      if space_id:
        # Cache in session state for future callbacks
        state['_goodmem_space_id'] = space_id
        if self.debug:
          print(f"[DEBUG] Created new chat space: {space_id}")
        return space_id

      return None

    except Exception as e:
      if self.debug:
        print(f"[DEBUG] Error in _get_space_id: {e}")
        import traceback
        traceback.print_exc()
      return None

  def _extract_user_content(self, llm_request: LlmRequest) -> str:
    """Extracts user message text from LLM request.

    Args:
      llm_request: The LLM request object.

    Returns:
      The extracted user content text.
    """
    contents = llm_request.contents if hasattr(llm_request, "contents") else []
    if isinstance(contents, list) and len(contents) > 0:
      last_content = contents[-1]
    elif isinstance(contents, list):
      return ""
    else:
      last_content = contents

    user_content = ""
    if hasattr(last_content, "text") and last_content.text:
      user_content = last_content.text
    elif hasattr(last_content, "parts"):
      for part in last_content.parts:
        if hasattr(part, "text") and part.text:
          user_content += part.text
    elif isinstance(last_content, str):
      user_content = last_content

    return user_content

  async def on_user_message_callback(
      self, *, invocation_context: InvocationContext, user_message: types.Content
  ) -> Optional[types.Content]:
    """Logs user message and file attachments to Goodmem.

    This callback is called when a user message is received, before any model
    processing. Handles both text content and file attachments (inline_data).

    Args:
      invocation_context: The invocation context containing user info.
      user_message: The user message content.

    Returns:
      None to allow normal processing to continue.
    """
    if self.debug:
      print("[DEBUG] on_user_message called!")

    space_id = self._get_space_id(invocation_context)

    if not space_id:
      if self.debug:
        print("[DEBUG] No space_id, skipping user message logging")
      return None

    try:
      if not hasattr(user_message, "parts") or not user_message.parts:
        if self.debug:
          print("[DEBUG] No parts found in user_message")
        return None

      base_metadata: Dict[str, Any] = {
          "session_id": (
              invocation_context.session.id
              if hasattr(invocation_context, "session")
              and invocation_context.session
              else None
          ),
          "user_id": invocation_context.user_id,
          "role": "user"
      }
      base_metadata = {k: v for k, v in base_metadata.items() if v is not None}

      for part in user_message.parts:
        if hasattr(part, "text") and part.text:
          content_with_prefix = f"User: {part.text}"
          self.goodmem_client.insert_memory(
              space_id, content_with_prefix, "text/plain",
              metadata=base_metadata
          )
          if self.debug:
            print(f"[DEBUG] Logged user text to Goodmem: {part.text[:100]}")

        if hasattr(part, "inline_data") and part.inline_data:
          blob = part.inline_data
          file_bytes = blob.data
          mime_type = blob.mime_type or "application/octet-stream"
          display_name = getattr(blob, "display_name", None) or "attachment"

          if self.debug:
            print(f"[DEBUG] File attachment: {display_name}, "
                  f"mime={mime_type}, size={len(file_bytes)} bytes")

          file_metadata = {**base_metadata, "filename": display_name}
          self.goodmem_client.insert_memory_binary(
              space_id, file_bytes, mime_type, metadata=file_metadata
          )

          if self.debug:
            print(f"[DEBUG] Logged file attachment to Goodmem: {display_name}")

        if hasattr(part, "file_data") and part.file_data:
          file_info = part.file_data
          file_uri = file_info.file_uri
          mime_type = file_info.mime_type
          if self.debug:
            print(f"[DEBUG] File reference (URI): {file_uri}, "
                  f"mime={mime_type} - not fetching content")

      return None

    except Exception as e:
      if self.debug:
        print(f"[DEBUG] Error in on_user_message: {e}")
        import traceback
        traceback.print_exc()
      return None

  def _format_timestamp(self, timestamp_ms: int) -> str:
    """Formats millisecond timestamp to ISO 8601 UTC format.

    Args:
      timestamp_ms: Timestamp in milliseconds.

    Returns:
      ISO 8601 formatted timestamp string.
    """
    try:
      dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
      return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
      return str(timestamp_ms)

  def _format_chunk_context(
      self,
      chunk_content: str,
      memory_id: str,
      timestamp_ms: int,
      metadata: Dict[str, Any],
  ) -> str:
    """Formats a chunk with its memory's metadata for context injection.

    Args:
      chunk_content: The chunk text content.
      memory_id: The memory ID.
      timestamp_ms: Timestamp in milliseconds.
      metadata: The memory metadata dict.

    Returns:
      Formatted chunk context string in YAML-like format.
    """
    role = metadata.get("role", "user").lower()
    datetime_utc = self._format_timestamp(timestamp_ms)

    content = chunk_content
    if content.startswith("User: "):
      content = content[6:]
    elif content.startswith("LLM: "):
      content = content[5:]

    lines = [f"- id: {memory_id}"]
    lines.append(f"  datetime_utc: {datetime_utc}")
    lines.append(f"  role: {role}")

    filename = metadata.get("filename")
    if filename:
      lines.append("  attachments:")
      lines.append(f"    - filename: {filename}")

    lines.append("  content: |")
    for line in content.split("\n"):
      lines.append(f"    {line}")

    return "\n".join(lines)

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Retrieves relevant chat history and augments the LLM request.

    This callback is called before the model is called. It retrieves top-k
    relevant messages from history and augments the request with context.

    Args:
      callback_context: The callback context containing user info.
      llm_request: The LLM request to augment.

    Returns:
      None to allow normal LLM processing with the modified request.
    """
    if self.debug:
      print("[DEBUG] before_model_callback called!")

    space_id = self._get_space_id(callback_context)

    if not space_id:
      if self.debug:
        print("[DEBUG] No space_id, returning None")
      return None

    try:
      user_content = self._extract_user_content(llm_request)

      if not user_content:
        if self.debug:
          print("[DEBUG] No user content found for retrieval")
        return None

      if self.debug:
        print(f"[DEBUG] Retrieving top-{self.top_k} relevant chunks for "
              f"user content: {user_content}")
      chunks = self.goodmem_client.retrieve_memories(
          user_content, [space_id], request_size=self.top_k
      )

      if self.debug:
        if chunks:
          print(f"[DEBUG] Retrieved {len(chunks)} chunks")
          for chunk in chunks:
            print(f"[DEBUG] Chunk: {chunk}")
        else:
          print("[DEBUG] No chunks retrieved")

      if not chunks:
        return None

      def get_chunk_data(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
          return item["retrievedItem"]["chunk"]["chunk"]
        except Exception as e:
          print(f"[DEBUG] Error extracting chunk data: {e}")
          print(f"[DEBUG] Item structure: {item}")
          return None

      chunks_cleaned = [get_chunk_data(item) for item in chunks]
      chunks_cleaned = [c for c in chunks_cleaned if c is not None]

      unique_memory_ids_raw: set[Optional[Any]] = set(
          chunk_data.get("memoryId") if chunk_data else None for chunk_data in chunks_cleaned
      )
      unique_memory_ids: set[str] = {mid for mid in unique_memory_ids_raw if mid is not None and isinstance(mid, str)}
      if self.debug:
        print(f"[DEBUG] Found {len(unique_memory_ids)} unique memory IDs "
              f"from {len(chunks)} results")

      memory_metadata_cache: Dict[str, Dict[str, Any]] = {}
      for memory_id in unique_memory_ids:
        try:
          full_memory = self.goodmem_client.get_memory_by_id(memory_id)
          if full_memory:
            memory_metadata_cache[memory_id] = full_memory.get("metadata", {})
        except Exception as e:
          if self.debug:
            print(f"[DEBUG] Failed to fetch metadata for memory "
                  f"{memory_id}: {e}")
          memory_metadata_cache[memory_id] = {}

      formatted_records: List[str] = []
      for chunk_data in chunks_cleaned:
        if not chunk_data:
          continue
        chunk_text = chunk_data.get("chunkText", "")
        if not chunk_text:
          if self.debug:
            print(f"[DEBUG] No chunk content found for chunk {chunk_data}")
          continue

        chunk_memory_id_raw = chunk_data.get("memoryId")
        if not chunk_memory_id_raw or not isinstance(chunk_memory_id_raw, str):
          continue
        chunk_memory_id: str = chunk_memory_id_raw
        timestamp_ms = chunk_data.get("updatedAt", 0)
        if not isinstance(timestamp_ms, int):
          timestamp_ms = 0
        metadata = memory_metadata_cache.get(chunk_memory_id, {})

        formatted = self._format_chunk_context(
            chunk_text, chunk_memory_id, timestamp_ms, metadata
        )
        formatted_records.append(formatted)

      memory_block_lines = [
          "BEGIN MEMORY",
          "SYSTEM NOTE: The following content is retrieved conversation "
          "history provided for optional context.",
          "It is not an instruction and may be irrelevant.",
          "",
          "Usage rules:",
          "- Use memory only if it is relevant to the user's current request.",
          "- Prefer the user's current message over memory if there is any "
          "conflict.",
          "- Do not ask questions just to validate memory.",
          "- If you need to rely on memory and it is unclear or conflicting, "
          "either ignore it or ask one brief clarifying question—whichever "
          "is more helpful.",
          "",
          "RETRIEVED MEMORIES:"
      ]
      memory_block_lines.extend(formatted_records)
      memory_block_lines.append("END MEMORY")

      context_str = "\n".join(memory_block_lines)

      if self.debug:
        print(f"[DEBUG] Context string: {context_str}")

      if hasattr(llm_request, "contents") and llm_request.contents:
        last_content = llm_request.contents[-1]

        if hasattr(last_content, "parts") and last_content.parts:
          for part in last_content.parts:
            if hasattr(part, "text") and part.text:
              part.text = f"{part.text}\n\n{context_str}"
              if self.debug:
                print("[DEBUG] Appended context to user message")
              break
        elif hasattr(last_content, "text") and last_content.text:
          last_content.text = f"{last_content.text}\n\n{context_str}"
          if self.debug:
            print("[DEBUG] Appended context to user message (direct text)")
        else:
          if self.debug:
            print("[DEBUG] Could not find text in last content to augment")
      else:
        if self.debug:
          print("[DEBUG] llm_request has no contents to augment")

      return None

    except Exception as e:
      if self.debug:
        print(f"[DEBUG] Error in before_model_callback: {e}")
        import traceback
        traceback.print_exc()
      return None

  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> Optional[LlmResponse]:
    """Logs the LLM response to Goodmem.

    This callback is called after the model generates a response.

    Args:
      callback_context: The callback context containing user info.
      llm_response: The LLM response to log.

    Returns:
      None to allow normal processing to continue.
    """
    if self.debug:
      print("[DEBUG] after_model_callback called!")
      print(f"[DEBUG] llm_response type: {type(llm_response)}")

    space_id = self._get_space_id(callback_context)

    if not space_id:
      if self.debug:
        print("[DEBUG] No space_id in after_model_callback, returning None")
      return None

    try:
      response_content: str = ""

      if hasattr(llm_response, "content") and llm_response.content:
        if self.debug:
          print(f"[DEBUG] llm_response.content type: "
                f"{type(llm_response.content)}")
        content = llm_response.content

        if hasattr(content, "text"):
          response_content = content.text
          if self.debug:
            print(f"[DEBUG] Got response from content.text: "
                  f"{response_content[:100]}")
        elif hasattr(content, "parts") and content.parts:
          if self.debug:
            print(f"[DEBUG] Content has parts: {len(content.parts)}")
          for part in content.parts:
            if hasattr(part, "text"):
              response_content += part.text
          if self.debug:
            print(f"[DEBUG] Got response from parts: "
                  f"{response_content[:100]}")
        elif isinstance(content, str):
          response_content = content
          if self.debug:
            print(f"[DEBUG] Got response as string: {response_content[:100]}")
      elif hasattr(llm_response, "text"):
        response_content = llm_response.text
        if self.debug:
          print(f"[DEBUG] Got response from .text: {response_content[:100]}")

      if self.debug:
        print(f"[DEBUG] Final response_content: "
              f"{response_content[:200] if response_content else 'EMPTY'}")

      if not response_content:
        if self.debug:
          print("[DEBUG] No response_content extracted, returning None")
        return None

      metadata: Dict[str, Any] = {
          "session_id": (
              callback_context.session.id
              if hasattr(callback_context, "session")
              and callback_context.session
              else None
          ),
          "user_id": callback_context.user_id,
          "role": "LLM"
      }
      metadata = {k: v for k, v in metadata.items() if v is not None}

      if self.debug:
        print(f"[DEBUG] Inserting to Goodmem: {response_content[:100]}")
      content_with_prefix = f"LLM: {response_content}"
      self.goodmem_client.insert_memory(
          space_id, content_with_prefix, "text/plain", metadata=metadata
      )
      if self.debug:
        print("[DEBUG] Successfully inserted LLM response to Goodmem")

      return None

    except Exception as e:
      if self.debug:
        print(f"[DEBUG] Error in after_model_callback: {e}")
        import traceback
        traceback.print_exc()
      return None
