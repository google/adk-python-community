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
from typing import Any, Dict, List, Optional, Union

import httpx
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import BasePlugin
from google.genai import types

from .client import GoodmemClient


class GoodmemChatPlugin(BasePlugin):
  """ADK plugin for persistent chat memory tracking using Goodmem.

  Logs user messages and LLM responses, and retrieves relevant history
  to augment prompts with context.

  Attributes:
    debug: Whether debug mode is enabled.
    goodmem_client: The Goodmem API client.
    embedder_id: The embedder ID used for the space (resolved on first use).
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

    No network calls are made in the constructor. Embedder resolution and
    validation are deferred until first use (e.g. when creating a chat space).

    Args:
      base_url: The base URL for the Goodmem API.
      api_key: The API key for authentication.
      name: The name of the plugin.
      embedder_id: The embedder ID to use. If not provided, the first
        available embedder is used when first needed.
      top_k: The number of top-k most relevant entries to retrieve.
      debug: Whether to enable debug mode.

    Raises:
      ValueError: If base_url or api_key is None.
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

    self.goodmem_client = GoodmemClient(base_url, api_key, debug=self.debug)
    self._embedder_id = embedder_id
    self._resolved_embedder_id: Optional[str] = None
    self.top_k: int = top_k

  def _get_embedder_id(self) -> str:
    """Returns the embedder ID, resolving and validating on first use.

    Fetches embedders from the API only when first needed (e.g. when
    creating a new space). Result is cached for subsequent use.

    Returns:
      The resolved embedder ID.

    Raises:
      ValueError: If no embedders are available or embedder_id is invalid.
    """
    if self._resolved_embedder_id is not None:
      return self._resolved_embedder_id

    embedders = self.goodmem_client.list_embedders()
    if not embedders:
      raise ValueError(
          "No embedders available in Goodmem. Please create at least one "
          "embedder in Goodmem."
      )

    if self._embedder_id is None:
      resolved = embedders[0].get("embedderId", None)
    else:
      if self._embedder_id in [e.get("embedderId") for e in embedders]:
        resolved = self._embedder_id
      else:
        raise ValueError(
            f"EMBEDDER_ID {self._embedder_id} is not valid. Please provide a "
            "valid embedder ID"
        )

    if resolved is None:
      raise ValueError(
          "EMBEDDER_ID is not set and no embedders available in Goodmem."
      )

    self._resolved_embedder_id = resolved
    return resolved

  @property
  def embedder_id(self) -> str:
    """Resolved embedder ID (validated on first access)."""
    return self._get_embedder_id()

  def _is_mime_type_supported(self, mime_type: str) -> bool:
    """Checks if a MIME type is supported by Goodmem's TextContentExtractor.

    Based on the Goodmem source code, TextContentExtractor supports:
    - All text/* MIME types
    - application/pdf
    - application/rtf
    - application/msword (.doc)
    - application/vnd.openxmlformats-officedocument.wordprocessingml.document (.docx)
    - Any MIME type containing "+xml" (e.g., application/xhtml+xml, application/epub+zip)
    - Any MIME type containing "json" (e.g., application/json)

    Args:
      mime_type: The MIME type to check (e.g., "image/png", "application/pdf").

    Returns:
      True if the MIME type is supported by Goodmem, False otherwise.
    """
    if not mime_type:
      return False

    mime_type_lower = mime_type.lower()

    # All text/* types are supported
    if mime_type_lower.startswith("text/"):
      return True

    # Specific application types
    if mime_type_lower in (
        "application/pdf",
        "application/rtf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ):
      return True

    # XML-based formats (contains "+xml")
    if "+xml" in mime_type_lower:
      return True

    # JSON formats (contains "json")
    if "json" in mime_type_lower:
      return True

    return False

  def _get_space_id(
      self, context: Union[InvocationContext, CallbackContext]
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
    try:
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

      # Search for existing space
      if self.debug:
        print(f"[DEBUG] Checking if {space_name} space exists...")
      spaces = self.goodmem_client.list_spaces(name=space_name)
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

      response = self.goodmem_client.create_space(
          space_name, self._get_embedder_id()
      )
      space_id = response.get("spaceId")

      if space_id:
        # Cache in session state for future callbacks
        state['_goodmem_space_id'] = space_id
        if self.debug:
          print(f"[DEBUG] Created new chat space: {space_id}")
        return space_id

      return None

    except httpx.HTTPError as e:
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
    
    Note: Only filters files for Goodmem storage. All files are passed through to
    the LLM without filtering. If the LLM doesn't support a file type (e.g., Gemini
    rejecting zip files), the error will propagate to the application layer. ADK plugins
    cannot catch LLM errors because the LLM call happens outside the plugin callback
    chain (between before_model_callback and after_model_callback). This is a design
    limitation of Google ADK - error handling for LLM failures must be done at the
    application level, not in plugins.

    Args:
      invocation_context: The invocation context containing user info.
      user_message: The user message content.

    Returns:
      None to allow normal processing to continue (all files go to LLM).
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

          # Only filter for Goodmem - let all files through to LLM
          # If LLM doesn't support a file type, it will return an error that
          # should be handled by the application (ADK doesn't provide error
          # callbacks for LLM failures in plugins)
          if not self._is_mime_type_supported(mime_type):
            # Always log skipped files (not just in debug mode) so users know
            # why their files aren't being stored in Goodmem
            print(
                f"[WARNING] Skipping file attachment '{display_name}' "
                f"for Goodmem storage (MIME type '{mime_type}' is not supported by Goodmem). "
                f"Supported types: text/*, application/pdf, application/rtf, "
                f"application/msword, application/vnd.openxmlformats-officedocument.wordprocessingml.document, "
                f"*+xml, *json. The file will still be sent to the LLM."
            )
            if self.debug:
              print(f"[DEBUG] Detailed skip reason: MIME type {mime_type} failed support check")
            # Don't send to Goodmem, but file will still go to LLM
            continue

          # Defensive check: double-verify before sending to Goodmem
          # This should never trigger if filtering is working correctly
          if not self._is_mime_type_supported(mime_type):
            print(
                f"[ERROR] Internal error: Attempted to send unsupported MIME type "
                f"'{mime_type}' to Goodmem. This should not happen. "
                f"File '{display_name}' will be skipped."
            )
            continue

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
          # Note: file_data references are not sent to Goodmem, so no
          # exclusion check needed here

      return None

    except httpx.HTTPError as e:
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
    except (ValueError, OSError, OverflowError):
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

  def _format_timestamp_for_table(self, timestamp_ms: int) -> str:
    """Formats timestamp for table display.

    Args:
      timestamp_ms: Timestamp in milliseconds.

    Returns:
      Formatted timestamp string in yyyy-mm-dd hh:mm format.
    """
    try:
      dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
      return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, OSError, OverflowError):
      return str(timestamp_ms)

  def _wrap_content(self, content: str, max_width: int = 55) -> List[str]:
    """Wraps content to fit within max_width characters.

    Args:
      content: The content to wrap.
      max_width: Maximum width in characters.

    Returns:
      List of wrapped lines.
    """
    lines = []
    words = content.split()
    current_line: List[str] = []
    current_length = 0

    for word in words:
      word_length = len(word)
      # If adding this word would exceed max_width, start a new line
      if current_length > 0 and current_length + 1 + word_length > max_width:
        lines.append(" ".join(current_line))
        current_line = [word]
        current_length = word_length
      else:
        current_line.append(word)
        current_length += (1 + word_length if current_length > 0 else word_length)

    if current_line:
      lines.append(" ".join(current_line))

    return lines if lines else [""]

  def _format_debug_table(
      self,
      records: List[Dict[str, Any]]
  ) -> str:
    """Formats memory records as a table for debug output.

    Args:
      records: List of dicts with keys: memory_id, timestamp_ms, role, content.

    Returns:
      Formatted table string.
    """
    if not records:
      return ""

    # Calculate column widths
    id_width = max(len(r["memory_id"]) for r in records)
    datetime_width = 16  # yyyy-mm-dd hh:mm
    role_width = max(len(r["role"]) for r in records)
    content_width = 55

    # Header
    header = (
        f"{'memory ID':<{id_width}} | "
        f"{'datetime':<{datetime_width}} | "
        f"{'role':<{role_width}} | "
        f"{'content':<{content_width}}"
    )
    separator = "-" * len(header)

    lines = [header, separator]

    # Rows
    for record in records:
      memory_id = record["memory_id"]
      datetime_str = self._format_timestamp_for_table(record["timestamp_ms"])
      role = record["role"]
      content_lines = self._wrap_content(record["content"], content_width)

      # First line with all columns
      if content_lines:
        first_line = (
            f"{memory_id:<{id_width}} | "
            f"{datetime_str:<{datetime_width}} | "
            f"{role:<{role_width}} | "
            f"{content_lines[0]:<{content_width}}"
        )
        lines.append(first_line)

        # Additional lines for wrapped content (only content column)
        for content_line in content_lines[1:]:
          lines.append(
              f"{'':<{id_width}} | "
              f"{'':<{datetime_width}} | "
              f"{'':<{role_width}} | "
              f"{content_line:<{content_width}}"
          )
      else:
        lines.append(
            f"{memory_id:<{id_width}} | "
            f"{datetime_str:<{datetime_width}} | "
            f"{role:<{role_width}} | "
            f"{'':<{content_width}}"
        )

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

      if not chunks:
        return None

      def get_chunk_data(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
          return item["retrievedItem"]["chunk"]["chunk"]
        except (KeyError, TypeError) as e:
          if self.debug:
            print(f"[DEBUG] Error extracting chunk data: {e}")
            print(f"[DEBUG] Item structure: {item}")
          return None

      chunks_cleaned = [get_chunk_data(item) for item in chunks]
      chunks_cleaned = [c for c in chunks_cleaned if c is not None]

      unique_memory_ids_raw: set[Optional[Any]] = set(
          chunk_data.get("memoryId") if chunk_data else None for chunk_data in chunks_cleaned
      )
      unique_memory_ids: set[str] = {mid for mid in unique_memory_ids_raw if mid is not None and isinstance(mid, str)}

      memory_metadata_cache: Dict[str, Dict[str, Any]] = {}
      try:
        batch = self.goodmem_client.get_memories_batch(list(unique_memory_ids))
        for full_memory in batch:
          mid = full_memory.get("memoryId")
          if mid is not None:
            memory_metadata_cache[mid] = full_memory.get("metadata", {})
        for memory_id in unique_memory_ids:
          if memory_id not in memory_metadata_cache:
            memory_metadata_cache[memory_id] = {}
      except httpx.HTTPError as e:
        if self.debug:
          print(f"[DEBUG] Failed to batch-fetch metadata for memories: {e}")
        for memory_id in unique_memory_ids:
          memory_metadata_cache[memory_id] = {}

      formatted_records: List[str] = []
      debug_records: List[Dict[str, Any]] = []
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

        # Prepare debug record
        role = metadata.get("role", "user").lower()
        content = chunk_text
        if content.startswith("User: "):
          content = content[6:]
        elif content.startswith("LLM: "):
          content = content[5:]
        debug_records.append({
            "memory_id": chunk_memory_id,
            "timestamp_ms": timestamp_ms,
            "role": role,
            "content": content
        })

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
        if debug_records:
          table = self._format_debug_table(debug_records)
          print(f"[DEBUG] Retrieved memories:\n{table}")
        else:
          print("[DEBUG] Retrieved memories: none")

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

    except httpx.HTTPError as e:
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

    space_id = self._get_space_id(callback_context)

    if not space_id:
      if self.debug:
        print("[DEBUG] No space_id in after_model_callback, returning None")
      return None

    try:
      response_content: str = ""

      if hasattr(llm_response, "content") and llm_response.content:
        content = llm_response.content

        if hasattr(content, "text"):
          response_content = content.text
        elif hasattr(content, "parts") and content.parts:
          for part in content.parts:
            if hasattr(part, "text") and isinstance(part.text, str):
              response_content += part.text
        elif isinstance(content, str):
          response_content = content
      elif hasattr(llm_response, "text"):
        response_content = llm_response.text

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

      content_with_prefix = f"LLM: {response_content}"
      self.goodmem_client.insert_memory(
          space_id, content_with_prefix, "text/plain", metadata=metadata
      )
      if self.debug:
        print("[DEBUG] Successfully inserted LLM response to Goodmem")

      return None

    except httpx.HTTPError as e:
      if self.debug:
        print(f"[DEBUG] Error in after_model_callback: {e}")
        import traceback
        traceback.print_exc()
      return None
