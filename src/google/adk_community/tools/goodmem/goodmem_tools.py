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

"""Goodmem tools for writing to and retrieving from Goodmem storage.

This module provides tools that allow agents to explicitly manage persistent
memory storage using Goodmem.ai:
- goodmem_save: Write content to memory with automatic metadata
- goodmem_fetch: Search and retrieve memories using semantic search
"""

from __future__ import annotations

import inspect
from datetime import datetime
from datetime import timezone
import threading
from typing import Dict
from typing import List
from typing import Optional
from typing import TypedDict

from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel
from pydantic import Field
from pydantic import JsonValue
import httpx

from google.adk_community.plugins.goodmem import GoodmemClient

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Module-level client cache to avoid recreating on every call
_client_cache: Dict[tuple[str, str, bool], GoodmemClient] = {}
_client_cache_lock = threading.Lock()

class DebugRecord(TypedDict):
  """Record used for debug table rendering."""

  memory_id: str
  timestamp_ms: Optional[int]
  role: str
  content: str


class ChunkData(TypedDict):
  memoryId: str
  chunkText: str
  updatedAt: Optional[int]


def _format_timestamp_for_table(timestamp_ms: Optional[int]) -> str:
  """Formats timestamp for table display.

  Args:
    timestamp_ms: Timestamp in milliseconds.

  Returns:
    Formatted timestamp string in yyyy-mm-dd hh:mm format.
  """
  if timestamp_ms is None:
    return ""
  try:
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M")
  except Exception:
    return str(timestamp_ms)


def _wrap_content(content: str, max_width: int = 55) -> List[str]:
  """Wraps content to fit within max_width characters.

  Args:
    content: The content to wrap.
    max_width: Maximum width in characters.

  Returns:
    List of wrapped lines.
  """
  lines: List[str] = []
  words: List[str] = content.split()
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
      current_length += 1 + word_length if current_length > 0 else word_length

  if current_line:
    lines.append(" ".join(current_line))

  return lines if lines else [""]


def _format_debug_table(records: List[DebugRecord]) -> str:
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
    datetime_str = _format_timestamp_for_table(record["timestamp_ms"])
    role = record["role"]
    content_lines = _wrap_content(record["content"], content_width)

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


def _extract_chunk_data(item: object) -> Optional[ChunkData]:
  """Extracts chunk data from a Goodmem retrieval item.

  Args:
    item: The raw NDJSON item from Goodmem.

  Returns:
    A ChunkData dict if the structure is valid, otherwise None.
  """
  if not isinstance(item, dict):
    return None

  retrieved_item = item.get("retrievedItem")
  if not isinstance(retrieved_item, dict):
    return None

  chunk_wrapper = retrieved_item.get("chunk")
  if not isinstance(chunk_wrapper, dict):
    return None

  chunk_data = chunk_wrapper.get("chunk")
  if not isinstance(chunk_data, dict):
    return None

  memory_id = chunk_data.get("memoryId")
  chunk_text = chunk_data.get("chunkText")
  updated_at = chunk_data.get("updatedAt")

  if not isinstance(memory_id, str) or not isinstance(chunk_text, str):
    return None
  if updated_at is not None and not isinstance(updated_at, int):
    return None

  return {
      "memoryId": memory_id,
      "chunkText": chunk_text,
      "updatedAt": updated_at,
  }


def _get_client(base_url: str, api_key: str, debug: bool) -> GoodmemClient:
  """Get or create a cached GoodmemClient instance.

  Args:
    base_url: The base URL for the Goodmem API.
    api_key: The API key for authentication.

  Returns:
    A cached or new GoodmemClient instance.
  """
  cache_key = (base_url, api_key, debug)
  client = _client_cache.get(cache_key)
  if client is not None:
    if debug:
      print(f"[DEBUG] Using cached GoodmemClient for {base_url}")
    return client

  with _client_cache_lock:
    client = _client_cache.get(cache_key)
    if client is not None:
      if debug:
        print(f"[DEBUG] Using cached GoodmemClient for {base_url}")
      return client

    if debug:
      print(
          "[DEBUG] Creating GoodmemClient for base_url="
          f"{base_url}, debug={debug}"
      )
    client = GoodmemClient(base_url=base_url, api_key=api_key, debug=debug)
    _client_cache[cache_key] = client
    return client


def _get_or_create_space(
    client: GoodmemClient,
    tool_context: ToolContext,
    embedder_id: Optional[str] = None,
    debug: bool = False,
) -> tuple[Optional[str], Optional[str]]:
  """Get or create Goodmem space for the current user.

  Returns a tuple of (space_id, error_message). If error_message is not None,
  space_id will be None.

  Args:
    client: The GoodmemClient instance.
    tool_context: The tool context with user_id and session state.
    embedder_id: Optional embedder ID to use when creating a new space.
      If None, uses the first available embedder.
    debug: Whether to print debug messages.

  Returns:
    Tuple of (space_id, error_message). error_message is None on success.
  """
  # Check cache first
  cached_space_id = tool_context.state.get("_goodmem_space_id")
  if cached_space_id:
    if debug:
      print(
          "[DEBUG] Using cached Goodmem space_id from session state: "
          f"{cached_space_id}"
      )
    return (cached_space_id, None)

  # Construct space name based on user_id
  space_name = f"adk_tool_{tool_context.user_id}"

  try:
    # Search for existing space
    if debug:
      print(f"[DEBUG] Checking for existing space: {space_name}")
    spaces = client.list_spaces(name=space_name)
    for space in spaces:
      if space.get("name") == space_name:
        space_id = space["spaceId"]
        # Cache it for future calls
        tool_context.state["_goodmem_space_id"] = space_id
        if debug:
          print(f"[DEBUG] Found existing space: {space_id}")
        return (space_id, None)

    # Space doesn't exist, need to create it
    if embedder_id:
      # Validate the embedder exists
      embedders = client.list_embedders()
      embedder_ids = [e["embedderId"] for e in embedders]

      if embedder_id not in embedder_ids:
        return (
            None,
            (
                f"Configuration error: embedder_id '{embedder_id}' not"
                f" found. Available embedders: {', '.join(embedder_ids)}"
            ),
        )
    else:
      # Use first available embedder
      embedders = client.list_embedders()
      if not embedders:
        return (None, "Configuration error: No embedders available in Goodmem.")
      embedder_id = embedders[0]["embedderId"]

    # Create the space
    if debug:
      print(
          "[DEBUG] Creating Goodmem space "
          f"{space_name} with embedder_id={embedder_id}"
      )
    response = client.create_space(space_name, embedder_id)
    space_id = response["spaceId"]

    # Cache it
    tool_context.state["_goodmem_space_id"] = space_id
    if debug:
      print(f"[DEBUG] Created new Goodmem space: {space_id}")
    return (space_id, None)

  except httpx.HTTPStatusError as e:
    status_code = e.response.status_code
    if status_code == 409:
      if debug:
        print(
            "[DEBUG] Space already exists; re-fetching space ID after conflict"
        )
      try:
        spaces = client.list_spaces(name=space_name)
        for space in spaces:
          if space.get("name") == space_name:
            space_id = space["spaceId"]
            tool_context.state["_goodmem_space_id"] = space_id
            if debug:
              print(
                  "[DEBUG] Found existing space after conflict: "
                  f"{space_id}"
              )
            return (space_id, None)
      except Exception as list_error:
        if debug:
          print(
              "[DEBUG] Error re-fetching space after conflict: "
              f"{list_error}"
          )
    if debug:
      print(f"[DEBUG] Error getting or creating space: {e}")
    return (None, f"Error getting or creating space: {str(e)}")

  except Exception as e:
    if debug:
      print(f"[DEBUG] Error getting or creating space: {e}")
    return (None, f"Error getting or creating space: {str(e)}")


# ============================================================================
# SAVE TOOL - Write to Goodmem
# ============================================================================


class GoodmemSaveResponse(BaseModel):
  """Response from the goodmem_save tool."""

  success: bool = Field(
      description="Whether the write operation was successful"
  )
  memory_id: Optional[str] = Field(
      default=None, description="The ID of the created memory in Goodmem"
  )
  message: str = Field(description="Status message")


async def goodmem_save(
    content: str,
    tool_context: ToolContext = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    embedder_id: Optional[str] = None,
    debug: bool = False,
) -> GoodmemSaveResponse:
  """Saves important information to persistent memory storage.

  WHEN TO USE:
  - User shares preferences, facts, personal information,
    important decisions, or anything you believe is important or
    worth remembering
  - After solving problems or making decisions worth remembering
  - Proactively save context that would help in future conversations
  - When the user asks you to remember something

  CRITICAL: Always confirm to the user what you saved. Check the 'success' field
  in the response - only claim you saved something if success=True.

  METADATA: user_id and session_id are automatically captured from context.

  Args:
    content: The text content to write to memory storage (plain text only).
    tool_context: The tool execution context (automatically provided by ADK).
    base_url: The base URL for the Goodmem API (required).
    api_key: The API key for authentication (required).
    embedder_id: Optional embedder ID to use when creating new spaces.

  Returns:
    A GoodmemSaveResponse containing the operation status and memory ID.
  """
  if debug:
    print("[DEBUG] goodmem_save called")

  if not base_url:
    return GoodmemSaveResponse(
        success=False,
        message=(
            "Configuration error: base_url is required. Please provide it when"
            " initializing GoodmemSaveTool or pass it as a parameter."
        ),
    )

  if not api_key:
    return GoodmemSaveResponse(
        success=False,
        message=(
            "Configuration error: api_key is required. Please provide it when"
            " initializing GoodmemSaveTool or pass it as a parameter."
        ),
    )

  if not tool_context:
    return GoodmemSaveResponse(
        success=False,
        message=(
            "Configuration error: tool_context is required for automatic space"
            " management. This should be provided automatically by ADK."
        ),
    )

  try:
    # Get cached Goodmem client
    client = _get_client(base_url=base_url, api_key=api_key, debug=debug)

    # Get or create space for this user
    space_id, error = _get_or_create_space(
        client, tool_context, embedder_id=embedder_id, debug=debug
    )
    if error:
      if debug:
        print(f"[DEBUG] Failed to get or create space: {error}")
      return GoodmemSaveResponse(success=False, message=error)
    if space_id is None:
      if debug:
        print("[DEBUG] No space_id returned, aborting dump")
      return GoodmemSaveResponse(
          success=False, message="Failed to get or create space"
      )

    # Build metadata from tool_context
    metadata: Dict[str, JsonValue] = {}

    # Add user_id from tool_context if available
    if tool_context and hasattr(tool_context, "user_id"):
      metadata["user_id"] = tool_context.user_id

    # Add session_id from tool_context if available
    if (
        tool_context
        and hasattr(tool_context, "session")
        and tool_context.session
    ):
      if hasattr(tool_context.session, "id"):
        metadata["session_id"] = tool_context.session.id

    # Insert memory into Goodmem
    if debug:
      print(f"[DEBUG] Inserting memory into space {space_id}")
    response = client.insert_memory(
        space_id=space_id,
        content=content,
        content_type="text/plain",
        metadata=metadata if metadata else None,
    )

    memory_id = response.get("memoryId")
    if debug:
      print(f"[DEBUG] Goodmem insert response memory_id={memory_id}")

    return GoodmemSaveResponse(
        success=True,
        memory_id=memory_id,
        message=f"Successfully wrote content to memory. Memory ID: {memory_id}",
    )

  except Exception as e:
    error_msg = str(e)

    # Determine specific error type
    if isinstance(e, httpx.ConnectError):
      return GoodmemSaveResponse(
          success=False,
          message=(
              f"Connection error: Cannot reach Goodmem server at {base_url}. "
              "Please check if the server is running and the URL is correct. "
              f"Details: {error_msg}"
          ),
      )
    elif isinstance(e, httpx.TimeoutException):
      return GoodmemSaveResponse(
          success=False,
          message=(
              f"Timeout error: Goodmem server at {base_url} is not responding. "
              "Please check your connection or server status."
          ),
      )
    elif isinstance(e, httpx.HTTPStatusError):
      status_code = e.response.status_code
      if status_code in (401, 403):
        return GoodmemSaveResponse(
            success=False,
            message=(
                "Authentication error: Invalid API key. "
                "Please check your GOODMEM_API_KEY is correct. "
                f"HTTP {status_code}"
            ),
        )
      elif status_code == 404:
        return GoodmemSaveResponse(
            success=False,
            message=(
                f"Not found error: Space ID '{space_id}' does not exist. "
                f"The space may have been deleted. HTTP {status_code}"
            ),
        )
      else:
        return GoodmemSaveResponse(
            success=False,
            message=(
                f"Server error: Goodmem API returned HTTP {status_code}. "
                f"Details: {error_msg}"
            ),
        )
    else:
      return GoodmemSaveResponse(
          success=False,
          message=f"Unexpected error while writing to memory: {error_msg}",
      )


class GoodmemSaveTool(FunctionTool):
  """A tool that writes content to Goodmem storage.

  This tool wraps the goodmem_save function and provides explicit memory
  writing capabilities to ADK agents.
  """

  def __init__(
      self,
      base_url: Optional[str] = None,
      api_key: Optional[str] = None,
      embedder_id: Optional[str] = None,
      debug: bool = False,
  ):
    """Initialize the Goodmem save tool.

    Args:
      base_url: The base URL for the Goodmem API (required).
      api_key: The API key for authentication (required).
      embedder_id: Optional embedder ID to use when creating new spaces.
      debug: Enable debug logging.
    """
    self._base_url = base_url
    self._api_key = api_key
    self._embedder_id = embedder_id
    self._debug = debug

    # Create a wrapper function that passes the stored config
    # We need to preserve the function signature for FunctionTool introspection
    async def _wrapped_save(
        content: str,
        tool_context: ToolContext = None,
    ) -> GoodmemSaveResponse:
      return await goodmem_save(
          content=content,
          tool_context=tool_context,
          base_url=self._base_url,
          api_key=self._api_key,
          embedder_id=self._embedder_id,
          debug=self._debug,
      )

    # Preserve function metadata for FunctionTool introspection
    # Copy signature from original function (excluding the config params)
    original_sig = inspect.signature(goodmem_save)
    params = []
    for name, param in original_sig.parameters.items():
      if name not in ("base_url", "api_key", "embedder_id", "debug"):
        params.append(param)
    setattr(
        _wrapped_save,
        "__signature__",
        original_sig.replace(parameters=params),
    )
    _wrapped_save.__name__ = goodmem_save.__name__
    _wrapped_save.__doc__ = goodmem_save.__doc__

    super().__init__(_wrapped_save)


# ============================================================================
# FETCH TOOL - Retrieve from Goodmem
# ============================================================================


class MemoryItem(BaseModel):
  """A single memory item retrieved from Goodmem."""

  memory_id: str = Field(description="The unique ID of the memory")
  content: str = Field(description="The text content of the memory")
  metadata: Dict[str, JsonValue] = Field(
      default_factory=dict,
      description=(
          "Metadata associated with the memory (user_id, session_id, etc.)"
      ),
  )
  updated_at: Optional[int] = Field(
      default=None,
      description="Timestamp when the memory was last updated (milliseconds)",
  )


class GoodmemFetchResponse(BaseModel):
  """Response from the goodmem_fetch tool."""

  success: bool = Field(
      description="Whether the fetch operation was successful"
  )
  memories: List[MemoryItem] = Field(
      default_factory=list, description="List of retrieved memories"
  )
  count: int = Field(default=0, description="Number of memories retrieved")
  message: str = Field(description="Status message")


async def goodmem_fetch(
    query: str,
    top_k: int = 5,
    tool_context: ToolContext = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    embedder_id: Optional[str] = None,
    debug: bool = False,
) -> GoodmemFetchResponse:
  """Searches for relevant memories using semantic search.

  CRITICAL: Use this BEFORE saying "I don't know" to any question about the
  user!

  WHEN TO USE:
  - User asks ANY question about themselves (preferences, history, background,
    facts)
  - User asks about previous conversations, facts, decisions, or other
    important information
  - You believe that the user may have had past interactions that are relevant
  - User asks you to look for history

  RESPONSE HANDLING:
  - When you use retrieved information, explicitly state it came from memory
    Example: "According to my memory, you went to school in Texas"
  - Present all retrieved memories to help answer the user's question
  - You are not required to use all or any of the memories.

  Args:
    query: The search query to find relevant memories (e.g., "user's favorite color").
    top_k: Maximum number of chunks to request (default: 5, max: 20). The
      response is de-duplicated by memory ID, so fewer memories may be returned.
    tool_context: The tool execution context (automatically provided by ADK).
    base_url: The base URL for the Goodmem API (required).
    api_key: The API key for authentication (required).
    embedder_id: Optional embedder ID to use when creating new spaces.

  Returns:
    A GoodmemFetchResponse containing the retrieved memories and metadata.
  """
  if debug:
    print(f"[DEBUG] goodmem_fetch called query='{query}' top_k={top_k}")

  # top_k validation
  if top_k > 20:
    top_k = 20
  if top_k < 1:
    top_k = 1

  if not base_url:
    return GoodmemFetchResponse(
        success=False,
        message=(
            "Configuration error: base_url is required. Please provide it when"
            " initializing GoodmemFetchTool or pass it as a parameter."
        ),
    )

  if not api_key:
    return GoodmemFetchResponse(
        success=False,
        message=(
            "Configuration error: api_key is required. Please provide it when"
            " initializing GoodmemFetchTool or pass it as a parameter."
        ),
    )

  if not tool_context:
    return GoodmemFetchResponse(
        success=False,
        message=(
            "Configuration error: tool_context is required for automatic space"
            " management. This should be provided automatically by ADK."
        ),
    )

  try:
    # Get cached Goodmem client
    client = _get_client(base_url=base_url, api_key=api_key, debug=debug)

    # Get or create space for this user
    space_id, error = _get_or_create_space(
        client, tool_context, embedder_id=embedder_id, debug=debug
    )
    if error:
      if debug:
        print(f"[DEBUG] Failed to get or create space: {error}")
      return GoodmemFetchResponse(success=False, message=error)
    if space_id is None:
      if debug:
        print("[DEBUG] No space_id returned, aborting fetch")
      return GoodmemFetchResponse(
          success=False, message="Failed to get or create space"
      )

    # Retrieve memories using semantic search
    if debug:
      print(f"[DEBUG] Retrieving memories from space {space_id}")
    chunks = client.retrieve_memories(
        query=query, space_ids=[space_id], request_size=top_k
    )

    if not chunks:
      if debug:
        print("[DEBUG] No chunks retrieved from Goodmem")
      return GoodmemFetchResponse(
          success=True,
          memories=[],
          count=0,
          message="No memories found matching the query",
      )

    # Extract memory IDs to fetch full metadata
    memory_ids: set[str] = set()
    chunk_data_list: List[ChunkData] = []

    for item in chunks:
      chunk_data = _extract_chunk_data(item)
      if not chunk_data:
        continue
      chunk_data_list.append(chunk_data)
      memory_ids.add(chunk_data["memoryId"])
    if debug:
      print(
          "[DEBUG] Retrieved "
          f"{len(chunk_data_list)} chunks, {len(memory_ids)} unique memory IDs"
      )

    # Fetch full memory metadata for each unique memory ID
    memory_metadata_cache: Dict[str, Dict[str, JsonValue]] = {}
    for memory_id in memory_ids:
      try:
        full_memory = client.get_memory_by_id(memory_id)
        if full_memory:
          memory_metadata_cache[memory_id] = full_memory.get("metadata", {})
      except Exception:
        memory_metadata_cache[memory_id] = {}

    # Build response with memories
    memories: List[MemoryItem] = []
    seen_memory_ids: set[str] = set()
    # Store role information for debug table (before content is cleaned)
    memory_roles: Dict[str, str] = {}

    for chunk_data in chunk_data_list:
      memory_id = chunk_data.get("memoryId")
      if not memory_id or memory_id in seen_memory_ids:
        continue

      seen_memory_ids.add(memory_id)

      content = chunk_data.get("chunkText", "")
      updated_at = chunk_data.get("updatedAt")
      metadata = memory_metadata_cache.get(memory_id, {})

      # Determine role from content prefix or metadata
      role = "user"  # default
      if content.startswith("User: "):
        role = "user"
        content = content[6:]
      elif content.startswith("LLM: "):
        role = "llm"
        content = content[5:]
      else:
        # Try to get role from metadata
        role_from_metadata = metadata.get("role", "user")
        if isinstance(role_from_metadata, str):
          role = role_from_metadata.lower()
        else:
          role = "user"

      memory_roles[memory_id] = role

      memories.append(
          MemoryItem(
              memory_id=memory_id,
              content=content,
              metadata=metadata,
              updated_at=updated_at,
          )
      )

    # Format debug table if debug mode is enabled
    if debug and memories:
      debug_records: List[DebugRecord] = []
      for memory in memories:
        role = memory_roles.get(memory.memory_id, "user")
        debug_records.append({
            "memory_id": memory.memory_id,
            "timestamp_ms": memory.updated_at,
            "role": role,
            "content": memory.content,
        })

      table = _format_debug_table(debug_records)
      print(f"[DEBUG] Retrieved memories:\n{table}")

    return GoodmemFetchResponse(
        success=True,
        memories=memories,
        count=len(memories),
        message=f"Successfully retrieved {len(memories)} memories",
    )

  except Exception as e:
    error_msg = str(e)

    # Determine specific error type
    if isinstance(e, httpx.ConnectError):
      return GoodmemFetchResponse(
          success=False,
          message=(
              f"Connection error: Cannot reach Goodmem server at {base_url}. "
              "Please check if the server is running and the URL is correct. "
              f"Details: {error_msg}"
          ),
      )
    elif isinstance(e, httpx.TimeoutException):
      return GoodmemFetchResponse(
          success=False,
          message=(
              f"Timeout error: Goodmem server at {base_url} is not responding. "
              "Please check your connection or server status."
          ),
      )
    elif isinstance(e, httpx.HTTPStatusError):
      status_code = e.response.status_code
      if status_code in (401, 403):
        return GoodmemFetchResponse(
            success=False,
            message=(
                "Authentication error: Invalid API key. "
                "Please check your GOODMEM_API_KEY is correct. "
                f"HTTP {status_code}"
            ),
        )
      elif status_code == 404:
        return GoodmemFetchResponse(
            success=False,
            message=(
                f"Not found error: Space ID '{space_id}' does not exist. "
                f"The space may have been deleted. HTTP {status_code}"
            ),
        )
      else:
        return GoodmemFetchResponse(
            success=False,
            message=(
                f"Server error: Goodmem API returned HTTP {status_code}. "
                f"Details: {error_msg}"
            ),
        )
    else:
      return GoodmemFetchResponse(
          success=False,
          message=f"Unexpected error while fetching memories: {error_msg}",
      )


class GoodmemFetchTool(FunctionTool):
  """A tool that fetches memories from Goodmem storage.

  This tool wraps the goodmem_fetch function and provides semantic search
  capabilities to ADK agents.
  """

  def __init__(
      self,
      base_url: Optional[str] = None,
      api_key: Optional[str] = None,
      embedder_id: Optional[str] = None,
      top_k: int = 5,
      debug: bool = False,
  ):
    """Initialize the Goodmem fetch tool.

    Args:
      base_url: The base URL for the Goodmem API (required).
      api_key: The API key for authentication (required).
      embedder_id: Optional embedder ID to use when creating new spaces.
      top_k: Default number of memories to retrieve (default: 5, max: 20).
      debug: Enable debug logging.
    """
    self._base_url = base_url
    self._api_key = api_key
    self._embedder_id = embedder_id
    self._top_k = top_k
    self._debug = debug

    # Create a wrapper function that uses instance top_k as default
    # We need a wrapper because top_k needs to use self._top_k as default
    async def _wrapped_fetch(
        query: str,
        top_k: Optional[int] = None,
        tool_context: ToolContext = None,
    ) -> GoodmemFetchResponse:
      # Use instance top_k if not provided
      if top_k is None:
        top_k = self._top_k
      return await goodmem_fetch(
          query=query,
          top_k=top_k,
          tool_context=tool_context,
          base_url=self._base_url,
          api_key=self._api_key,
          embedder_id=self._embedder_id,
          debug=self._debug,
      )

    # Preserve function metadata for FunctionTool introspection
    # Copy signature from original function (excluding the config params)
    original_sig = inspect.signature(goodmem_fetch)
    params = []
    for name, param in original_sig.parameters.items():
      if name not in ("base_url", "api_key", "embedder_id", "debug"):
        # Update top_k default to use instance default
        if name == "top_k":
          params.append(param.replace(default=self._top_k))
        else:
          params.append(param)
    setattr(
        _wrapped_fetch,
        "__signature__",
        original_sig.replace(parameters=params),
    )
    _wrapped_fetch.__name__ = goodmem_fetch.__name__
    _wrapped_fetch.__doc__ = goodmem_fetch.__doc__

    super().__init__(_wrapped_fetch)


# ============================================================================
# Singleton instances (following Google ADK pattern)
# ============================================================================
# Note: These singleton instances require configuration to be passed when
# creating tool instances. See agent.py examples for usage.
