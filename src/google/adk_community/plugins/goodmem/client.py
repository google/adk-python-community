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

"""Goodmem API client for interacting with Goodmem.ai.

Lives under plugins/goodmem and is shared: used by GoodmemChatPlugin and
re-exported for use by tools (goodmem_save, goodmem_fetch). Uses httpx for
HTTP calls.
"""

import json
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx


class GoodmemClient:
  """Client for interacting with the Goodmem API.

  Attributes:
    _base_url: The base URL for the Goodmem API.
    _api_key: The API key for authentication.
    _headers: HTTP headers for API requests.
  """

  def __init__(self, base_url: str, api_key: str, debug: bool = False) -> None:
    """Initializes the Goodmem client.

    Args:
      base_url: The base URL for the Goodmem API, without the /v1 suffix
        (e.g., "https://api.goodmem.ai").
      api_key: The Goodmem API key for authentication.
      debug: Whether to enable debug mode.
    """
    self._base_url = base_url.rstrip("/")
    self._api_key = api_key
    self._headers = {"x-api-key": self._api_key}
    self._debug = debug
    self._client = httpx.Client(
        base_url=self._base_url,
        headers=self._headers,
        timeout=30.0,
    )

  def close(self) -> None:
    """Closes the underlying HTTP client."""
    self._client.close()

  def __enter__(self) -> "GoodmemClient":
    return self

  def __exit__(self, *args: Any) -> None:
    self.close()

  def _safe_json_dumps(self, value: Any) -> str:
    try:
      return json.dumps(value, indent=2)
    except (TypeError, ValueError):
      return f"<non-serializable: {type(value).__name__}>"

  def create_space(self, space_name: str, embedder_id: str) -> Dict[str, Any]:
    """Creates a new Goodmem space.

    Args:
      space_name: The name of the space to create.
      embedder_id: The embedder ID to use for the space.

    Returns:
      The response JSON containing spaceId.

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails (e.g. connection, timeout).
    """
    url = "/v1/spaces"
    payload = {
        "name": space_name,
        "spaceEmbedders": [
            {"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}
        ],
        "defaultChunkingConfig": {
            "recursive": {
                "chunkSize": 512,
                "chunkOverlap": 64,
                "keepStrategy": "KEEP_END",
                "lengthMeasurement": "CHARACTER_COUNT",
            }
        },
    }
    response = self._client.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()

  def insert_memory(
      self,
      space_id: str,
      content: str,
      content_type: str = "text/plain",
      metadata: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Inserts a text memory into a Goodmem space.

    Args:
      space_id: The ID of the space to insert into.
      content: The content of the memory.
      content_type: The content type (default: text/plain).
      metadata: Optional metadata dict (e.g., session_id, user_id).

    Returns:
      The response JSON containing memoryId and processingStatus.

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails.
    """
    url = "/v1/memories"
    payload: Dict[str, Any] = {
        "spaceId": space_id,
        "originalContent": content,
        "contentType": content_type,
    }
    if metadata:
      payload["metadata"] = metadata
    response = self._client.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()

  def insert_memory_binary(
      self,
      space_id: str,
      content_bytes: bytes,
      content_type: str,
      metadata: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    """Inserts a binary memory into a Goodmem space using multipart upload.

    Args:
      space_id: The ID of the space to insert into.
      content_bytes: The raw binary content as bytes.
      content_type: The MIME type (e.g., application/pdf, image/png).
      metadata: Optional metadata dict (e.g., session_id, user_id, filename).

    Returns:
      The response JSON containing memoryId and processingStatus.

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails.
    """
    url = "/v1/memories"

    if self._debug:
      print("[DEBUG] insert_memory_binary called:")
      print(f"  - space_id: {space_id}")
      print(f"  - content_type: {content_type}")
      print(f"  - content_bytes length: {len(content_bytes)} bytes")
      if metadata:
        print(f"  - metadata:\n{self._safe_json_dumps(metadata)}")

    request_data: Dict[str, Any] = {
        "spaceId": space_id,
        "contentType": content_type,
    }
    if metadata:
      request_data["metadata"] = metadata

    if self._debug:
      print(f"[DEBUG] request_data:\n{self._safe_json_dumps(request_data)}")

    data = {"request": json.dumps(request_data)}
    files = {"file": ("upload", content_bytes, content_type)}

    if self._debug:
      print(f"[DEBUG] Making POST request to {url}")
    response = self._client.post(
        url,
        data=data,
        files=files,
        timeout=120.0,
    )
    if self._debug:
      print(f"[DEBUG] Response status: {response.status_code}")

    response.raise_for_status()
    result = response.json()
    if self._debug:
      print(f"[DEBUG] Response:\n{self._safe_json_dumps(result)}")
    return result

  def retrieve_memories(
      self,
      query: str,
      space_ids: List[str],
      request_size: int = 5,
  ) -> List[Dict[str, Any]]:
    """Searches for chunks matching a query in given spaces.

    Args:
      query: The search query message.
      space_ids: List of space IDs to search in.
      request_size: The number of chunks to retrieve.

    Returns:
      List of matching chunks (parsed from NDJSON response).

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails.
    """
    url = "/v1/memories:retrieve"
    headers = {**self._headers, "Accept": "application/x-ndjson"}
    payload = {
        "message": query,
        "spaceKeys": [{"spaceId": sid} for sid in space_ids],
        "requestedSize": request_size,
    }

    response = self._client.post(
        url, json=payload, headers=headers, timeout=30.0
    )
    response.raise_for_status()

    chunks: List[Dict[str, Any]] = []
    for line in response.text.strip().split("\n"):
      if line.strip():
        try:
          tmp_dict = json.loads(line)
          if "retrievedItem" in tmp_dict:
            chunks.append(tmp_dict)
        except json.JSONDecodeError:
          continue
    return chunks

  def list_spaces(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists spaces, optionally filtering by name.

    Returns:
      List of spaces (optionally filtered by name).

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails.
    """
    url = "/v1/spaces"
    all_spaces: List[Dict[str, Any]] = []
    next_token: Optional[str] = None
    max_results = 1000

    while True:
      params: Dict[str, Any] = {"maxResults": max_results}
      if next_token:
        params["nextToken"] = next_token
      if name:
        params["nameFilter"] = name

      response = self._client.get(url, params=params, timeout=30.0)
      response.raise_for_status()

      data = response.json()
      spaces = data.get("spaces", [])
      all_spaces.extend(spaces)

      next_token = data.get("nextToken")
      if not next_token:
        break

    return all_spaces

  def list_embedders(self) -> List[Dict[str, Any]]:
    """Lists all embedders.

    Returns:
      List of embedders.

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails.
    """
    url = "/v1/embedders"
    response = self._client.get(url, timeout=30.0)
    response.raise_for_status()
    return response.json().get("embedders", [])

  def get_memory_by_id(self, memory_id: str) -> Dict[str, Any]:
    """Gets a memory by its ID.

    Args:
      memory_id: The ID of the memory to retrieve.

    Returns:
      The memory object including metadata, contentType, etc.

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails.
    """
    encoded_memory_id = quote(memory_id, safe="")
    url = f"/v1/memories/{encoded_memory_id}"
    response = self._client.get(url, timeout=30.0)
    response.raise_for_status()
    return response.json()

  def get_memories_batch(self, memory_ids: List[str]) -> List[Dict[str, Any]]:
    """Gets multiple memories by ID in a single request (batch get).

    Uses POST /v1/memories:batchGet to avoid N+1 queries when enriching
    many chunks with full memory metadata.

    Args:
      memory_ids: List of memory IDs to fetch.

    Returns:
      List of memory objects (same shape as get_memory_by_id). Order and
      presence may not match request; missing or failed IDs are omitted.

    Raises:
      httpx.HTTPStatusError: If the API request fails with an error status.
      httpx.RequestError: If the request fails.
    """
    if not memory_ids:
      return []
    url = "/v1/memories:batchGet"
    payload = {"memoryIds": list(memory_ids)}
    response = self._client.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    data = response.json()
    return data.get("memories", [])
