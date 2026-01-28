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

"""Goodmem API client for interacting with Goodmem.ai."""

import json
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests


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
    """
    # Remove trailing slash if present to avoid double slashes in URLs
    self._base_url = base_url.rstrip("/")
    self._api_key = api_key
    self._headers = {
        "x-api-key": self._api_key,
        "Content-Type": "application/json"
    }
    self._debug = debug

  def _safe_json_dumps(self, value: Any) -> str:
    try:
      return json.dumps(value, indent=2)
    except (TypeError, ValueError):
      return "<non-serializable>"

  def create_space(self, space_name: str, embedder_id: str) -> Dict[str, Any]:
    """Creates a new Goodmem space.

    Args:
      space_name: The name of the space to create.
      embedder_id: The embedder ID to use for the space.

    Returns:
      The response JSON containing spaceId.

    Raises:
      requests.exceptions.RequestException: If the API request fails.
    """
    url = f"{self._base_url}/v1/spaces"
    payload = {
        "name": space_name,
        "spaceEmbedders": [
            {
                "embedderId": embedder_id,
                "defaultRetrievalWeight": 1.0
            }
        ],
        "defaultChunkingConfig": {
            "recursive": {
                "chunkSize": 512,
                "chunkOverlap": 64,
                "keepStrategy": "KEEP_END",
                "lengthMeasurement": "CHARACTER_COUNT"
            }
        }
    }
    response = requests.post(url, json=payload, headers=self._headers, timeout=30)
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
      requests.exceptions.RequestException: If the API request fails.
    """
    url = f"{self._base_url}/v1/memories"
    payload: Dict[str, Any] = {
        "spaceId": space_id,
        "originalContent": content,
        "contentType": content_type
    }
    if metadata:
      payload["metadata"] = metadata
    response = requests.post(url, json=payload, headers=self._headers, timeout=30)
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

    If debug is enabled, this method prints debug information to stdout.

    Args:
      space_id: The ID of the space to insert into.
      content_bytes: The raw binary content as bytes.
      content_type: The MIME type (e.g., application/pdf, image/png).
      metadata: Optional metadata dict (e.g., session_id, user_id, filename).

    Returns:
      The response JSON containing memoryId and processingStatus.

    Raises:
      requests.exceptions.RequestException: If the API request fails.
    """
    url = f"{self._base_url}/v1/memories"

    if self._debug:
      print("[DEBUG] insert_memory_binary called:")
      print(f"  - space_id: {space_id}")
      print(f"  - content_type: {content_type}")
      print(f"  - content_bytes length: {len(content_bytes)} bytes")
      if metadata:
        print(f"  - metadata:\n{self._safe_json_dumps(metadata)}")

    # Build the JSON request metadata
    request_data: Dict[str, Any] = {
        "spaceId": space_id,
        "contentType": content_type
    }
    if metadata:
      request_data["metadata"] = metadata

    if self._debug:
      print(f"[DEBUG] request_data:\n{self._safe_json_dumps(request_data)}")

    # Multipart form data: 'request' as form field, 'file' as file upload
    data = {"request": json.dumps(request_data)}
    files = {"file": ("upload", content_bytes, content_type)}

    # Use only API key header; requests will set Content-Type for multipart
    headers = {"x-api-key": self._api_key}

    if self._debug:
      print(f"[DEBUG] Making POST request to {url}")
    response = requests.post(
        url, data=data, files=files, headers=headers, timeout=120
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
      requests.exceptions.RequestException: If the API request fails.
    """
    url = f"{self._base_url}/v1/memories:retrieve"
    headers = self._headers.copy()
    headers["Accept"] = "application/x-ndjson"

    payload = {
        "message": query,
        "spaceKeys": [{"spaceId": space_id} for space_id in space_ids],
        "requestedSize": request_size
    }

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()

    chunks = []
    for line in response.text.strip().split("\n"):
      if line.strip():  # Skip blank/empty lines
        try:
          tmp_dict = json.loads(line)
          if "retrievedItem" in tmp_dict:
            chunks.append(tmp_dict)
        except json.JSONDecodeError:
          # Skip malformed lines (e.g., transmission errors)
          continue
    return chunks

  def list_spaces(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists spaces, optionally filtering by name.

    Returns:
      List of spaces (optionally filtered by name).

    Raises:
      requests.exceptions.RequestException: If the API request fails.
    """
    url = f"{self._base_url}/v1/spaces"

    all_spaces = []
    next_token = None
    max_results = 1000

    while True:
      # Build query parameters
      params = {"maxResults": max_results}
      if next_token:
        params["nextToken"] = next_token
      if name:
        params["nameFilter"] = name

      response = requests.get(
          url, headers=self._headers, params=params, timeout=30
      )
      response.raise_for_status()

      data = response.json()
      spaces = data.get("spaces", [])
      all_spaces.extend(spaces)

      # Check for next page
      next_token = data.get("nextToken")
      if not next_token:
        break

    return all_spaces

  def list_embedders(self) -> List[Dict[str, Any]]:
    """Lists all embedders.

    Returns:
      List of embedders.

    Raises:
      requests.exceptions.RequestException: If the API request fails.
    """
    url = f"{self._base_url}/v1/embedders"
    response = requests.get(url, headers=self._headers, timeout=30)
    response.raise_for_status()
    return response.json().get("embedders", [])

  def get_memory_by_id(self, memory_id: str) -> Dict[str, Any]:
    """Gets a memory by its ID.

    Args:
      memory_id: The ID of the memory to retrieve.

    Returns:
      The memory object including metadata, contentType, etc.

    Raises:
      requests.exceptions.RequestException: If the API request fails.
    """
    # URL-encode the memory_id to handle special characters
    encoded_memory_id = quote(memory_id, safe="")
    url = f"{self._base_url}/v1/memories/{encoded_memory_id}"
    response = requests.get(url, headers=self._headers, timeout=30)
    response.raise_for_status()
    return response.json()
