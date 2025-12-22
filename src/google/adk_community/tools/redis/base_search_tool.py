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

"""Base class for Redis search tools using RedisVL."""

from __future__ import annotations

from abc import abstractmethod
import asyncio
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from redisvl.index import AsyncSearchIndex
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import BaseVectorizer


class BaseRedisSearchTool(BaseTool):
  """Base class for Redis search tools using RedisVL.

  This class provides common functionality for all Redis search tools:
  - Index and vectorizer management
  - Common error handling
  - Standard response format

  Subclasses must implement `_build_query()` to create the appropriate
  RedisVL query object for their search type.
  """

  def __init__(
      self,
      *,
      name: str,
      description: str,
      index: Union[SearchIndex, AsyncSearchIndex],
      vectorizer: Optional[BaseVectorizer] = None,
      return_fields: Optional[List[str]] = None,
  ):
    """Initialize the base Redis search tool.

    Args:
        name: The name of the tool (exposed to LLM).
        description: The description of the tool (exposed to LLM).
        index: The RedisVL SearchIndex or AsyncSearchIndex to query.
        vectorizer: Optional vectorizer for embedding queries.
        return_fields: Optional list of fields to return in results.
    """
    super().__init__(name=name, description=description)
    self._index = index
    self._vectorizer = vectorizer
    self._return_fields = return_fields
    self._is_async_index = isinstance(index, AsyncSearchIndex)

  def _get_declaration(self) -> types.FunctionDeclaration:
    """Get the function declaration for the LLM.

    Returns a simple interface with just a query parameter.
    Subclasses can override to add additional parameters.
    """
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description="The search query text.",
                ),
            },
            required=["query"],
        ),
    )

  @abstractmethod
  def _build_query(
      self, query_text: str, embedding: List[float], **kwargs: Any
  ) -> Any:
    """Build the RedisVL query object.

    Args:
        query_text: The original query text from the user.
        embedding: The vector embedding of the query text.
        **kwargs: Additional parameters from the LLM call.

    Returns:
        A RedisVL query object (VectorQuery, HybridQuery, etc.)
    """
    pass

  async def run_async(
      self, *, args: Dict[str, Any], tool_context: ToolContext
  ) -> Dict[str, Any]:
    """Execute the search query.

    Args:
        args: Arguments from the LLM, must include 'query'.
        tool_context: The tool execution context.

    Returns:
        A dictionary with status, count, and results.
    """
    query_text = args.get("query", "")

    if not query_text:
      return {"status": "error", "error": "Query text is required."}

    try:
      # Embed the query text
      if self._vectorizer is None:
        return {
            "status": "error",
            "error": "Vectorizer is required for this search type.",
        }

      embedding = await self._vectorizer.aembed(query_text)

      # Build the query (subclass-specific)
      redisvl_query = self._build_query(query_text, embedding, **args)

      # Execute the query - handle both sync and async indexes
      if self._is_async_index:
        results = await self._index.query(redisvl_query)
      else:
        # Run sync query in thread pool to avoid blocking
        results = await asyncio.to_thread(self._index.query, redisvl_query)

      # Format results
      formatted_results = [dict(r) for r in results] if results else []

      return {
          "status": "success",
          "count": len(formatted_results),
          "results": formatted_results,
      }

    except Exception as e:
      return {"status": "error", "error": str(e)}
