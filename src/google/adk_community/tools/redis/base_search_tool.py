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

"""Base classes for Redis search tools using RedisVL."""

from __future__ import annotations

from abc import abstractmethod
import asyncio
from typing import Any
from typing import Callable
from typing import Coroutine
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
  """Base class for ALL Redis search tools using RedisVL.

  This class provides common functionality shared by all Redis search tools:
  - Index management (sync and async)
  - Query execution with proper async handling
  - Standard response formatting
  - Error handling

  Subclasses should use `_run_search` to execute queries with consistent
  error handling and response formatting.
  """

  def __init__(
      self,
      *,
      name: str,
      description: str,
      index: Union[SearchIndex, AsyncSearchIndex],
      return_fields: Optional[List[str]] = None,
  ):
    """Initialize the base Redis search tool.

    Args:
        name: The name of the tool (exposed to LLM).
        description: The description of the tool (exposed to LLM).
        index: The RedisVL SearchIndex or AsyncSearchIndex to query.
        return_fields: Optional list of fields to return in results.
    """
    super().__init__(name=name, description=description)
    self._index = index
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

  async def _execute_query(self, query: Any) -> List[Dict[str, Any]]:
    """Execute a RedisVL query and return formatted results.

    Args:
        query: A RedisVL query object (VectorQuery, TextQuery, etc.)

    Returns:
        List of result dictionaries.
    """
    if self._is_async_index:
      results = await self._index.query(query)
    else:
      # Run sync query in thread pool to avoid blocking
      results = await asyncio.to_thread(self._index.query, query)

    return [dict(r) for r in results] if results else []

  async def _run_search(
      self,
      args: Dict[str, Any],
      build_query_fn: Callable[[str, Dict[str, Any]], Coroutine[Any, Any, Any]],
  ) -> Dict[str, Any]:
    """Execute a search with consistent validation, execution, and formatting.

    This is a template method that handles:
    - Query text validation
    - Query building (via the provided async function)
    - Query execution
    - Response formatting
    - Error handling

    Args:
        args: Arguments from the LLM, must include 'query'.
        build_query_fn: Async function that takes (query_text, args) and
            returns a RedisVL query object.

    Returns:
        A dictionary with status, count, and results (or error).
    """
    query_text = args.get("query", "")

    if not query_text:
      return {"status": "error", "error": "Query text is required."}

    try:
      # Build the query using the provided function
      redisvl_query = await build_query_fn(query_text, args)

      # Execute and format results
      results = await self._execute_query(redisvl_query)

      return {
          "status": "success",
          "count": len(results),
          "results": results,
      }

    except Exception as e:
      return {"status": "error", "error": str(e)}


class VectorizedSearchTool(BaseRedisSearchTool):
  """Base class for Redis search tools that require vector embeddings.

  This class extends BaseRedisSearchTool with:
  - Required vectorizer for embedding queries
  - Abstract _build_query method for subclasses to implement

  Use this as the base class for vector-based search tools like
  VectorSearchTool, HybridSearchTool, and RangeSearchTool.
  """

  def __init__(
      self,
      *,
      name: str,
      description: str,
      index: Union[SearchIndex, AsyncSearchIndex],
      vectorizer: BaseVectorizer,
      return_fields: Optional[List[str]] = None,
  ):
    """Initialize the vectorized search tool.

    Args:
        name: The name of the tool (exposed to LLM).
        description: The description of the tool (exposed to LLM).
        index: The RedisVL SearchIndex or AsyncSearchIndex to query.
        vectorizer: The vectorizer for embedding queries (required).
        return_fields: Optional list of fields to return in results.
    """
    super().__init__(
        name=name,
        description=description,
        index=index,
        return_fields=return_fields,
    )
    self._vectorizer = vectorizer

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
    """Execute the vector-based search query.

    Args:
        args: Arguments from the LLM, must include 'query'.
        tool_context: The tool execution context.

    Returns:
        A dictionary with status, count, and results.
    """

    async def build_query_fn(query_text: str, args: Dict[str, Any]) -> Any:
      embedding = await self._vectorizer.aembed(query_text)
      return self._build_query(query_text, embedding, **args)

    return await self._run_search(args, build_query_fn)
