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

"""Redis full-text search tool using BM25."""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from google.adk.tools.tool_context import ToolContext
from google.genai import types
from redisvl.index import AsyncSearchIndex
from redisvl.index import SearchIndex
from redisvl.query import TextQuery

from .base_search_tool import BaseRedisSearchTool
from .config import RedisTextQueryConfig


class RedisTextSearchTool(BaseRedisSearchTool):
  """Full-text search tool using BM25 scoring.

  This tool performs keyword-based full-text search using BM25 scoring.
  Unlike vector search, it doesn't require embeddings - it matches
  documents based on keyword relevance.

  Example:
      ```python
      from redisvl.index import SearchIndex
      from google.adk_community.tools.redis import (
          RedisTextSearchTool,
          RedisTextQueryConfig,
      )

      index = SearchIndex.from_yaml("schema.yaml")

      # Using config object (recommended)
      config = RedisTextQueryConfig(
          text_field_name="content",
          num_results=10,
          text_scorer="BM25STD",
      )
      tool = RedisTextSearchTool(
          index=index,
          config=config,
          return_fields=["title", "content"],
      )

      agent = Agent(model="gemini-2.5-flash", tools=[tool])
      ```
  """

  def __init__(
      self,
      *,
      index: Union[SearchIndex, AsyncSearchIndex],
      config: Optional[RedisTextQueryConfig] = None,
      return_fields: Optional[List[str]] = None,
      filter_expression: Optional[Any] = None,
      name: str = "redis_text_search",
      description: str = "Search for documents using keyword matching.",
  ):
    """Initialize the text search tool.

    Args:
        index: The RedisVL SearchIndex or AsyncSearchIndex to query.
        config: Configuration for text query parameters. If not provided,
            defaults will be used.
        return_fields: Optional list of fields to return in results.
        filter_expression: Optional filter expression to narrow results.
        name: The name of the tool (exposed to LLM).
        description: The description of the tool (exposed to LLM).
    """
    super().__init__(
        name=name,
        description=description,
        index=index,
        return_fields=return_fields,
    )
    self._config = config or RedisTextQueryConfig()
    self._filter_expression = filter_expression

  def _get_declaration(self) -> types.FunctionDeclaration:
    """Get the function declaration for the LLM."""
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
                "num_results": types.Schema(
                    type=types.Type.INTEGER,
                    description=(
                        "Number of results to return (default:"
                        f" {self._config.num_results})."
                    ),
                ),
            },
            required=["query"],
        ),
    )

  async def run_async(
      self, *, args: Dict[str, Any], tool_context: ToolContext
  ) -> Dict[str, Any]:
    """Execute the text search query.

    Args:
        args: Arguments from the LLM, must include 'query'.
        tool_context: The tool execution context.

    Returns:
        A dictionary with status, count, and results.
    """

    async def build_query_fn(
        query_text: str, args: Dict[str, Any]
    ) -> TextQuery:
      # Get query kwargs from config
      query_kwargs = self._config.to_query_kwargs(
          text=query_text,
          return_fields=self._return_fields,
          filter_expression=self._filter_expression,
      )
      # Allow LLM to override num_results
      if "num_results" in args:
        query_kwargs["num_results"] = args["num_results"]
      return TextQuery(**query_kwargs)

    return await self._run_search(args, build_query_fn)
