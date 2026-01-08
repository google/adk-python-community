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
from typing import Set
from typing import Tuple
from typing import Union

from google.adk.tools.tool_context import ToolContext
from google.genai import types
from redisvl.index import AsyncSearchIndex
from redisvl.index import SearchIndex
from redisvl.query import TextQuery

from .base_search_tool import BaseRedisSearchTool

# Type alias for sort specification
SortSpec = Optional[
    Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]]
]


class RedisTextSearchTool(BaseRedisSearchTool):
  """Full-text search tool using BM25 scoring.

  This tool performs keyword-based full-text search using BM25 scoring.
  Unlike vector search, it doesn't require embeddings - it matches
  documents based on keyword relevance.

  Example:
      ```python
      from redisvl.index import SearchIndex
      from google.adk_community.tools.redis import RedisTextSearchTool

      index = SearchIndex.from_yaml("schema.yaml")

      tool = RedisTextSearchTool(
          index=index,
          text_field_name="content",
          num_results=10,
          return_fields=["title", "content"],
      )

      agent = Agent(model="gemini-2.5-flash", tools=[tool])
      ```
  """

  def __init__(
      self,
      *,
      index: Union[SearchIndex, AsyncSearchIndex],
      text_field_name: str = "content",
      text_scorer: str = "BM25STD",
      num_results: int = 10,
      return_fields: Optional[List[str]] = None,
      filter_expression: Optional[Any] = None,
      return_score: bool = True,
      dialect: int = 2,
      sort_by: SortSpec = None,
      in_order: bool = False,
      stopwords: Optional[Union[str, Set[str]]] = "english",
      name: str = "redis_text_search",
      description: str = "Search for documents using keyword matching.",
  ):
    """Initialize the text search tool.

    Args:
        index: The RedisVL SearchIndex or AsyncSearchIndex to query.
        text_field_name: The name of the text field to search.
        text_scorer: The text scoring algorithm (default: "BM25STD").
        num_results: Default number of results to return (default: 10).
        return_fields: Optional list of fields to return in results.
        filter_expression: Optional filter expression to narrow results.
        return_score: Whether to return the text score (default: True).
        dialect: The RediSearch query dialect (default: 2).
        sort_by: Field(s) to order results by.
        in_order: Require query terms in same order (default: False).
        stopwords: Stopwords to remove from query (default: "english").
        name: The name of the tool (exposed to LLM).
        description: The description of the tool (exposed to LLM).
    """
    super().__init__(
        name=name,
        description=description,
        index=index,
        return_fields=return_fields,
    )
    self._text_field_name = text_field_name
    self._text_scorer = text_scorer
    self._num_results = num_results
    self._filter_expression = filter_expression
    self._return_score = return_score
    self._dialect = dialect
    self._sort_by = sort_by
    self._in_order = in_order
    self._stopwords = stopwords

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
                        f" {self._num_results})."
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
      num_results = args.get("num_results", self._num_results)
      return TextQuery(
          text=query_text,
          text_field_name=self._text_field_name,
          text_scorer=self._text_scorer,
          filter_expression=self._filter_expression,
          return_fields=self._return_fields,
          num_results=num_results,
          return_score=self._return_score,
          dialect=self._dialect,
          sort_by=self._sort_by,
          in_order=self._in_order,
          stopwords=self._stopwords,
      )

    return await self._run_search(args, build_query_fn)
