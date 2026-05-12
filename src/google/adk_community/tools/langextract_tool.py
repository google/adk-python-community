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

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field
import logging
from typing import Any
from typing import Optional

from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from typing_extensions import override

try:
  import langextract as lx
except ImportError as e:
  raise ImportError(
      'LangExtract tools require pip install langextract.'
  ) from e

logger = logging.getLogger(__name__)


class LangExtractTool(BaseTool):
  """A tool that extracts structured information from text using LangExtract.

  This tool wraps the langextract library to enable LLM agents to extract
  structured data (entities, attributes, relationships) from unstructured
  text. The agent provides the text to extract from and a description of
  what to extract; other parameters are pre-configured at construction time.

  Args:
    name: The name of the tool. Defaults to 'langextract'.
    description: The description of the tool shown to the LLM.
    examples: Optional list of langextract ExampleData for few-shot
      extraction guidance.
    model_id: The model ID for langextract to use internally.
      Defaults to 'gemini-2.5-flash'.
    api_key: Optional API key for langextract. If None, uses the
      LANGEXTRACT_API_KEY environment variable.
    extraction_passes: Number of extraction passes. Defaults to 1.
    max_workers: Maximum worker threads for langextract. Defaults to 1.
    max_char_buffer: Maximum character buffer size for text chunking.
      Defaults to 4000.

  Examples::

      from google.adk_community.tools import LangExtractTool
      import langextract as lx

      tool = LangExtractTool(
          name='extract_entities',
          description='Extract named entities from text.',
          examples=[
              lx.data.ExampleData(
                  text='John is a software engineer at Google.',
                  extractions=[
                      lx.data.Extraction(
                          extraction_class='person',
                          extraction_text='John',
                          attributes={
                              'role': 'software engineer',
                              'company': 'Google',
                          },
                      )
                  ],
              )
          ],
      )
  """

  def __init__(
      self,
      *,
      name: str = 'langextract',
      description: str = (
          'Extracts structured information from unstructured'
          ' text. Provide the text and a description of what'
          ' to extract.'
      ),
      examples: Optional[list[lx.data.ExampleData]] = None,
      model_id: str = 'gemini-2.5-flash',
      api_key: Optional[str] = None,
      extraction_passes: int = 1,
      max_workers: int = 1,
      max_char_buffer: int = 4000,
  ):
    super().__init__(name=name, description=description)
    self._examples = examples or []
    self._model_id = model_id
    self._api_key = api_key
    self._extraction_passes = extraction_passes
    self._max_workers = max_workers
    self._max_char_buffer = max_char_buffer

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                'text': types.Schema(
                    type=types.Type.STRING,
                    description=(
                        'The unstructured text to extract information from.'
                    ),
                ),
                'prompt_description': types.Schema(
                    type=types.Type.STRING,
                    description=(
                        'A description of what kind of information to'
                        ' extract from the text.'
                    ),
                ),
            },
            required=['text', 'prompt_description'],
        ),
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    text = args.get('text')
    prompt_description = args.get('prompt_description')

    if not text:
      return {'error': 'The "text" parameter is required.'}
    if not prompt_description:
      return {'error': 'The "prompt_description" parameter is required.'}

    try:
      extract_kwargs: dict[str, Any] = {
          'text_or_documents': text,
          'prompt_description': prompt_description,
          'examples': self._examples,
          'model_id': self._model_id,
          'extraction_passes': self._extraction_passes,
          'max_workers': self._max_workers,
          'max_char_buffer': self._max_char_buffer,
      }
      if self._api_key is not None:
        extract_kwargs['api_key'] = self._api_key

      # lx.extract() is synchronous; run in a thread to avoid
      # blocking the event loop.
      result = await asyncio.to_thread(lx.extract, **extract_kwargs)

      extractions = []
      for extraction in result:
        entry = {
            'extraction_class': extraction.extraction_class,
            'extraction_text': extraction.extraction_text,
        }
        if extraction.attributes:
          entry['attributes'] = extraction.attributes
        extractions.append(entry)

      return {'extractions': extractions}

    except Exception as e:
      logger.error('LangExtract extraction failed: %s', e)
      return {'error': f'Extraction failed: {e}'}


@dataclass
class LangExtractToolConfig:
  """Configuration for LangExtractTool."""

  name: str = 'langextract'
  description: str = (
      'Extracts structured information from unstructured text.'
  )
  examples: list[lx.data.ExampleData] = field(default_factory=list)
  model_id: str = 'gemini-2.5-flash'
  api_key: Optional[str] = None
  extraction_passes: int = 1
  max_workers: int = 1
  max_char_buffer: int = 4000

  def build(self) -> LangExtractTool:
    """Instantiate a LangExtractTool from this config."""
    return LangExtractTool(
        name=self.name,
        description=self.description,
        examples=self.examples,
        model_id=self.model_id,
        api_key=self.api_key,
        extraction_passes=self.extraction_passes,
        max_workers=self.max_workers,
        max_char_buffer=self.max_char_buffer,
    )
