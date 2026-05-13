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

"""Amazon Bedrock model integration for Google ADK.

Provides native integration with Amazon Bedrock via the Converse API,
supporting all Bedrock-hosted models including Amazon Nova, Anthropic Claude,
Meta Llama, Mistral, Cohere, and more.
"""

from __future__ import annotations

import asyncio
from functools import cached_property
import json
import logging
import os
from typing import Any
from typing import AsyncGenerator
from typing import NoReturn
from typing import Optional
from typing import TYPE_CHECKING

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field
from pydantic import model_validator
from typing_extensions import override

from google.adk_community import version as _community_version

if TYPE_CHECKING:
  import boto3
  from botocore.exceptions import ClientError
  from google.adk.models.llm_request import LlmRequest

__all__ = ["BedrockModel"]

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_MAX_TOKENS = 4096
DEFAULT_REGION = "us-east-1"

# Bedrock ValidationException messages that signal context window overflow.
_CONTEXT_OVERFLOW_MESSAGES = (
    "Input is too long for requested model",
    "input length and `max_tokens` exceed context limit",
    "too many total text bytes",
    "prompt is too long",
)

# Bedrock stopReason -> ADK FinishReason
_FINISH_REASON_MAP: dict[str, types.FinishReason] = {
    "end_turn": types.FinishReason.STOP,
    "tool_use": types.FinishReason.STOP,
    "stop_sequence": types.FinishReason.STOP,
    "max_tokens": types.FinishReason.MAX_TOKENS,
    "guardrail_intervened": types.FinishReason.SAFETY,
}

# image/* MIME type -> Bedrock image format string
_IMAGE_FORMAT_MAP: dict[str, str] = {
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}


# ---------------------------------------------------------------------------
# ADK -> Bedrock conversion helpers
# ---------------------------------------------------------------------------


def _part_to_bedrock_block(part: types.Part) -> dict[str, Any] | None:
  """Convert a single ADK Part to a Bedrock content block.

  Args:
    part: An ADK types.Part object.

  Returns:
    A Bedrock-compatible content block dict, or None if the part type is
    not supported.
  """
  if part.text is not None:
    return {"text": part.text}

  if part.function_call:
    assert part.function_call.name
    return {
        "toolUse": {
            "toolUseId": part.function_call.id or "",
            "name": part.function_call.name,
            "input": part.function_call.args or {},
        }
    }

  if part.function_response:
    response_data = part.function_response.response
    # Normalise the tool result into a plain text content block.
    if isinstance(response_data, dict):
      if "result" in response_data:
        content_text = str(response_data["result"])
      elif "content" in response_data:
        items = response_data["content"]
        # Handle list-of-dicts content produced by some tool wrappers.
        if isinstance(items, list):
          texts = [
              item.get("text", str(item))
              if isinstance(item, dict)
              else str(item)
              for item in items
          ]
          content_text = "\n".join(texts)
        else:
          content_text = str(items)
      else:
        content_text = json.dumps(response_data)
    else:
      content_text = str(response_data)

    return {
        "toolResult": {
            "toolUseId": part.function_response.id or "",
            "content": [{"text": content_text}],
            "status": "success",
        }
    }

  if part.inline_data and part.inline_data.data and part.inline_data.mime_type:
    mime = part.inline_data.mime_type.lower().split(";")[0].strip()
    img_format = _IMAGE_FORMAT_MAP.get(mime)
    if img_format:
      return {
          "image": {
              "format": img_format,
              "source": {"bytes": part.inline_data.data},
          }
      }
    logger.warning(
        "mime_type=<%s> | unsupported inline_data MIME type, skipping", mime
    )
    return None

  logger.warning("Unsupported ADK Part type, skipping: %s", part)
  return None


def _content_to_bedrock_message(
    content: types.Content,
) -> dict[str, Any] | None:
  """Convert an ADK Content object to a Bedrock Converse API message.

  Args:
    content: An ADK types.Content object.

  Returns:
    A Bedrock message dict, or None if the resulting content list is empty.
  """
  role = "assistant" if content.role in ("model", "assistant") else "user"
  bedrock_content: list[dict[str, Any]] = []
  for part in content.parts or []:
    block = _part_to_bedrock_block(part)
    if block is not None:
      bedrock_content.append(block)
  if not bedrock_content:
    return None
  return {"role": role, "content": bedrock_content}


def _bedrock_block_to_part(block: dict[str, Any]) -> types.Part | None:
  """Convert a Bedrock response content block to an ADK Part.

  Args:
    block: A Bedrock content block dict from a converse response.

  Returns:
    An ADK types.Part, or None if the block type is not handled.
  """
  if "text" in block:
    return types.Part.from_text(text=block["text"])

  if "toolUse" in block:
    tool_use = block["toolUse"]
    part = types.Part.from_function_call(
        name=tool_use["name"],
        args=tool_use.get("input", {}),
    )
    part.function_call.id = tool_use.get("toolUseId", "")
    return part

  return None


def _function_declaration_to_tool_spec(
    func_decl: types.FunctionDeclaration,
) -> dict[str, Any]:
  """Convert an ADK FunctionDeclaration to a Bedrock toolSpec dict.

  Args:
    func_decl: An ADK types.FunctionDeclaration.

  Returns:
    A Bedrock toolSpec dict suitable for the toolConfig.tools list.
  """
  assert func_decl.name

  properties: dict[str, Any] = {}
  required_params: list[str] = []

  if func_decl.parameters_json_schema:
    input_schema = func_decl.parameters_json_schema
  else:
    if func_decl.parameters and func_decl.parameters.properties:
      for key, schema in func_decl.parameters.properties.items():
        prop = schema.model_dump(exclude_none=True)
        # Normalise type enum to a lowercase string.
        if "type" in prop:
          t = prop["type"]
          prop["type"] = (
              t.value.lower() if hasattr(t, "value") else str(t).lower()
          )
        properties[key] = prop
    if func_decl.parameters and func_decl.parameters.required:
      required_params = func_decl.parameters.required

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required_params:
      input_schema["required"] = required_params

  return {
      "toolSpec": {
          "name": func_decl.name,
          "description": func_decl.description or "",
          "inputSchema": {"json": input_schema},
      }
  }


# ---------------------------------------------------------------------------
# BedrockModel
# ---------------------------------------------------------------------------


class BedrockModel(BaseLlm):
  """Amazon Bedrock integration for Google ADK via the Converse API.

  Supports all models available on Amazon Bedrock including:
  - Amazon Nova  (``amazon.nova-pro-v1:0``, ``amazon.nova-lite-v1:0``, …)
  - Anthropic Claude (``anthropic.claude-3-5-sonnet-20241022-v2:0``, …)
  - Meta Llama  (``meta.llama3-70b-instruct-v1:0``, …)
  - Mistral AI, Cohere, AI21, DeepSeek, and more
  - Cross-region inference profiles (``us.*``, ``eu.*``, ``ap.*`` prefixes)

  Example usage::

    from google.adk.agents import Agent
    from google.adk_community.models.bedrock_model import BedrockModel

    agent = Agent(
        model=BedrockModel(model="us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        ...
    )

  AWS credentials are resolved via the standard boto3 credential chain:
  environment variables, ``~/.aws/credentials``, or an IAM instance/task role.

  To use a custom boto3 session (e.g. assumed role)::

    import boto3
    session = boto3.Session(profile_name="my-profile")
    model = BedrockModel(
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        boto_session=session,
    )

  Attributes:
    model: Bedrock model ID or cross-region inference profile ID.
    region_name: AWS region. Resolved from ``AWS_REGION`` / ``AWS_DEFAULT_REGION``
      environment variables, then falls back to ``us-east-1``.
    max_tokens: Maximum tokens to generate. Defaults to 4096.
    guardrail_id: Optional Bedrock Guardrail identifier.
    guardrail_version: Optional Bedrock Guardrail version (e.g. ``"1"`` or
      ``"DRAFT"``).
    boto_session: Optional pre-configured :class:`boto3.Session`. Takes
      precedence over ``region_name`` when both are supplied; however,
      supplying both raises a ``ValueError``.
  """

  model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
  region_name: Optional[str] = None
  max_tokens: int = DEFAULT_MAX_TOKENS
  guardrail_id: Optional[str] = None
  guardrail_version: Optional[str] = None
  # boto_session is excluded from pydantic serialisation because boto3.Session
  # is not a pydantic-serialisable type.
  boto_session: Optional[Any] = Field(default=None, exclude=True)

  @model_validator(mode="after")
  def _validate_boto_session_and_region(self) -> "BedrockModel":
    if self.boto_session is not None and self.region_name is not None:
      raise ValueError(
          "Cannot specify both `boto_session` and `region_name`. "
          "Pass `region_name` when constructing the boto3.Session instead."
      )
    return self

  @classmethod
  @override
  def supported_models(cls) -> list[str]:
    """Return regex patterns that match Bedrock model IDs.

    Covers:
    - Cross-region inference profiles: ``us.*``, ``eu.*``, ``ap.*``
    - Direct model IDs for all major providers available on Bedrock
    """
    return [
        # Cross-region inference profiles
        r"(us|eu|ap)\.(amazon|anthropic|meta|mistral|cohere|ai21|deepseek|writer)\..+",
        # Direct model IDs
        r"(amazon|anthropic|meta|mistral|cohere|ai21|deepseek|writer)\..+",
    ]

  @override
  async def generate_content_async(
      self,
      llm_request: "LlmRequest",
      stream: bool = False,
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generate content using the Amazon Bedrock Converse API.

    Args:
      llm_request: The ADK LlmRequest containing messages, tools, and config.
      stream: When ``True``, streams partial responses via ``converse_stream``.

    Yields:
      :class:`~google.adk.models.llm_response.LlmResponse` objects.
      In streaming mode multiple partial responses are yielded, followed by a
      final aggregated response with ``partial=False``.
    """
    request = self._build_request(llm_request)
    logger.debug(
        "model=<%s> | sending request to Bedrock",
        llm_request.model or self.model,
    )

    if stream:
      async for response in self._generate_streaming(request):
        yield response
    else:
      yield await self._generate_non_streaming(request)

  # ------------------------------------------------------------------
  # Request building
  # ------------------------------------------------------------------

  def _build_request(self, llm_request: "LlmRequest") -> dict[str, Any]:
    """Build a Bedrock Converse API request dict from an LlmRequest.

    Args:
      llm_request: The ADK LlmRequest.

    Returns:
      A dict ready to be unpacked into ``client.converse(**request)`` or
      ``client.converse_stream(**request)``.
    """
    # --- messages ---
    messages: list[dict[str, Any]] = []
    for content in llm_request.contents or []:
      msg = _content_to_bedrock_message(content)
      if msg:
        messages.append(msg)

    # --- system prompt ---
    system_blocks: list[dict[str, Any]] = []
    if llm_request.config and llm_request.config.system_instruction:
      system_blocks = [{"text": llm_request.config.system_instruction}]

    # --- tools ---
    tool_config: dict[str, Any] | None = None
    if (
        llm_request.config
        and llm_request.config.tools
        and llm_request.config.tools[0].function_declarations
    ):
      tools = [
          _function_declaration_to_tool_spec(fd)
          for fd in llm_request.config.tools[0].function_declarations
      ]
      tool_config = {"tools": tools, "toolChoice": {"auto": {}}}

    # --- inference config ---
    inference_config: dict[str, Any] = {"maxTokens": self.max_tokens}
    if llm_request.config:
      cfg = llm_request.config
      if cfg.temperature is not None:
        inference_config["temperature"] = cfg.temperature
      if cfg.top_p is not None:
        inference_config["topP"] = cfg.top_p
      if cfg.stop_sequences:
        inference_config["stopSequences"] = cfg.stop_sequences
      if cfg.max_output_tokens is not None:
        inference_config["maxTokens"] = cfg.max_output_tokens

    request: dict[str, Any] = {
        "modelId": llm_request.model or self.model,
        "messages": messages,
        "system": system_blocks,
        "inferenceConfig": inference_config,
    }
    if tool_config:
      request["toolConfig"] = tool_config
    if self.guardrail_id and self.guardrail_version:
      request["guardrailConfig"] = {
          "guardrailIdentifier": self.guardrail_id,
          "guardrailVersion": self.guardrail_version,
          "trace": "enabled",
      }

    return request

  # ------------------------------------------------------------------
  # Non-streaming
  # ------------------------------------------------------------------

  async def _generate_non_streaming(
      self, request: dict[str, Any]
  ) -> LlmResponse:
    """Call Bedrock ``converse`` and return a single LlmResponse.

    Args:
      request: A Bedrock Converse API request dict.

    Returns:
      An :class:`~google.adk.models.llm_response.LlmResponse`.

    Raises:
      botocore.exceptions.ClientError: Re-raised with model/region context
        appended to the exception message.
    """
    from botocore.exceptions import ClientError

    try:
      loop = asyncio.get_running_loop()
      response = await loop.run_in_executor(
          None, lambda: self._client.converse(**request)
      )
    except ClientError as e:
      self._handle_client_error(e)
    return self._parse_converse_response(response)

  def _parse_converse_response(self, response: dict[str, Any]) -> LlmResponse:
    """Convert a Bedrock ``converse`` response dict to an LlmResponse.

    Args:
      response: The raw response dict returned by ``client.converse()``.

    Returns:
      An :class:`~google.adk.models.llm_response.LlmResponse`.
    """
    message = response["output"]["message"]
    parts: list[types.Part] = []
    for block in message.get("content", []):
      part = _bedrock_block_to_part(block)
      if part:
        parts.append(part)

    stop_reason = response.get("stopReason", "end_turn")
    finish_reason = _FINISH_REASON_MAP.get(stop_reason, types.FinishReason.STOP)

    usage_metadata = None
    if "usage" in response:
      usage = response["usage"]
      usage_metadata = types.GenerateContentResponseUsageMetadata(
          prompt_token_count=usage.get("inputTokens", 0),
          candidates_token_count=usage.get("outputTokens", 0),
          total_token_count=usage.get("totalTokens", 0),
      )

    return LlmResponse(
        content=types.Content(role="model", parts=parts),
        finish_reason=finish_reason,
        usage_metadata=usage_metadata,
    )

  # ------------------------------------------------------------------
  # Streaming
  # ------------------------------------------------------------------

  async def _generate_streaming(
      self, request: dict[str, Any]
  ) -> AsyncGenerator[LlmResponse, None]:
    """Call Bedrock ``converse_stream`` and yield partial + final LlmResponse.

    The synchronous boto3 streaming call is offloaded to a thread; events are
    forwarded to the async caller via a queue, following the same thread/queue
    bridge pattern used by the Strands Agents SDK.

    Args:
      request: A Bedrock Converse API request dict.

    Yields:
      Partial :class:`~google.adk.models.llm_response.LlmResponse` objects
      (``partial=True``) for each text delta, followed by a single final
      response (``partial=False``) with all accumulated content.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None | Exception] = asyncio.Queue()

    def _stream_in_thread() -> None:
      from botocore.exceptions import ClientError

      try:
        response = self._client.converse_stream(**request)
        for chunk in response["stream"]:
          loop.call_soon_threadsafe(queue.put_nowait, chunk)
      except ClientError as e:
        self._handle_client_error(e)
        raise
      except Exception as e:
        loop.call_soon_threadsafe(queue.put_nowait, e)
      finally:
        loop.call_soon_threadsafe(queue.put_nowait, None)

    task = asyncio.create_task(asyncio.to_thread(_stream_in_thread))

    # --- accumulation state ---
    text_buffer = ""
    # block_index -> {"id": str, "name": str, "args": str}
    function_calls: dict[int, dict[str, Any]] = {}
    finish_reason: types.FinishReason | None = None
    usage_metadata: types.GenerateContentResponseUsageMetadata | None = None

    while True:
      chunk = await queue.get()
      if chunk is None:
        break
      if isinstance(chunk, Exception):
        raise chunk

      # messageStart — carries role, not needed for output.
      if "messageStart" in chunk:
        pass

      # contentBlockStart — toolUse blocks announce name/id here.
      elif "contentBlockStart" in chunk:
        cbs = chunk["contentBlockStart"]
        block_index: int = cbs.get("contentBlockIndex", 0)
        start = cbs.get("start", {})
        if "toolUse" in start:
          function_calls[block_index] = {
              "id": start["toolUse"]["toolUseId"],
              "name": start["toolUse"]["name"],
              "args": "",
          }

      # contentBlockDelta — text or toolUse input fragments.
      elif "contentBlockDelta" in chunk:
        cbd = chunk["contentBlockDelta"]
        block_index = cbd.get("contentBlockIndex", 0)
        delta = cbd.get("delta", {})

        if "text" in delta:
          text = delta["text"]
          text_buffer += text
          yield LlmResponse(
              content=types.Content(
                  role="model",
                  parts=[types.Part.from_text(text=text)],
              ),
              partial=True,
          )

        elif "toolUse" in delta:
          if block_index in function_calls:
            function_calls[block_index]["args"] += delta["toolUse"].get(
                "input", ""
            )

      # contentBlockStop — end of a content block; no action needed here.
      elif "contentBlockStop" in chunk:
        pass

      # messageStop — carries the final stop reason.
      elif "messageStop" in chunk:
        stop_reason = chunk["messageStop"].get("stopReason", "end_turn")
        finish_reason = _FINISH_REASON_MAP.get(
            stop_reason, types.FinishReason.STOP
        )

      # metadata — usage token counts.
      elif "metadata" in chunk:
        meta = chunk["metadata"]
        if "usage" in meta:
          usage = meta["usage"]
          usage_metadata = types.GenerateContentResponseUsageMetadata(
              prompt_token_count=usage.get("inputTokens", 0),
              candidates_token_count=usage.get("outputTokens", 0),
              total_token_count=usage.get("totalTokens", 0),
          )

    await task

    # --- build final aggregated response ---
    parts: list[types.Part] = []
    if text_buffer:
      parts.append(types.Part.from_text(text=text_buffer))

    for _, fc in sorted(function_calls.items()):
      try:
        args = json.loads(fc["args"]) if fc["args"] else {}
      except json.JSONDecodeError:
        logger.warning(
            "tool_name=<%s> | failed to parse tool input JSON, using empty"
            " dict",
            fc["name"],
        )
        args = {}
      part = types.Part.from_function_call(name=fc["name"], args=args)
      part.function_call.id = fc["id"]
      parts.append(part)

    yield LlmResponse(
        content=types.Content(role="model", parts=parts),
        partial=False,
        finish_reason=finish_reason,
        usage_metadata=usage_metadata,
    )

  # ------------------------------------------------------------------
  # Error handling
  # ------------------------------------------------------------------

  def _handle_client_error(self, error: "ClientError") -> NoReturn:
    """Enrich and re-raise a botocore ClientError with diagnostic context.

    Logs the model ID and region, then re-raises the original exception so
    callers can inspect ``error.response["Error"]["Code"]`` as usual.

    Common error codes surfaced here:

    * ``ThrottlingException`` — the Bedrock service is rate-limiting the
      request; callers should implement back-off / retry logic.
    * ``ValidationException`` with a context-overflow message — the combined
      prompt and ``max_tokens`` exceeds the model's context window.
    * ``AccessDeniedException`` — the IAM principal does not have access to
      the requested model; check model-access settings in the Bedrock console.

    Args:
      error: The :class:`botocore.exceptions.ClientError` to handle.

    Raises:
      botocore.exceptions.ClientError: Always re-raises *error*.
    """
    code = error.response["Error"]["Code"]
    message = str(error)
    region = getattr(
        getattr(self._client, "meta", None), "region_name", "unknown"
    )

    logger.error(
        "model=<%s> region=<%s> error_code=<%s> | Bedrock ClientError: %s",
        self.model,
        region,
        code,
        message,
    )

    if code in ("ThrottlingException", "throttlingException"):
      logger.warning(
          "model=<%s> | request throttled; consider adding retry logic",
          self.model,
      )

    if code == "ValidationException" and any(
        msg in message for msg in _CONTEXT_OVERFLOW_MESSAGES
    ):
      logger.warning(
          "model=<%s> | context window overflow — reduce prompt length or"
          " max_tokens",
          self.model,
      )

    if code == "AccessDeniedException" and "model" in message.lower():
      logger.warning(
          "model=<%s> | access denied — enable model access at "
          "https://console.aws.amazon.com/bedrock/home#/modelaccess",
          self.model,
      )

    raise error

  # ------------------------------------------------------------------
  # boto3 client
  # ------------------------------------------------------------------

  @cached_property
  def _client(self) -> Any:
    """Create and cache a boto3 ``bedrock-runtime`` client.

    Resolves the AWS region in priority order:
    1. ``region_name`` attribute
    2. Region from ``boto_session`` (if provided)
    3. ``AWS_REGION`` environment variable
    4. ``AWS_DEFAULT_REGION`` environment variable
    5. Hard-coded fallback ``us-east-1``

    Returns:
      A boto3 ``bedrock-runtime`` client.

    Raises:
      ImportError: If ``boto3`` is not installed.
    """
    try:
      import boto3
    except ImportError as e:
      raise ImportError(
          "BedrockModel requires the boto3 package.\n"
          "Install it with: pip install google-adk-community[bedrock]\n"
          "Or: pip install boto3"
      ) from e

    from botocore.config import Config as BotocoreConfig

    session: boto3.Session = self.boto_session or boto3.Session()
    region = (
        self.region_name
        or session.region_name
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or DEFAULT_REGION
    )
    user_agent = f"google-adk-community/{_community_version.__version__}"
    client_config = BotocoreConfig(user_agent_extra=user_agent)
    logger.debug("region=<%s> | creating bedrock-runtime client", region)
    return session.client(
        "bedrock-runtime", region_name=region, config=client_config
    )
