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

"""Unit tests for BedrockModel."""

import json
import os
import sys
from unittest.mock import ANY
from unittest.mock import MagicMock
from unittest.mock import patch

from google.genai import types
import pytest

from google.adk_community.models.bedrock_model import _bedrock_block_to_part
from google.adk_community.models.bedrock_model import _content_to_bedrock_message
from google.adk_community.models.bedrock_model import _function_declaration_to_tool_spec
from google.adk_community.models.bedrock_model import _part_to_bedrock_block
from google.adk_community.models.bedrock_model import BedrockModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_llm_request(
    contents=None,
    system_instruction=None,
    function_declarations=None,
    temperature=None,
    max_output_tokens=None,
    top_p=None,
    stop_sequences=None,
    model=None,
):
  """Build a minimal LlmRequest-like mock."""
  config = MagicMock()
  config.system_instruction = system_instruction
  config.temperature = temperature
  config.top_p = top_p
  config.stop_sequences = stop_sequences
  config.max_output_tokens = max_output_tokens
  config.response_schema = None

  if function_declarations:
    tool = MagicMock()
    tool.function_declarations = function_declarations
    config.tools = [tool]
  else:
    config.tools = []

  req = MagicMock()
  req.contents = contents or []
  req.config = config
  req.model = model
  return req


def _make_model(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", **kw):
  """Return a BedrockModel with a mocked boto3 client."""
  m = BedrockModel(model=model_id, **kw)
  return m


def _mock_client(model: BedrockModel) -> MagicMock:
  """Patch the cached _client property and return the mock."""
  client = MagicMock()
  # Override the cached_property by writing directly to __dict__
  model.__dict__["_client"] = client
  return client


# ---------------------------------------------------------------------------
# supported_models
# ---------------------------------------------------------------------------


class TestSupportedModels:

  def test_cross_region_inference_profile(self):
    import re

    patterns = BedrockModel.supported_models()
    model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert any(re.fullmatch(p, model_id) for p in patterns)

  def test_direct_model_id(self):
    import re

    patterns = BedrockModel.supported_models()
    model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert any(re.fullmatch(p, model_id) for p in patterns)

  def test_amazon_nova(self):
    import re

    patterns = BedrockModel.supported_models()
    assert any(re.fullmatch(p, "amazon.nova-pro-v1:0") for p in patterns)

  def test_meta_llama(self):
    import re

    patterns = BedrockModel.supported_models()
    assert any(
        re.fullmatch(p, "meta.llama3-70b-instruct-v1:0") for p in patterns
    )

  def test_eu_inference_profile(self):
    import re

    patterns = BedrockModel.supported_models()
    assert any(
        re.fullmatch(p, "eu.anthropic.claude-3-5-sonnet-20241022-v2:0")
        for p in patterns
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:

  def test_boto_session_and_region_raises(self):
    mock_session = MagicMock()
    with pytest.raises(ValueError, match="Cannot specify both"):
      BedrockModel(
          model="amazon.nova-pro-v1:0",
          boto_session=mock_session,
          region_name="us-east-1",
      )


# ---------------------------------------------------------------------------
# ADK -> Bedrock conversion helpers
# ---------------------------------------------------------------------------


class TestPartToBedrockBlock:

  def test_text_part(self):
    part = types.Part.from_text(text="hello")
    block = _part_to_bedrock_block(part)
    assert block == {"text": "hello"}

  def test_function_call_part(self):
    part = types.Part.from_function_call(name="my_tool", args={"key": "val"})
    part.function_call.id = "tool-123"
    block = _part_to_bedrock_block(part)
    assert block == {
        "toolUse": {
            "toolUseId": "tool-123",
            "name": "my_tool",
            "input": {"key": "val"},
        }
    }

  def test_function_response_with_result(self):
    part = types.Part.from_function_response(
        name="my_tool", response={"result": "42"}
    )
    part.function_response.id = "tool-123"
    block = _part_to_bedrock_block(part)
    assert block is not None
    assert block["toolResult"]["toolUseId"] == "tool-123"
    assert block["toolResult"]["status"] == "success"
    assert block["toolResult"]["content"][0]["text"] == "42"

  def test_function_response_with_content_list(self):
    part = types.Part.from_function_response(
        name="my_tool",
        response={"content": [{"text": "line1"}, {"text": "line2"}]},
    )
    part.function_response.id = "tool-456"
    block = _part_to_bedrock_block(part)
    assert "line1" in block["toolResult"]["content"][0]["text"]
    assert "line2" in block["toolResult"]["content"][0]["text"]

  def test_image_part_jpeg(self):
    part = types.Part(
        inline_data=types.Blob(mime_type="image/jpeg", data=b"\xff\xd8")
    )
    block = _part_to_bedrock_block(part)
    assert block is not None
    assert block["image"]["format"] == "jpeg"
    assert block["image"]["source"]["bytes"] == b"\xff\xd8"

  def test_image_part_png(self):
    part = types.Part(
        inline_data=types.Blob(mime_type="image/png", data=b"\x89PNG")
    )
    block = _part_to_bedrock_block(part)
    assert block["image"]["format"] == "png"

  def test_unsupported_mime_returns_none(self):
    part = types.Part(
        inline_data=types.Blob(mime_type="audio/mp3", data=b"\x00\x01")
    )
    block = _part_to_bedrock_block(part)
    assert block is None


class TestContentToBedrockMessage:

  def test_user_role(self):
    content = types.Content(
        role="user", parts=[types.Part.from_text(text="hi")]
    )
    msg = _content_to_bedrock_message(content)
    assert msg["role"] == "user"
    assert msg["content"] == [{"text": "hi"}]

  def test_model_role_mapped_to_assistant(self):
    content = types.Content(
        role="model", parts=[types.Part.from_text(text="reply")]
    )
    msg = _content_to_bedrock_message(content)
    assert msg["role"] == "assistant"

  def test_empty_parts_returns_none(self):
    content = types.Content(role="user", parts=[])
    msg = _content_to_bedrock_message(content)
    assert msg is None


class TestBedrockBlockToADKPart:

  def test_text_block(self):
    part = _bedrock_block_to_part({"text": "hello"})
    assert part is not None
    assert part.text == "hello"

  def test_tool_use_block(self):
    block = {
        "toolUse": {
            "toolUseId": "t1",
            "name": "search",
            "input": {"query": "aws"},
        }
    }
    part = _bedrock_block_to_part(block)
    assert part is not None
    assert part.function_call.name == "search"
    assert part.function_call.args == {"query": "aws"}
    assert part.function_call.id == "t1"

  def test_unknown_block_returns_none(self):
    part = _bedrock_block_to_part({"unknownField": "value"})
    assert part is None


class TestFunctionDeclarationToToolSpec:

  def test_basic_declaration(self):
    fd = types.FunctionDeclaration(
        name="get_weather",
        description="Get the weather",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "location": types.Schema(
                    type=types.Type.STRING, description="City name"
                )
            },
            required=["location"],
        ),
    )
    spec = _function_declaration_to_tool_spec(fd)
    assert spec["toolSpec"]["name"] == "get_weather"
    assert spec["toolSpec"]["description"] == "Get the weather"
    schema = spec["toolSpec"]["inputSchema"]["json"]
    assert schema["type"] == "object"
    assert "location" in schema["properties"]
    assert schema["required"] == ["location"]

  def test_no_parameters(self):
    fd = types.FunctionDeclaration(name="ping", description="Ping the service")
    spec = _function_declaration_to_tool_spec(fd)
    assert spec["toolSpec"]["name"] == "ping"
    assert spec["toolSpec"]["inputSchema"]["json"]["type"] == "object"


# ---------------------------------------------------------------------------
# _build_request
# ---------------------------------------------------------------------------


class TestBuildRequest:

  def test_basic_text_request(self):
    model = _make_model()
    req = _make_llm_request(
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text="hi")])
        ],
        system_instruction="You are helpful.",
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    payload = model._build_request(req)
    assert payload["modelId"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert payload["messages"][0]["role"] == "user"
    assert payload["system"] == [{"text": "You are helpful."}]
    assert payload["inferenceConfig"]["maxTokens"] == 4096

  def test_inference_config_overrides(self):
    model = _make_model()
    req = _make_llm_request(
        temperature=0.5,
        top_p=0.9,
        max_output_tokens=512,
        stop_sequences=["STOP"],
    )
    payload = model._build_request(req)
    cfg = payload["inferenceConfig"]
    assert cfg["temperature"] == 0.5
    assert cfg["topP"] == 0.9
    assert cfg["maxTokens"] == 512
    assert cfg["stopSequences"] == ["STOP"]

  def test_tool_config_included(self):
    fd = types.FunctionDeclaration(name="echo", description="Echo input")
    model = _make_model()
    req = _make_llm_request(function_declarations=[fd])
    payload = model._build_request(req)
    assert "toolConfig" in payload
    tools = payload["toolConfig"]["tools"]
    assert tools[0]["toolSpec"]["name"] == "echo"

  def test_guardrail_config_included(self):
    model = _make_model(guardrail_id="abc123", guardrail_version="1")
    req = _make_llm_request()
    payload = model._build_request(req)
    assert payload["guardrailConfig"]["guardrailIdentifier"] == "abc123"
    assert payload["guardrailConfig"]["guardrailVersion"] == "1"

  def test_no_guardrail_when_not_set(self):
    model = _make_model()
    req = _make_llm_request()
    payload = model._build_request(req)
    assert "guardrailConfig" not in payload

  def test_model_id_from_request_overrides_default(self):
    model = _make_model(model_id="amazon.nova-lite-v1:0")
    req = _make_llm_request(model="amazon.nova-pro-v1:0")
    payload = model._build_request(req)
    assert payload["modelId"] == "amazon.nova-pro-v1:0"


# ---------------------------------------------------------------------------
# Non-streaming generation
# ---------------------------------------------------------------------------


class TestGenerateNonStreaming:

  @pytest.mark.asyncio
  async def test_text_response(self):
    model = _make_model()
    client = _mock_client(model)
    client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hello, world!"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }

    req = _make_llm_request(
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text="hi")])
        ]
    )
    responses = []
    async for r in model.generate_content_async(req, stream=False):
      responses.append(r)

    assert len(responses) == 1
    resp = responses[0]
    assert resp.content.parts[0].text == "Hello, world!"
    assert resp.usage_metadata.total_token_count == 15

  @pytest.mark.asyncio
  async def test_tool_use_response(self):
    model = _make_model()
    client = _mock_client(model)
    client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{
                    "toolUse": {
                        "toolUseId": "tool-1",
                        "name": "get_weather",
                        "input": {"location": "Seattle"},
                    }
                }],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 20, "outputTokens": 10, "totalTokens": 30},
    }

    req = _make_llm_request()
    responses = []
    async for r in model.generate_content_async(req, stream=False):
      responses.append(r)

    resp = responses[0]
    fc = resp.content.parts[0].function_call
    assert fc.name == "get_weather"
    assert fc.args == {"location": "Seattle"}
    assert fc.id == "tool-1"

  @pytest.mark.asyncio
  async def test_max_tokens_finish_reason(self):
    model = _make_model()
    client = _mock_client(model)
    client.converse.return_value = {
        "output": {
            "message": {"role": "assistant", "content": [{"text": "truncated"}]}
        },
        "stopReason": "max_tokens",
        "usage": {"inputTokens": 5, "outputTokens": 4096, "totalTokens": 4101},
    }

    req = _make_llm_request()
    responses = []
    async for r in model.generate_content_async(req, stream=False):
      responses.append(r)

    from google.genai import types as gtypes

    assert responses[0].finish_reason == gtypes.FinishReason.MAX_TOKENS


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------


class TestGenerateStreaming:

  def _build_stream_chunks(self, text="Hello!", tool_use=None):
    """Helper to build a realistic list of Bedrock stream chunks."""
    chunks = [{"messageStart": {"role": "assistant"}}]
    if text:
      chunks += [
          {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
          {
              "contentBlockDelta": {
                  "contentBlockIndex": 0,
                  "delta": {"text": text},
              }
          },
          {"contentBlockStop": {"contentBlockIndex": 0}},
      ]
    if tool_use:
      chunks += [
          {
              "contentBlockStart": {
                  "contentBlockIndex": 1,
                  "start": {
                      "toolUse": {
                          "toolUseId": tool_use["id"],
                          "name": tool_use["name"],
                      }
                  },
              }
          },
          {
              "contentBlockDelta": {
                  "contentBlockIndex": 1,
                  "delta": {
                      "toolUse": {"input": json.dumps(tool_use["input"])}
                  },
              }
          },
          {"contentBlockStop": {"contentBlockIndex": 1}},
      ]
    chunks += [
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 8,
                    "totalTokens": 18,
                }
            }
        },
    ]
    return chunks

  @pytest.mark.asyncio
  async def test_streaming_text(self):
    model = _make_model()
    client = _mock_client(model)
    chunks = self._build_stream_chunks(text="Hi there!")

    def fake_converse_stream(**kwargs):
      return {"stream": iter(chunks)}

    client.converse_stream.side_effect = fake_converse_stream

    req = _make_llm_request(
        contents=[
            types.Content(
                role="user", parts=[types.Part.from_text(text="hello")]
            )
        ]
    )
    responses = []
    async for r in model.generate_content_async(req, stream=True):
      responses.append(r)

    # Partial chunks + final aggregated response
    partials = [r for r in responses if r.partial]
    finals = [r for r in responses if not r.partial]
    assert len(partials) >= 1
    assert len(finals) == 1
    assert finals[0].content.parts[0].text == "Hi there!"
    assert finals[0].usage_metadata.total_token_count == 18

  @pytest.mark.asyncio
  async def test_streaming_tool_use(self):
    model = _make_model()
    client = _mock_client(model)
    chunks = self._build_stream_chunks(
        text="",
        tool_use={"id": "t1", "name": "search", "input": {"q": "bedrock"}},
    )

    def fake_converse_stream(**kwargs):
      return {"stream": iter(chunks)}

    client.converse_stream.side_effect = fake_converse_stream

    req = _make_llm_request()
    responses = []
    async for r in model.generate_content_async(req, stream=True):
      responses.append(r)

    final = [r for r in responses if not r.partial][0]
    fc_parts = [p for p in final.content.parts if p.function_call]
    assert len(fc_parts) == 1
    assert fc_parts[0].function_call.name == "search"
    assert fc_parts[0].function_call.args == {"q": "bedrock"}
    assert fc_parts[0].function_call.id == "t1"


# ---------------------------------------------------------------------------
# boto3 client — importerror, session, region resolution
# ---------------------------------------------------------------------------


class TestClient:

  def test_boto3_import_error(self):
    model = _make_model()
    # Ensure _client is not already cached.
    assert "_client" not in model.__dict__
    with patch.dict("sys.modules", {"boto3": None}):
      with pytest.raises(ImportError, match="BedrockModel requires"):
        _ = model._client

  def test_custom_boto_session_used(self):
    mock_session = MagicMock()
    mock_session.region_name = "eu-central-1"
    mock_session.client.return_value = MagicMock()

    model = BedrockModel(
        model="amazon.nova-pro-v1:0",
        boto_session=mock_session,
    )
    _ = model._client

    mock_session.client.assert_called_once_with(
        "bedrock-runtime", region_name="eu-central-1", config=ANY
    )

  def test_region_from_aws_region_env(self):
    with patch.dict(os.environ, {"AWS_REGION": "ap-northeast-1"}, clear=False):
      with patch("boto3.Session") as mock_cls:
        mock_sess = MagicMock()
        mock_sess.region_name = None
        mock_cls.return_value = mock_sess
        mock_sess.client.return_value = MagicMock()

        model = BedrockModel(model="amazon.nova-pro-v1:0")
        _ = model._client

        mock_sess.client.assert_called_once_with(
            "bedrock-runtime", region_name="ap-northeast-1", config=ANY
        )

  def test_region_fallback_to_default(self):
    env_clean = {
        k: v
        for k, v in os.environ.items()
        if k not in ("AWS_REGION", "AWS_DEFAULT_REGION")
    }
    with patch.dict(os.environ, env_clean, clear=True):
      with patch("boto3.Session") as mock_cls:
        mock_sess = MagicMock()
        mock_sess.region_name = None
        mock_cls.return_value = mock_sess
        mock_sess.client.return_value = MagicMock()

        model = BedrockModel(model="amazon.nova-pro-v1:0")
        _ = model._client

        mock_sess.client.assert_called_once_with(
            "bedrock-runtime", region_name="us-east-1", config=ANY
        )

  def test_user_agent_set_on_client(self):
    with patch("boto3.Session") as mock_cls:
      mock_sess = MagicMock()
      mock_sess.region_name = "us-east-1"
      mock_cls.return_value = mock_sess
      mock_sess.client.return_value = MagicMock()

      model = BedrockModel(model="amazon.nova-pro-v1:0")
      _ = model._client

      _, kwargs = mock_sess.client.call_args
      config = kwargs.get("config")
      assert config is not None
      assert "google-adk-community" in config.user_agent_extra

  def test_boto_session_excluded_from_serialisation(self):
    mock_session = MagicMock()
    model = BedrockModel(
        model="amazon.nova-pro-v1:0",
        boto_session=mock_session,
    )
    dumped = model.model_dump()
    assert "boto_session" not in dumped


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def _make_client_error(code: str, message: str) -> "MagicMock":
  """Build a mock botocore ClientError."""
  from botocore.exceptions import ClientError

  error_response = {"Error": {"Code": code, "Message": message}}
  return ClientError(error_response, "converse")


class TestClientErrorHandling:

  @pytest.mark.asyncio
  async def test_throttling_exception_propagates(self):
    model = _make_model()
    client = _mock_client(model)
    from botocore.exceptions import ClientError

    client.converse.side_effect = _make_client_error(
        "ThrottlingException", "Rate exceeded"
    )

    req = _make_llm_request()
    with pytest.raises(ClientError) as exc_info:
      async for _ in model.generate_content_async(req, stream=False):
        pass

    assert exc_info.value.response["Error"]["Code"] == "ThrottlingException"

  @pytest.mark.asyncio
  async def test_context_overflow_exception_propagates(self):
    model = _make_model()
    client = _mock_client(model)
    from botocore.exceptions import ClientError

    client.converse.side_effect = _make_client_error(
        "ValidationException", "Input is too long for requested model"
    )

    req = _make_llm_request()
    with pytest.raises(ClientError) as exc_info:
      async for _ in model.generate_content_async(req, stream=False):
        pass

    assert exc_info.value.response["Error"]["Code"] == "ValidationException"

  @pytest.mark.asyncio
  async def test_streaming_throttling_propagates(self):
    model = _make_model()
    client = _mock_client(model)
    from botocore.exceptions import ClientError

    client.converse_stream.side_effect = _make_client_error(
        "ThrottlingException", "Rate exceeded"
    )

    req = _make_llm_request()
    with pytest.raises(ClientError):
      async for _ in model.generate_content_async(req, stream=True):
        pass


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

  def test_empty_contents_builds_empty_messages(self):
    model = _make_model()
    req = _make_llm_request(contents=[])
    payload = model._build_request(req)
    assert payload["messages"] == []

  def test_content_with_all_unsupported_parts_skipped(self):
    # A Content where every Part is unsupported → message is None → not added.
    model = _make_model()
    content = types.Content(role="user", parts=[types.Part()])
    msg = _content_to_bedrock_message(content)
    assert msg is None

  @pytest.mark.asyncio
  async def test_guardrail_intervened_finish_reason_non_streaming(self):
    model = _make_model()
    client = _mock_client(model)
    client.converse.return_value = {
        "output": {
            "message": {"role": "assistant", "content": [{"text": "blocked"}]}
        },
        "stopReason": "guardrail_intervened",
        "usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6},
    }

    responses = []
    async for r in model.generate_content_async(
        _make_llm_request(), stream=False
    ):
      responses.append(r)

    assert responses[0].finish_reason == types.FinishReason.SAFETY

  def test_function_response_dict_fallback_json_dumps(self):
    """Dict with neither 'result' nor 'content' key → json.dumps path."""
    part = types.Part.from_function_response(
        name="my_tool", response={"foo": "bar", "baz": 123}
    )
    part.function_response.id = "tool-789"
    block = _part_to_bedrock_block(part)
    assert block is not None
    text = block["toolResult"]["content"][0]["text"]
    assert json.loads(text) == {"foo": "bar", "baz": 123}

  def test_function_declaration_with_parameters_json_schema(self):
    """When parameters_json_schema is set, it should be used directly."""
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    fd = types.FunctionDeclaration(
        name="get_weather",
        description="Get weather",
        parameters_json_schema=schema,
    )
    spec = _function_declaration_to_tool_spec(fd)
    assert spec["toolSpec"]["inputSchema"]["json"] == schema

  def test_empty_text_part_preserved(self):
    """Empty string text part should not be silently dropped."""
    part = types.Part.from_text(text="")
    block = _part_to_bedrock_block(part)
    assert block == {"text": ""}

  def test_region_name_directly_specified(self):
    with patch("boto3.Session") as mock_cls:
      mock_sess = MagicMock()
      mock_cls.return_value = mock_sess
      mock_sess.client.return_value = MagicMock()

      model = BedrockModel(
          model="amazon.nova-pro-v1:0", region_name="ap-northeast-2"
      )
      _ = model._client

      mock_sess.client.assert_called_once_with(
          "bedrock-runtime", region_name="ap-northeast-2", config=ANY
      )


class TestStreamingEdgeCases:

  def _build_text_and_tool_chunks(self):
    """Build chunks with both text and tool_use in a single response."""
    return [
        {"messageStart": {"role": "assistant"}},
        # Text block
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "Let me search."},
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        # Tool use block
        {
            "contentBlockStart": {
                "contentBlockIndex": 1,
                "start": {"toolUse": {"toolUseId": "t1", "name": "search"}},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '{"q": "adk"}'}},
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
        {
            "metadata": {
                "usage": {
                    "inputTokens": 15,
                    "outputTokens": 12,
                    "totalTokens": 27,
                }
            }
        },
    ]

  @pytest.mark.asyncio
  async def test_streaming_text_and_tool_combined(self):
    """Streaming response with both text and tool_use in one turn."""
    model = _make_model()
    client = _mock_client(model)
    chunks = self._build_text_and_tool_chunks()

    def fake_converse_stream(**kwargs):
      return {"stream": iter(chunks)}

    client.converse_stream.side_effect = fake_converse_stream

    req = _make_llm_request()
    responses = []
    async for r in model.generate_content_async(req, stream=True):
      responses.append(r)

    final = [r for r in responses if not r.partial][0]
    assert len(final.content.parts) == 2
    assert final.content.parts[0].text == "Let me search."
    assert final.content.parts[1].function_call.name == "search"
    assert final.content.parts[1].function_call.args == {"q": "adk"}
    assert final.content.parts[1].function_call.id == "t1"

  @pytest.mark.asyncio
  async def test_streaming_malformed_tool_json_fallback(self):
    """Malformed tool input JSON should fall back to empty dict."""
    model = _make_model()
    client = _mock_client(model)
    chunks = [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "t1", "name": "broken"}},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": "{invalid json!!"}},
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
        {
            "metadata": {
                "usage": {
                    "inputTokens": 5,
                    "outputTokens": 3,
                    "totalTokens": 8,
                }
            }
        },
    ]

    def fake_converse_stream(**kwargs):
      return {"stream": iter(chunks)}

    client.converse_stream.side_effect = fake_converse_stream

    req = _make_llm_request()
    responses = []
    async for r in model.generate_content_async(req, stream=True):
      responses.append(r)

    final = [r for r in responses if not r.partial][0]
    assert final.content.parts[0].function_call.name == "broken"
    assert final.content.parts[0].function_call.args == {}

  @pytest.mark.asyncio
  async def test_streaming_non_client_error_propagates(self):
    """Non-ClientError exceptions in stream thread should propagate."""
    model = _make_model()
    client = _mock_client(model)

    client.converse_stream.side_effect = ConnectionError("Network unreachable")

    req = _make_llm_request()
    with pytest.raises(ConnectionError, match="Network unreachable"):
      async for _ in model.generate_content_async(req, stream=True):
        pass
