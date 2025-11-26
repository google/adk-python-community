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

import base64
import os
from unittest.mock import AsyncMock, MagicMock, patch
from unittest.mock import Mock

import openai
import pytest

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from google.adk.models.llm_request import LlmRequest
from google.adk_community.models.openai_llm import (
    OpenAI,
    content_to_openai_message,
    function_declaration_to_openai_tool,
    openai_response_to_llm_response,
    part_to_openai_content,
    to_openai_role,
)
from google.genai import types


class TestHelperFunctions:
    """Test cases for helper functions."""

    def test_to_openai_role(self):
        """Test role conversion from ADK to OpenAI format."""
        assert to_openai_role("model") == "assistant"
        assert to_openai_role("system") == "system"
        assert to_openai_role("user") == "user"
        assert to_openai_role(None) == "user"
        assert to_openai_role("unknown") == "user"

    @pytest.mark.asyncio
    async def test_part_to_openai_content_text(self):
        """Test conversion of text part to OpenAI content."""
        part = types.Part(text="Hello, world!")
        result = await part_to_openai_content(part)
        assert result == {"type": "text", "text": "Hello, world!"}


    @pytest.mark.asyncio
    async def test_part_to_openai_content_inline_image(self):
        """Test conversion of inline image part to OpenAI content."""
        image_data = b"fake_image_data"
        part = types.Part(
            inline_data=types.Blob(
                mime_type="image/png", data=image_data, display_name="test.png"
            )
        )
        result = await part_to_openai_content(part)
        assert result["type"] == "image_url"
        assert "data:image/png;base64," in result["image_url"]["url"]
        # Verify base64 encoding
        b64_data = result["image_url"]["url"].split(",")[1]
        assert base64.b64decode(b64_data) == image_data

    @pytest.mark.asyncio
    async def test_part_to_openai_content_executable_code(self):
        """Test conversion of executable code part to OpenAI content."""
        part = types.Part(
            executable_code=types.ExecutableCode(code="print('hello')")
        )
        result = await part_to_openai_content(part)
        assert result == {
            "type": "text",
            "text": "```python\nprint('hello')\n```",
        }

    @pytest.mark.asyncio
    async def test_part_to_openai_content_code_execution_result(self):
        """Test conversion of code execution result part to OpenAI content."""
        part = types.Part(
            code_execution_result=types.CodeExecutionResult(
                output="hello world"
            )
        )
        result = await part_to_openai_content(part)
        assert result == {
            "type": "text",
            "text": "Execution Result:\n```\nhello world\n```",
        }

    @pytest.mark.asyncio
    async def test_content_to_openai_message_text(self):
        """Test conversion of text content to OpenAI message."""
        content = types.Content(
            role="user", parts=[types.Part(text="Hello!")]
        )
        message = await content_to_openai_message(content)
        assert message["role"] == "user"
        assert len(message["content"]) == 1
        # Responses API uses input_text for user/system messages
        assert message["content"][0] == {"type": "input_text", "text": "Hello!"}

    @pytest.mark.asyncio
    async def test_content_to_openai_message_text_model_role(self):
        """Test conversion of text content with model role uses output_text."""
        content = types.Content(
            role="model", parts=[types.Part(text="Hello from model!")]
        )
        message = await content_to_openai_message(content)
        assert message["role"] == "assistant"
        assert len(message["content"]) == 1
        # Responses API uses output_text for assistant/model messages
        assert message["content"][0] == {"type": "output_text", "text": "Hello from model!"}

    @pytest.mark.asyncio
    async def test_content_to_openai_message_with_function_call(self):
        """Test conversion of content with function call to OpenAI message."""
        content = types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="call_123",
                        name="get_weather",
                        args={"location": "San Francisco"},
                    )
                )
            ],
        )
        message = await content_to_openai_message(content)
        assert message["role"] == "assistant"
        assert "tool_calls" in message
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["id"] == "call_123"
        assert message["tool_calls"][0]["type"] == "function"
        assert message["tool_calls"][0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_content_to_openai_message_with_function_response(self):
        """Test conversion of content with function response to OpenAI tool message."""
        content = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="call_123", response={"temperature": 72}
                    )
                )
            ],
        )
        message = await content_to_openai_message(content)
        # Function responses should create tool messages
        assert isinstance(message, list)
        assert len(message) == 1
        tool_message = message[0]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == "call_123"
        # Tool message content should be a string, not an array
        assert isinstance(tool_message["content"], str)
        assert "temperature" in tool_message["content"]

    @pytest.mark.asyncio
    async def test_content_to_openai_message_with_multiple_function_responses(self):
        """Test conversion of content with multiple function responses."""
        content = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="call_123", response={"result": "first"}
                    )
                ),
                types.Part(
                    function_response=types.FunctionResponse(
                        id="call_456", response={"result": "second"}
                    )
                ),
            ],
        )
        messages = await content_to_openai_message(content)
        # Should return a list of tool messages
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_123"
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "call_456"

    @pytest.mark.asyncio
    async def test_content_to_openai_message_with_function_response_no_id(self):
        """Test that function response without id raises ValueError.
        
        Function responses require an id field to create tool messages.
        The Responses API requires tool_call_id for function_call_output items.
        """
        content = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=None, response={"result": "test"}
                    )
                )
            ],
        )
        # Should raise ValueError when function_response has no id
        with pytest.raises(ValueError, match="Function response missing required 'id' field"):
            await content_to_openai_message(content)

    def test_function_declaration_to_openai_tool(self):
        """Test conversion of function declaration to OpenAI tool."""
        func_decl = types.FunctionDeclaration(
            name="get_weather",
            description="Get weather for a location",
            parameters=types.Schema(
                type="object",
                properties={
                    "location": types.Schema(
                        type="string", description="City name"
                    )
                },
                required=["location"],
            ),
        )
        tool = function_declaration_to_openai_tool(func_decl)
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["description"] == "Get weather for a location"
        assert "location" in tool["function"]["parameters"]["properties"]
        assert "location" in tool["function"]["parameters"]["required"]

    def test_function_declaration_to_openai_tool_type_conversion(self):
        """Test type conversion in function declaration."""
        func_decl = types.FunctionDeclaration(
            name="test_func",
            parameters=types.Schema(
                type="object",
                properties={
                    "string_param": types.Schema(type="string"),
                    "number_param": types.Schema(type="number"),
                    "bool_param": types.Schema(type="boolean"),
                },
            ),
        )
        tool = function_declaration_to_openai_tool(func_decl)
        props = tool["function"]["parameters"]["properties"]
        assert props["string_param"]["type"] == "string"
        assert props["number_param"]["type"] == "number"
        assert props["bool_param"]["type"] == "boolean"

    @pytest.mark.asyncio
    async def test_openai_response_to_llm_response_text(self):
        """Test conversion of OpenAI Responses API response with text to LlmResponse."""
        mock_response = MagicMock()
        # Responses API format: output is a list of output items
        mock_output_item = MagicMock()
        mock_output_item.type = "message"
        mock_content_part = MagicMock()
        mock_content_part.type = "text"
        mock_content_part.text = "Hello, world!"
        mock_output_item.content = [mock_content_part]
        mock_response.output = [mock_output_item]
        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5
        mock_response.usage = mock_usage
        mock_response.incomplete_details = None  # Response completed normally

        llm_response = await openai_response_to_llm_response(mock_response)
        assert llm_response.content is not None
        assert llm_response.content.role == "model"
        assert len(llm_response.content.parts) == 1
        assert llm_response.content.parts[0].text == "Hello, world!"
        assert llm_response.usage_metadata is not None
        assert llm_response.usage_metadata.prompt_token_count == 10
        assert llm_response.usage_metadata.candidates_token_count == 5
        assert llm_response.finish_reason == "STOP"

    @pytest.mark.asyncio
    async def test_openai_response_to_llm_response_with_tool_calls(self):
        """Test conversion of OpenAI Responses API response with tool calls."""
        mock_response = MagicMock()
        # Responses API format: output is a list of output items
        # Responses API uses "function_call" type (not "function")
        mock_output_item = MagicMock()
        mock_output_item.type = "function_call"
        mock_output_item.call_id = "call_123"
        mock_output_item.name = "get_weather"
        mock_output_item.arguments = '{"location": "SF"}'
        mock_response.output = [mock_output_item]
        mock_response.output_text = None  # Explicitly set to None
        mock_response.usage = None
        mock_response.incomplete_details = None  # Response completed normally

        llm_response = await openai_response_to_llm_response(mock_response)
        assert llm_response.content is not None
        assert len(llm_response.content.parts) == 1
        assert llm_response.content.parts[0].function_call is not None
        assert llm_response.content.parts[0].function_call.name == "get_weather"
        assert llm_response.finish_reason == "STOP"

    @pytest.mark.asyncio
    async def test_openai_response_to_llm_response_empty(self):
        """Test conversion of empty OpenAI Responses API response."""
        mock_response = MagicMock()
        mock_response.output = []  # Empty output list
        mock_response.output_text = None  # Explicitly set to None to avoid MagicMock issues
        mock_response.usage = None
        mock_response.incomplete_details = None

        llm_response = await openai_response_to_llm_response(mock_response)
        assert llm_response.content is None


class TestOpenAIClass:
    """Test cases for OpenAI class."""

    def test_supported_models(self):
        """Test supported_models class method."""
        models = OpenAI.supported_models()
        assert len(models) > 0
        assert any("gpt" in model for model in models)
        assert any("o1" in model for model in models)

    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"})
    def test_is_azure_with_azure_endpoint(self):
        """Test Azure detection with explicit azure_endpoint from env."""
        openai_client = OpenAI()
        assert openai_client._is_azure() is True

    @patch.dict(os.environ, {"OPENAI_BASE_URL": "https://test.openai.azure.com/v1/"})
    def test_is_azure_with_azure_base_url(self):
        """Test Azure detection with Azure-looking base_url from env."""
        openai_client = OpenAI()
        assert openai_client._is_azure() is True

    @patch.dict(os.environ, {"OPENAI_BASE_URL": "https://test.services.ai.azure.com/v1/"})
    def test_is_azure_with_services_ai_azure(self):
        """Test Azure detection with services.ai.azure.com domain from env."""
        openai_client = OpenAI()
        assert openai_client._is_azure() is True

    @patch.dict(os.environ, {"OPENAI_BASE_URL": "https://api.openai.com/v1/"})
    def test_is_azure_with_regular_openai(self):
        """Test Azure detection returns False for regular OpenAI."""
        openai_client = OpenAI()
        assert openai_client._is_azure() is False

    def test_is_azure_without_endpoint(self):
        """Test Azure detection returns False when no Azure indicators."""
        openai_client = OpenAI()
        assert openai_client._is_azure() is False

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("openai.AsyncOpenAI")
    def test_get_openai_client_regular(self, mock_openai_class):
        """Test getting regular OpenAI client."""
        openai_client = OpenAI()
        client = openai_client._get_openai_client()
        mock_openai_class.assert_called_once()
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs["api_key"] == "test-key"

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_API_VERSION": "2024-02-15-preview",
        },
    )
    @patch("openai.AsyncAzureOpenAI")
    def test_get_openai_client_azure(self, mock_azure_openai_class):
        """Test getting Azure OpenAI client."""
        openai_client = OpenAI()
        client = openai_client._get_openai_client()
        mock_azure_openai_class.assert_called_once()
        call_kwargs = mock_azure_openai_class.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["azure_endpoint"] == "https://test.openai.azure.com/"
        assert call_kwargs["api_version"] == "2024-02-15-preview"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"})
    @patch("openai.AsyncOpenAI")
    def test_get_openai_client_from_env(self, mock_openai_class):
        """Test getting OpenAI client with API key from environment."""
        openai_client = OpenAI()
        client = openai_client._get_openai_client()
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs["api_key"] == "env-key"

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_API_KEY": "azure-env-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        },
    )
    @patch("openai.AsyncAzureOpenAI")
    def test_get_openai_client_azure_from_env(self, mock_azure_openai_class):
        """Test getting Azure OpenAI client with API key from environment."""
        openai_client = OpenAI()
        client = openai_client._get_openai_client()
        call_kwargs = mock_azure_openai_class.call_args[1]
        assert call_kwargs["api_key"] == "azure-env-key"

    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"})
    @patch("openai.AsyncAzureOpenAI")
    def test_get_openai_client_azure_fallback(self, mock_azure_openai_class):
        """Test that ImportError is raised when AsyncAzureOpenAI not available."""
        mock_azure_openai_class.side_effect = AttributeError(
            "No AsyncAzureOpenAI"
        )
        openai_client = OpenAI()
        with pytest.raises(ImportError) as exc_info:
            openai_client._get_openai_client()
        assert "openai>=2.7.2" in str(exc_info.value)

    def test_get_file_extension(self):
        """Test file extension mapping from MIME type."""
        openai_client = OpenAI()
        assert openai_client._get_file_extension("application/pdf") == ".pdf"
        assert openai_client._get_file_extension("image/jpeg") == ".jpg"
        assert openai_client._get_file_extension("image/png") == ".png"
        assert openai_client._get_file_extension("text/plain") == ".txt"
        assert openai_client._get_file_extension("unknown/type") == ".bin"

    def test_preprocess_request(self):
        """Test request preprocessing."""
        openai_client = OpenAI(model="gpt-4")
        llm_request = LlmRequest(contents=[])
        openai_client._preprocess_request(llm_request)
        assert llm_request.model == "gpt-4"

    def test_preprocess_request_with_existing_model(self):
        """Test request preprocessing doesn't override existing model."""
        openai_client = OpenAI(model="gpt-4")
        llm_request = LlmRequest(contents=[], model="gpt-3.5-turbo")
        openai_client._preprocess_request(llm_request)
        assert llm_request.model == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_handle_file_data_inline_image(self):
        """Test handling inline image data."""
        openai_client = OpenAI()
        image_data = b"fake_image_data"
        part = types.Part(
            inline_data=types.Blob(
                mime_type="image/png", data=image_data, display_name="test.png"
            )
        )
        result = await openai_client._handle_file_data(part)
        assert result["type"] == "image_url"
        assert "data:image/png;base64," in result["image_url"]["url"]

    @pytest.mark.asyncio
    async def test_handle_file_data_file_reference(self):
        """Test handling file reference data."""
        openai_client = OpenAI()
        part = types.Part(
            file_data=types.FileData(
                file_uri="gs://bucket/file.pdf",
                mime_type="application/pdf",
                display_name="test.pdf",
            )
        )
        result = await openai_client._handle_file_data(part)
        assert result["type"] == "text"
        assert "FILE REFERENCE" in result["text"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_non_streaming(self):
        """Test non-streaming content generation."""
        openai_client = OpenAI(model="gpt-4")
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Hello!")]
                )
            ]
        )

        # Mock OpenAI client and response (Responses API format)
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_output_item = MagicMock()
        mock_output_item.type = "message"
        mock_content_part = MagicMock()
        mock_content_part.type = "text"
        mock_content_part.text = "Hi there!"
        mock_output_item.content = [mock_content_part]
        mock_response.output = [mock_output_item]
        mock_usage = MagicMock()
        mock_usage.input_tokens = 5
        mock_usage.output_tokens = 3
        mock_response.usage = mock_usage
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            responses = []
            async for response in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                responses.append(response)

        assert len(responses) == 1
        assert responses[0].content is not None
        assert responses[0].content.parts[0].text == "Hi there!"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_streaming(self):
        """Test streaming content generation."""
        openai_client = OpenAI(model="gpt-4")
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Hello!")]
                )
            ]
        )

        # Mock streaming response (Responses API format)
        mock_client = AsyncMock()
        # First event: response.created
        mock_event1 = MagicMock()
        mock_event1.type = "response.created"
        mock_event1.response = MagicMock()
        mock_event1.response.model = "gpt-4"
        mock_event1.response.id = "resp_123"
        
        # Second event: content part added
        mock_event2 = MagicMock()
        mock_event2.type = "response.content_part.added"
        mock_part2 = MagicMock()
        mock_part2.type = "text"
        mock_part2.text = "Hi"
        mock_event2.part = mock_part2
        
        # Third event: content part added
        mock_event3 = MagicMock()
        mock_event3.type = "response.content_part.added"
        mock_part3 = MagicMock()
        mock_part3.type = "text"
        mock_part3.text = " there!"
        mock_event3.part = mock_part3
        
        # Fourth event: response.done
        mock_event4 = MagicMock()
        mock_event4.type = "response.done"
        mock_event4.response = MagicMock()
        mock_event4.response.usage = MagicMock()
        mock_event4.response.usage.input_tokens = 5
        mock_event4.response.usage.output_tokens = 3
        mock_event4.response.incomplete_details = None

        async def mock_stream():
            yield mock_event1
            yield mock_event2
            yield mock_event3
            yield mock_event4

        mock_client.responses.create = AsyncMock(
            return_value=mock_stream()
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            responses = []
            async for response in openai_client.generate_content_async(
                llm_request, stream=True
            ):
                responses.append(response)

        # Should have partial responses plus final turn_complete response
        assert len(responses) >= 2
        # Check that we have partial responses
        partial_responses = [
            r for r in responses if r.partial and not r.turn_complete
        ]
        assert len(partial_responses) > 0
        # Check final response
        final_response = responses[-1]
        assert final_response.turn_complete is True

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_system_instruction(self):
        """Test content generation with system instruction."""
        openai_client = OpenAI()
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Hello!")]
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful assistant."
            ),
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        # Verify system instruction was added
        call_args = mock_client.responses.create.call_args[1]
        # Responses API uses 'instructions' parameter for system prompt
        assert "instructions" in call_args
        assert call_args["instructions"] == "You are a helpful assistant."
        # Verify input (messages) is present
        assert "input" in call_args
        messages = call_args["input"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_tools(self):
        """Test content generation with tools."""
        openai_client = OpenAI()
        func_decl = types.FunctionDeclaration(
            name="get_weather",
            description="Get weather",
            parameters=types.Schema(type="object", properties={}),
        )
        tool = types.Tool(function_declarations=[func_decl])
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="What's the weather?")]
                )
            ],
            config=types.GenerateContentConfig(tools=[tool]),
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        # Verify tools were added
        call_args = mock_client.responses.create.call_args[1]
        assert "tools" in call_args
        assert len(call_args["tools"]) == 1
        # Smart default: fresh conversation with tools uses "required" (not "auto")
        assert call_args["tool_choice"] == "required"
        assert call_args["parallel_tool_calls"] is True


    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_error_handling(self):
        """Test error handling in content generation."""
        openai_client = OpenAI()
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Hello!")]
                )
            ]
        )

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(
            side_effect=openai.APIError("API Error", request=None, body=None)
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            responses = []
            async for response in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                responses.append(response)

        assert len(responses) == 1
        assert responses[0].error_code == "OPENAI_API_ERROR"
        assert "API Error" in responses[0].error_message

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_upload_file_to_openai_disabled(self):
        """Test file upload when Files API is disabled."""
        openai_client = OpenAI(use_files_api=False)
        with pytest.raises(ValueError, match="Files API is disabled"):
            await openai_client._upload_file_to_openai(
                b"test data", "application/pdf", "test.pdf"
            )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_upload_file_to_openai_enabled(self):
        """Test file upload when Files API is enabled."""
        openai_client = OpenAI(use_files_api=True)

        mock_client = AsyncMock()
        mock_file = MagicMock()
        mock_file.id = "file-123"
        mock_client.files.create = AsyncMock(return_value=mock_file)

        import tempfile
        import os

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            # Create a real temp file that will be used
            real_temp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            )
            real_temp.write(b"test data")
            real_temp.close()
            real_temp_path = real_temp.name

            try:
                # Mock tempfile.NamedTemporaryFile to return our real file
                def mock_named_tempfile(*args, **kwargs):
                    mock_file = MagicMock()
                    mock_file.name = real_temp_path
                    mock_file.write = Mock()
                    mock_file.__enter__ = Mock(return_value=mock_file)
                    mock_file.__exit__ = Mock(return_value=None)
                    return mock_file

                with patch(
                    "google.adk_community.models.openai_llm.tempfile.NamedTemporaryFile",
                    side_effect=mock_named_tempfile,
                ):
                    # Mock open to return the file handle
                    with patch("builtins.open") as mock_open:
                        mock_file_handle = open(real_temp_path, "rb")
                        mock_open.return_value.__enter__ = Mock(
                            return_value=mock_file_handle
                        )
                        mock_open.return_value.__exit__ = Mock(return_value=None)

                        # Mock os.path.exists and os.unlink
                        with patch("os.path.exists", return_value=True):
                            with patch("os.unlink"):
                                file_id = await openai_client._upload_file_to_openai(
                                    b"test data",
                                    "application/pdf",
                                    "test.pdf",
                                )

                    assert file_id == "file-123"
                    mock_client.files.create.assert_called_once()
            finally:
                # Clean up
                if os.path.exists(real_temp_path):
                    os.unlink(real_temp_path)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_handle_file_data_with_files_api_pdf(self):
        """Test handling PDF file with Files API enabled."""
        openai_client = OpenAI(use_files_api=True)
        pdf_data = b"fake_pdf_data"
        part = types.Part(
            inline_data=types.Blob(
                mime_type="application/pdf",
                data=pdf_data,
                display_name="test.pdf",
            )
        )

        mock_client = AsyncMock()
        mock_file = MagicMock()
        mock_file.id = "file-123"
        mock_client.files.create = AsyncMock(return_value=mock_file)

        import tempfile
        import os

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            # Create a real temp file that will be used
            real_temp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            )
            real_temp.write(pdf_data)
            real_temp.close()
            real_temp_path = real_temp.name

            try:
                # Mock tempfile.NamedTemporaryFile to return our real file
                def mock_named_tempfile(*args, **kwargs):
                    mock_file = MagicMock()
                    mock_file.name = real_temp_path
                    mock_file.write = Mock()
                    mock_file.__enter__ = Mock(return_value=mock_file)
                    mock_file.__exit__ = Mock(return_value=None)
                    return mock_file

                with patch(
                    "google.adk_community.models.openai_llm.tempfile.NamedTemporaryFile",
                    side_effect=mock_named_tempfile,
                ):
                    # Mock open to return the file handle
                    with patch("builtins.open") as mock_open:
                        mock_file_handle = open(real_temp_path, "rb")
                        mock_open.return_value.__enter__ = Mock(
                            return_value=mock_file_handle
                        )
                        mock_open.return_value.__exit__ = Mock(return_value=None)

                        # Mock os.path.exists and os.unlink
                        with patch("os.path.exists", return_value=True):
                            with patch("os.unlink"):
                                result = await openai_client._handle_file_data(part)

                    # Should return file type with file_id
                    assert result["type"] == "file"
                    assert result["file_id"] == "file-123"
            finally:
                # Clean up
                if os.path.exists(real_temp_path):
                    os.unlink(real_temp_path)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_handle_file_data_files_api_fallback(self):
        """Test Files API fallback to base64 on error.
        
        Non-image files that fail to upload should fallback to base64
        and be returned as type: "text" with data URI (not image_url).
        """
        openai_client = OpenAI(use_files_api=True)
        pdf_data = b"fake_pdf_data"
        part = types.Part(
            inline_data=types.Blob(
                mime_type="application/pdf",
                data=pdf_data,
                display_name="test.pdf",
            )
        )

        mock_client = AsyncMock()
        mock_client.files.create = AsyncMock(side_effect=Exception("Upload failed"))

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            result = await openai_client._handle_file_data(part)

        # Non-image files should return as type: "text" with data URI (not image_url)
        assert result["type"] == "text"
        assert "base64" in result["text"]
        assert "data:application/pdf;base64," in result["text"]

    @pytest.mark.asyncio
    async def test_handle_file_data_with_pdf_disabled(self):
        """Test handling PDF file with Files API disabled (base64 fallback).
        
        Non-image files should return as type: "text" with data URI
        when Files API is disabled (not image_url).
        """
        openai_client = OpenAI(use_files_api=False)
        pdf_data = b"fake_pdf_data"
        part = types.Part(
            inline_data=types.Blob(
                mime_type="application/pdf",
                data=pdf_data,
                display_name="test.pdf",
            )
        )

        result = await openai_client._handle_file_data(part)

        # Non-image files should return as type: "text" with data URI (not image_url)
        assert result["type"] == "text"
        assert "base64" in result["text"]
        assert "data:application/pdf;base64," in result["text"]

    @pytest.mark.asyncio
    async def test_part_to_openai_content_with_openai_instance(self):
        """Test part conversion with OpenAI instance for file handling."""
        openai_client = OpenAI()
        image_data = b"fake_image_data"
        part = types.Part(
            inline_data=types.Blob(
                mime_type="image/jpeg", data=image_data, display_name="test.jpg"
            )
        )
        result = await part_to_openai_content(part, openai_client)
        assert result["type"] == "image_url"
        assert "data:image/jpeg;base64," in result["image_url"]["url"]

    @pytest.mark.asyncio
    async def test_content_to_openai_message_with_openai_instance(self):
        """Test content conversion with OpenAI instance."""
        openai_client = OpenAI()
        content = types.Content(
            role="user",
            parts=[
                types.Part(
                    inline_data=types.Blob(
                        mime_type="image/png", data=b"fake_data"
                    )
                )
            ],
        )
        message = await content_to_openai_message(content, openai_client)
        assert message["role"] == "user"
        assert len(message["content"]) == 1
        # Responses API uses input_image for user/system messages
        assert message["content"][0]["type"] == "input_image"

    def test_function_declaration_to_openai_tool_no_parameters(self):
        """Test function declaration conversion without parameters."""
        func_decl = types.FunctionDeclaration(
            name="simple_func", description="A simple function"
        )
        tool = function_declaration_to_openai_tool(func_decl)
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "simple_func"
        assert tool["function"]["parameters"]["type"] == "object"
        assert "required" not in tool["function"]["parameters"]

    def test_function_declaration_to_openai_tool_no_description(self):
        """Test function declaration conversion without description."""
        func_decl = types.FunctionDeclaration(
            name="no_desc_func",
            parameters=types.Schema(type="object", properties={}),
        )
        tool = function_declaration_to_openai_tool(func_decl)
        assert tool["function"]["description"] == ""

    @pytest.mark.asyncio
    async def test_openai_response_to_llm_response_no_content(self):
        """Test conversion of OpenAI Responses API response with no content."""
        mock_response = MagicMock()
        mock_response.output = []  # Empty output
        mock_response.output_text = None  # Explicitly set to None to avoid MagicMock issues
        mock_response.usage = None
        mock_response.incomplete_details = None

        llm_response = await openai_response_to_llm_response(mock_response)
        # When there are no content parts, content is None
        assert llm_response.content is None
        assert llm_response.finish_reason == "STOP"

    @pytest.mark.asyncio
    async def test_openai_response_to_llm_response_invalid_json_tool_call(self):
        """Test conversion with invalid JSON in tool call arguments (Responses API format)."""
        mock_response = MagicMock()
        mock_output_item = MagicMock()
        # Responses API uses "function_call" type (not "function")
        mock_output_item.type = "function_call"
        mock_output_item.call_id = "call_123"
        mock_output_item.name = "test_func"
        mock_output_item.arguments = "invalid json {"
        mock_response.output = [mock_output_item]
        mock_response.output_text = None  # Explicitly set to None
        mock_response.usage = None
        mock_response.incomplete_details = None

        llm_response = await openai_response_to_llm_response(mock_response)
        assert llm_response.content is not None
        assert len(llm_response.content.parts) == 1
        # Should handle invalid JSON gracefully
        assert (
            llm_response.content.parts[0].function_call.args["arguments"]
            == "invalid json {"
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_config_max_tokens(self):
        """Test content generation with max_output_tokens from config (Responses API uses max_output_tokens directly)."""
        openai_client = OpenAI()
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Hello!")]
                )
            ],
            config=types.GenerateContentConfig(max_output_tokens=200),
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        call_args = mock_client.responses.create.call_args[1]
        assert call_args["max_output_tokens"] == 200

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_config_max_completion_tokens_o1(self):
        """Test content generation with max_output_tokens for o1 models (Responses API uses max_output_tokens directly)."""
        openai_client = OpenAI()
        llm_request = LlmRequest(
            model="o1-preview",
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Hello!")]
                )
            ],
            config=types.GenerateContentConfig(max_output_tokens=200),
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        call_args = mock_client.responses.create.call_args[1]
        assert call_args["max_output_tokens"] == 200

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_config_max_completion_tokens_o1_mini(self):
        """Test content generation with max_output_tokens for o1-mini model (Responses API uses max_output_tokens directly)."""
        openai_client = OpenAI()
        llm_request = LlmRequest(
            model="o1-mini",
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Hello!")]
                )
            ],
            config=types.GenerateContentConfig(max_output_tokens=150),
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        call_args = mock_client.responses.create.call_args[1]
        assert call_args["max_output_tokens"] == 150

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_config_temperature(self):
        """Test content generation with temperature from config."""
        openai_client = OpenAI()
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Hello!")]
                )
            ],
            config=types.GenerateContentConfig(temperature=0.9),
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        call_args = mock_client.responses.create.call_args[1]
        assert call_args["temperature"] == 0.9

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_json_mode(self):
        """Test content generation with JSON mode (structured output)."""
        openai_client = OpenAI()
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Return JSON")]
                )
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        call_args = mock_client.responses.create.call_args[1]
        # Responses API uses 'text' parameter with 'format'
        assert "text" in call_args
        assert call_args["text"]["format"] == {"type": "json_object"}

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_response_format(self):
        """Test content generation with JSON mode via response_mime_type.
        
        The response_mime_type="application/json" is converted to the Responses API's
        text format parameter: {"format": {"type": "json_object"}}.
        """
        openai_client = OpenAI()
        # Use response_mime_type for JSON mode (standard GenAI format)
        config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )

        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Return JSON")]
                )
            ],
            config=config,
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        call_args = mock_client.responses.create.call_args[1]
        # Responses API uses 'text' parameter with 'format'
        assert "text" in call_args
        assert call_args["text"]["format"] == {"type": "json_object"}

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_response_schema(self):
        """Test content generation with JSON schema for structured outputs."""
        openai_client = OpenAI()
        config = types.GenerateContentConfig(
            response_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name"],
            }
        )

        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Return structured data")]
                )
            ],
            config=config,
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        call_args = mock_client.responses.create.call_args[1]
        # Responses API uses 'text' parameter with 'format'
        assert "text" in call_args
        assert "format" in call_args["text"]
        format_obj = call_args["text"]["format"]
        assert format_obj["type"] == "json_schema"
        # Responses API has a flatter structure - name and schema directly in format
        assert "name" in format_obj
        assert "schema" in format_obj
        assert format_obj["schema"]["type"] == "object"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_top_p(self):
        """Test content generation with top_p parameter."""
        openai_client = OpenAI()
        config = types.GenerateContentConfig(top_p=0.9)
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Hello!")]
                )
            ],
            config=config,
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        call_args = mock_client.responses.create.call_args[1]
        assert call_args["top_p"] == 0.9

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_frequency_penalty(self):
        """Test content generation with frequency_penalty parameter."""
        openai_client = OpenAI()
        config = types.GenerateContentConfig(frequency_penalty=0.5)
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Hello!")]
                )
            ],
            config=config,
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        call_args = mock_client.responses.create.call_args[1]
        assert call_args["frequency_penalty"] == 0.5

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_presence_penalty(self):
        """Test content generation with presence_penalty parameter."""
        openai_client = OpenAI()
        config = types.GenerateContentConfig(presence_penalty=0.3)
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Hello!")]
                )
            ],
            config=config,
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        call_args = mock_client.responses.create.call_args[1]
        assert call_args["presence_penalty"] == 0.3

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_seed(self):
        """Test content generation with seed parameter."""
        openai_client = OpenAI()
        config = types.GenerateContentConfig(seed=42)
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Hello!")]
                )
            ],
            config=config,
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass
        call_args = mock_client.responses.create.call_args[1]
        assert call_args["seed"] == 42

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_stop(self):
        """Test content generation with stop sequences."""
        openai_client = OpenAI()
        # Use proper Google GenAI format with stop_sequences
        config = types.GenerateContentConfig(stop_sequences=["\n", "END"])

        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Hello!")]
                )
            ],
            config=config,
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        call_args = mock_client.responses.create.call_args[1]
        assert call_args["stop"] == ["\n", "END"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_tool_choice(self):
        """Test content generation with smart default tool_choice.
        
        NOTE: tool_config cannot be mapped (not in GenerateContentConfig).
        The implementation uses smart defaults based on conversation state.
        For a fresh conversation with tools, tool_choice defaults to "required".
        """
        openai_client = OpenAI()
        func_decl = types.FunctionDeclaration(
            name="test_function",
            description="A test function",
        )
        tool = types.Tool(function_declarations=[func_decl])
        # NOTE: tool_config cannot be mapped, so we don't set it
        config = types.GenerateContentConfig(tools=[tool])

        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Hello!")]
                )
            ],
            config=config,
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        call_args = mock_client.responses.create.call_args[1]
        # Smart default: fresh conversation with tools uses "required"
        assert call_args["tool_choice"] == "required"
        assert call_args["parallel_tool_calls"] is True

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_streaming_with_tool_calls(self):
        """Test streaming response with tool calls accumulation."""
        openai_client = OpenAI()
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Call a function")]
                )
            ],
        )

        # Create mock streaming events with tool calls (Responses API format)
        # First event: response.created
        mock_event1 = MagicMock()
        mock_event1.type = "response.created"
        mock_event1.response = MagicMock()
        mock_event1.response.model = "gpt-5.1"
        mock_event1.response.id = "resp_123"
        
        # Second event: function call added
        mock_event2 = MagicMock()
        mock_event2.type = "response.function_call.added"
        mock_func_call = MagicMock()
        mock_func_call.call_id = "call_123"
        mock_func_call.name = "test_function"
        mock_func_call.arguments = '{"key": "val'
        mock_event2.function_call = mock_func_call
        
        # Third event: function arguments delta
        mock_event3 = MagicMock()
        mock_event3.type = "response.function_call.arguments_delta"
        mock_event3.call_id = "call_123"
        mock_delta = MagicMock()
        mock_delta.text = 'ue"}'
        mock_event3.delta = mock_delta
        
        # Fourth event: response.done
        mock_event4 = MagicMock()
        mock_event4.type = "response.done"
        mock_event4.response = MagicMock()
        mock_event4.response.usage = MagicMock()
        mock_event4.response.usage.input_tokens = 10
        mock_event4.response.usage.output_tokens = 5
        mock_event4.response.incomplete_details = None

        async def mock_stream():
            yield mock_event1
            yield mock_event2
            yield mock_event3
            yield mock_event4

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_stream())

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            responses = []
            async for response in openai_client.generate_content_async(
                llm_request, stream=True
            ):
                responses.append(response)

        # Should have partial responses and final response
        assert len(responses) > 0
        final_response = responses[-1]
        assert final_response.turn_complete is True
        assert final_response.content is not None
        # Check that tool calls were accumulated
        if final_response.content and final_response.content.parts:
            has_tool_call = any(
                part.function_call for part in final_response.content.parts
            )
            assert has_tool_call

    @pytest.mark.asyncio
    async def test_openai_response_to_llm_response_with_model(self):
        """Test response conversion includes model name (Responses API format)."""
        mock_response = MagicMock()
        mock_output_item = MagicMock()
        mock_output_item.type = "message"
        mock_content_part = MagicMock()
        mock_content_part.type = "text"
        mock_content_part.text = "Hello"
        mock_output_item.content = [mock_content_part]
        mock_response.output = [mock_output_item]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.model = "gpt-5.1"
        mock_response.incomplete_details = None
        mock_response.system_fingerprint = "fp_123"

        result = await openai_response_to_llm_response(mock_response)
        assert result.content is not None
        assert result.usage_metadata is not None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_response_schema_with_title(self):
        """Test content generation with JSON schema that has a title."""
        openai_client = OpenAI()
        config = types.GenerateContentConfig(
            response_schema={
                "title": "PersonSchema",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            }
        )
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Return structured data")]
                )
            ],
            config=config,
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        call_args = mock_client.responses.create.call_args[1]
        # Responses API uses 'text' parameter with 'format'
        assert "text" in call_args
        format_obj = call_args["text"]["format"]
        # Responses API has a flatter structure - name and schema directly in format
        assert format_obj["name"] == "PersonSchema"
        assert "schema" in format_obj

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_response_schema_already_formatted(self):
        """Test content generation with response_schema already in OpenAI format."""
        openai_client = OpenAI()
        config = types.GenerateContentConfig(
            response_schema={
                "name": "MySchema",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            }
        )
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Return structured data")]
                )
            ],
            config=config,
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        call_args = mock_client.responses.create.call_args[1]
        # Responses API uses 'text' parameter with 'format'
        assert "text" in call_args
        format_obj = call_args["text"]["format"]
        # Responses API has a flatter structure - name and schema directly in format
        assert format_obj["name"] == "MySchema"
        assert "schema" in format_obj

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_tool_calls_and_responses(self):
        """Test content generation with tool calls followed by tool responses."""
        openai_client = OpenAI()
        # Create a request with assistant message (tool calls) and tool responses
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id="call_123",
                                name="get_weather",
                                args={"location": "SF"},
                            )
                        )
                    ],
                ),
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id="call_123", response={"temp": 72}
                            )
                        )
                    ],
                ),
            ],
        )

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        # Verify the message sequence
        call_args = mock_client.responses.create.call_args[1]
        input_items = call_args["input"]  # Responses API uses 'input' instead of 'messages'
        # Responses API extracts tool_calls to separate function_call items
        # First item should be the assistant message (without tool_calls)
        assert input_items[0]["type"] == "message"
        assert input_items[0]["role"] == "assistant"
        # Second item should be the function_call item
        assert input_items[1]["type"] == "function_call"
        assert input_items[1]["call_id"] == "call_123"
        assert input_items[1]["name"] == "get_weather"
        # Third item should be the function_call_output (tool response)
        assert input_items[2]["type"] == "function_call_output"
        assert input_items[2]["call_id"] == "call_123"

    @pytest.mark.asyncio
    async def test_content_to_openai_message_with_function_response_and_regular_parts(self):
        """Test content with both function response and regular parts."""
        content = types.Content(
            role="user",
            parts=[
                types.Part(text="Here's the result:"),
                types.Part(
                    function_response=types.FunctionResponse(
                        id="call_123", response={"result": "success"}
                    )
                ),
            ],
        )
        messages = await content_to_openai_message(content)
        # Should return list with regular message and tool message
        assert isinstance(messages, list)
        assert len(messages) == 2
        # First should be regular message
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) > 0
        # Content should use input_text for user role
        assert messages[0]["content"][0]["type"] == "input_text"
        # Second should be tool message
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "call_123"

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_generate_content_async_with_pydantic_model_schema(self):
        """Test content generation with Pydantic model as response schema.
        
        Pydantic models use create() with text format (strict schema)
        instead of parse(). The schema is extracted and made strict
        (additionalProperties: false) before passing to create().
        """
        from pydantic import BaseModel, Field

        class PersonModel(BaseModel):
            name: str = Field(description="Person's name")
            age: int = Field(description="Person's age")

        openai_client = OpenAI()
        config = types.GenerateContentConfig(response_schema=PersonModel)
        llm_request = LlmRequest(
            contents=[
                types.Content(
                    role="user", parts=[types.Part(text="Return a person")]
                )
            ],
            config=config,
        )

        mock_client = AsyncMock()
        # create() returns a response with structured output
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.output_text = '{"name": "John", "age": 30}'
        mock_response.usage = None
        mock_response.incomplete_details = None
        mock_client.responses.create = AsyncMock(
            return_value=mock_response
        )

        with patch.object(
            openai_client, "_get_openai_client", return_value=mock_client
        ):
            async for _ in openai_client.generate_content_async(
                llm_request, stream=False
            ):
                pass

        # Verify create() was called (not parse())
        assert mock_client.responses.create.called
        
        call_args = mock_client.responses.create.call_args[1]
        # create() uses text parameter with format containing strict JSON schema
        assert "text" in call_args
        assert "format" in call_args["text"]
        format_obj = call_args["text"]["format"]
        assert format_obj["type"] == "json_schema"
        # Responses API has a flatter structure - schema and strict directly in format
        assert format_obj["strict"] == True
        assert "schema" in format_obj
        schema = format_obj["schema"]
        # Schema should have additionalProperties: false
        assert "additionalProperties" in schema
        assert schema["additionalProperties"] == False
        # Should have properties from Pydantic model
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

