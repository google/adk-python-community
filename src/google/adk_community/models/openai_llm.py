"""OpenAI integration for GPT models in the ADK framework.

This file is an adapted version of your original plugin with optional Azure OpenAI support.
Key changes:
- Detect Azure when `azure_endpoint` (or an Azure-looking `base_url`) is set.
- Use `AsyncAzureOpenAI` when targeting Azure, `AsyncOpenAI` otherwise.
- Keep the existing Files API flow but note Azure-specific differences in comments.

References: Azure/OpenAI client usage patterns and Files API differences.
"""

from __future__ import annotations

import base64
import inspect
import json
import logging
import os
import re
import tempfile
from typing import Any
from typing import AsyncGenerator
from typing import Dict
from typing import Literal
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

import openai
from google.genai import types
from typing_extensions import override

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse

if TYPE_CHECKING:
    from google.adk.models.llm_request import LlmRequest

__all__ = ["OpenAI"]

logger = logging.getLogger(__name__)


def to_openai_role(role: Optional[str]) -> Literal["user", "assistant", "system"]:
    """Converts ADK role to OpenAI role."""
    if role == "model":
        return "assistant"
    elif role == "system":
        return "system"
    else:
        return "user"


def to_google_genai_finish_reason(
    openai_finish_reason: Optional[str],
) -> types.FinishReason:
    """Converts OpenAI finish reason to Google GenAI finish reason."""
    if openai_finish_reason == "stop":
        return "STOP"
    elif openai_finish_reason == "length":
        return "MAX_TOKENS"
    elif openai_finish_reason == "content_filter":
        return "SAFETY"
    elif openai_finish_reason == "tool_calls":
        return "STOP"
    else:
        return "FINISH_REASON_UNSPECIFIED"


def _is_image_part(part: types.Part) -> bool:
    """Checks if a part contains image data."""
    return (
        part.inline_data
        and part.inline_data.mime_type
        and part.inline_data.mime_type.startswith("image")
    )


async def part_to_openai_content(
    part: types.Part,
    openai_instance: Optional[OpenAI] = None,
) -> Union[Dict[str, Any], str]:
    """Converts ADK Part to OpenAI content format.

    OpenAI supports:
    - Text content
    - Images (base64 encoded)
    - PDF files (via Files API or base64 for vision models)
    - Other documents (via Files API or base64 for vision models)
    """
    if part.text:
        return {"type": "text", "text": part.text}
    elif part.function_call:
        # Function calls are handled separately in the message structure
        return {"type": "text", "text": f"Function call: {part.function_call.name}"}
    elif part.function_response:
        # Function responses are handled separately in the message structure
        return {
            "type": "text",
            "text": f"Function response: {part.function_response.response}",
        }
    elif part.inline_data or part.file_data:
        # Handle file data using the OpenAI instance's file handling
        if openai_instance:
            return await openai_instance._handle_file_data(part)
        else:
            # Fallback to simple base64 encoding if no OpenAI instance provided
            if part.inline_data:
                mime_type = part.inline_data.mime_type or "application/octet-stream"
                data = base64.b64encode(part.inline_data.data).decode()

                if mime_type.startswith("image/"):
                    return {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{data}"},
                    }
                else:
                    return {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{data}"},
                    }
            else:
                return {
                    "type": "text",
                    "text": f"[FILE REFERENCE: {part.file_data.display_name or 'unnamed'}]",
                }
    elif part.executable_code:
        return {"type": "text", "text": f"```python\n{part.executable_code.code}\n```"}
    elif part.code_execution_result:
        return {
            "type": "text",
            "text": f"Execution Result:\n```\n{part.code_execution_result.output}\n```",
        }
    else:
        # Fallback for unsupported parts
        logger.warning(f"Unsupported part type in OpenAI conversion: {type(part)}")
        return {"type": "text", "text": f"[UNSUPPORTED CONTENT: {str(part)[:100]}...]"}


async def content_to_openai_message(
    content: types.Content,
    openai_instance: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """Converts ADK Content to OpenAI message format."""
    message_content = []
    tool_calls = []
    tool_call_id = None

    for part in content.parts or []:
        if part.function_call:
            # Handle function calls
            tool_calls.append(
                {
                    "id": part.function_call.id or f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": part.function_call.name,
                        "arguments": (
                            json.dumps(part.function_call.args)
                            if part.function_call.args
                            else "{}"
                        ),
                    },
                }
            )
        elif part.function_response:
            # Handle function responses
            tool_call_id = part.function_response.id
            message_content.append(
                {
                    "type": "text",
                    "text": (
                        json.dumps(part.function_response.response)
                        if isinstance(part.function_response.response, dict)
                        else str(part.function_response.response)
                    ),
                }
            )
        else:
            # Handle regular content
            openai_content = await part_to_openai_content(part, openai_instance)
            if isinstance(openai_content, dict):
                message_content.append(openai_content)
            else:
                message_content.append({"type": "text", "text": openai_content})

    message = {"role": to_openai_role(content.role), "content": message_content}

    if tool_calls:
        message["tool_calls"] = tool_calls
    if tool_call_id:
        message["tool_call_id"] = tool_call_id

    return message


def function_declaration_to_openai_tool(
    function_declaration: types.FunctionDeclaration,
) -> Dict[str, Any]:
    """Converts ADK function declaration to OpenAI tool format."""
    properties = {}
    required_params = []

    if function_declaration.parameters:
        if function_declaration.parameters.properties:
            for key, value in function_declaration.parameters.properties.items():
                value_dict = value.model_dump(exclude_none=True)
                # Convert type string to OpenAI format (normalize to lowercase)
                if "type" in value_dict:
                    type_value = value_dict["type"]
                    original_type = type_value
                    type_str = None
                    
                    # Handle enum types (e.g., Type.STRING -> "STRING")
                    # Check if it's an enum instance (has name or value attribute)
                    if hasattr(type_value, "name"):
                        # It's an enum, get the name (e.g., Type.STRING.name -> "STRING")
                        type_str = type_value.name.lower()
                    elif hasattr(type_value, "value"):
                        # It's an enum, get the value
                        type_value = type_value.value
                        type_str = str(type_value).lower()
                    
                    # If not an enum, convert to string and handle enum format like "Type.STRING"
                    if type_str is None:
                        type_str = str(type_value)
                        
                        # Handle enum format like "Type.STRING" - extract the part after the dot
                        if "." in type_str:
                            # Extract the part after the dot (e.g., "Type.STRING" -> "STRING")
                            type_str = type_str.split(".")[-1]
                        
                        # Normalize to lowercase
                        type_str = type_str.lower()
                    
                    # Map common type names
                    type_mapping = {
                        "string": "string",
                        "str": "string",
                        "number": "number",
                        "num": "number",
                        "integer": "integer",
                        "int": "integer",
                        "boolean": "boolean",
                        "bool": "boolean",
                        "array": "array",
                        "list": "array",
                        "object": "object",
                        "dict": "object",
                    }
                    
                    if type_str in type_mapping:
                        value_dict["type"] = type_mapping[type_str]
                    elif type_str in [
                        "string",
                        "number",
                        "integer",
                        "boolean",
                        "array",
                        "object",
                    ]:
                        value_dict["type"] = type_str
                    else:
                        # If type is not recognized, log a warning and default to string
                        logger.warning(
                            f"Unknown type '{original_type}' for parameter '{key}', defaulting to 'string'"
                        )
                        value_dict["type"] = "string"
                properties[key] = value_dict

        if function_declaration.parameters.required:
            required_params = function_declaration.parameters.required

    function_schema = {
        "name": function_declaration.name,
        "description": function_declaration.description or "",
        "parameters": {
            "type": "object",
            "properties": properties,
        },
    }

    if required_params:
        function_schema["parameters"]["required"] = required_params

    return {"type": "function", "function": function_schema}


def openai_response_to_llm_response(
    response: Any,
) -> LlmResponse:
    """Converts OpenAI response to ADK LlmResponse."""
    logger.info("Received response from OpenAI.")
    logger.debug(f"OpenAI response: {response}")

    # Extract content from response
    content_parts = []

    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        message = choice.message

        if hasattr(message, "content") and message.content:
            content_parts.append(types.Part(text=message.content))

        # Handle tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.type == "function":
                    function_args = {}
                    if tool_call.function.arguments:
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            function_args = {"arguments": tool_call.function.arguments}

                    content_parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                id=tool_call.id,
                                name=tool_call.function.name,
                                args=function_args,
                            )
                        )
                    )

    # Create content
    content = (
        types.Content(role="model", parts=content_parts) if content_parts else None
    )

    # Extract usage metadata
    usage_metadata = None
    if hasattr(response, "usage") and response.usage:
        usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=response.usage.prompt_tokens,
            candidates_token_count=response.usage.completion_tokens,
            total_token_count=response.usage.total_tokens,
        )

    # Extract finish reason
    finish_reason = None
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "finish_reason"):
            finish_reason = to_google_genai_finish_reason(choice.finish_reason)

    # Extract model name and system fingerprint if available
    model_name = None
    if hasattr(response, "model"):
        model_name = response.model
        logger.debug(f"Response from model: {model_name}")

    system_fingerprint = None
    if hasattr(response, "system_fingerprint"):
        system_fingerprint = response.system_fingerprint
        logger.debug(f"System fingerprint: {system_fingerprint}")

    # Extract logprobs if available (from choice)
    logprobs = None
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "logprobs") and choice.logprobs:
            logprobs = choice.logprobs
            logger.debug(f"Logprobs available: {logprobs}")

    return LlmResponse(
        content=content, usage_metadata=usage_metadata, finish_reason=finish_reason
    )


class OpenAI(BaseLlm):
    """Integration with OpenAI GPT models.

    Configuration is read from environment variables and the GenerateContentConfig:
    - API keys: OPENAI_API_KEY or AZURE_OPENAI_API_KEY
    - Base URL: OPENAI_BASE_URL (for OpenAI) or AZURE_OPENAI_ENDPOINT (for Azure)
    - Azure API version: AZURE_OPENAI_API_VERSION (defaults to "2024-02-15-preview")
    - max_tokens, temperature: Set via GenerateContentConfig in the request
    - Additional parameters: top_p, frequency_penalty, presence_penalty, logit_bias,
      seed, n, stop, logprobs, top_logprobs, user can be set via GenerateContentConfig
    - Structured outputs: Supported via response_format, response_schema, or response_mime_type
      in GenerateContentConfig
    - Tool choice: Configurable via tool_choice in GenerateContentConfig (none, auto, required, or function)

    Structured Output Support:
    - JSON mode: Set response_mime_type="application/json" or response_format={"type": "json_object"}
    - Structured outputs with schema: Set response_schema with a JSON schema dict or Schema object
      This will be converted to OpenAI's response_format with type "json_schema"

    Attributes:
      model: The name of the OpenAI model or (for Azure) the deployment name.
      use_files_api: Whether to use OpenAI's Files API for file uploads (default: False).
    """

    model: str = "gpt-4o"
    use_files_api: bool = False

    @classmethod
    @override
    def supported_models(cls) -> list[str]:
        """Provides the list of supported models.

        Returns:
          A list of supported OpenAI model patterns.
        """
        return [
            r"gpt-.*",
            r"o1-.*",
            r"dall-e-.*",
            r"tts-.*",
            r"whisper-.*",
        ]

    def _is_azure(self) -> bool:
        """Heuristic to decide whether we're talking to Azure OpenAI.

        Checks environment variables for Azure endpoints or Azure-looking URLs.
        """
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint:
            return True

        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url and (
            "openai.azure.com" in base_url
            or "services.ai.azure.com" in base_url
        ):
            return True
        return False

    def _get_openai_client(self) -> openai.AsyncOpenAI:
        """Creates and returns an OpenAI client suitable for the target endpoint.

        Configuration is read from environment variables:
        - OPENAI_API_KEY or AZURE_OPENAI_API_KEY for authentication
        - OPENAI_BASE_URL for custom OpenAI base URL
        - AZURE_OPENAI_ENDPOINT for Azure endpoint
        - AZURE_OPENAI_API_VERSION for Azure API version

        Returns an *async* client in all cases (either AsyncAzureOpenAI or AsyncOpenAI).
        """
        # Get API key from environment
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("AZURE_OPENAI_API_KEY")
        )

        # Azure path: use AsyncAzureOpenAI and provide azure_endpoint + api_version
        if self._is_azure():
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            # Fallback to OPENAI_BASE_URL if it looks like Azure
            if not azure_endpoint:
                base_url = os.getenv("OPENAI_BASE_URL")
                if base_url and (
                    "openai.azure.com" in base_url
                    or "services.ai.azure.com" in base_url
                ):
                    azure_endpoint = base_url

            azure_kwargs: Dict[str, Any] = {}
            if api_key:
                azure_kwargs["api_key"] = api_key
            if azure_endpoint:
                azure_kwargs["azure_endpoint"] = azure_endpoint

            # Get API version from environment or use default
            api_version = os.getenv(
                "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
            )
            azure_kwargs["api_version"] = api_version

            # The openai package exports AsyncAzureOpenAI for async calls.
            try:
                return openai.AsyncAzureOpenAI(**azure_kwargs)
            except AttributeError:
                # Fallback: older SDKs may not expose AsyncAzureOpenAI; try AzureOpenAI
                return openai.AzureOpenAI(**azure_kwargs)

        # Default (OpenAI) path
        client_kwargs: Dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key

        # Get base URL from environment if set
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url

        return openai.AsyncOpenAI(**client_kwargs)

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Sends a request to the OpenAI model.

        Detects Azure vs OpenAI automatically and uses the appropriate async client.
        """
        await self._preprocess_request(llm_request)
        self._maybe_append_user_content(llm_request)

        # Initialize OpenAI client
        client = self._get_openai_client()

        # Convert request to OpenAI format
        messages = []

        # Add system instruction if present
        if llm_request.config and llm_request.config.system_instruction:
            if isinstance(llm_request.config.system_instruction, str):
                messages.append(
                    {"role": "system", "content": llm_request.config.system_instruction}
                )

        # Convert contents to messages
        for content in llm_request.contents:
            message = await content_to_openai_message(content, self)
            messages.append(message)

        # Prepare tools if present
        tools = []
        if llm_request.config and llm_request.config.tools:
            for tool in llm_request.config.tools:
                if isinstance(tool, types.Tool) and tool.function_declarations:
                    for func_decl in tool.function_declarations:
                        openai_tool = function_declaration_to_openai_tool(func_decl)
                        tools.append(openai_tool)

        # Prepare request parameters
        # NOTE: For Azure, `model` should be the deployment name (the name you configured in the Azure portal).
        request_model = llm_request.model or self.model
        # Validate model name using supported_models patterns
        supported_patterns = self.supported_models()
        model_supported = any(
            re.match(pattern, request_model) for pattern in supported_patterns
        )
        if not model_supported:
            raise ValueError(
                f"Invalid model name '{request_model}'. Model must match one of the supported patterns: "
                f"{', '.join(supported_patterns)}."
            )
        request_params = {
            "model": request_model,
            "messages": messages,
            "stream": stream,
        }

        if tools:
            request_params["tools"] = tools
            # Support configurable tool_choice (none, auto, required, or specific function)
            tool_choice = "auto"  # Default
            if llm_request.config:
                tool_choice_config = getattr(
                    llm_request.config, "tool_choice", None
                )
                if tool_choice_config is not None:
                    tool_choice = tool_choice_config
            request_params["tool_choice"] = tool_choice

        # Get max_tokens and temperature from config
        if llm_request.config:
            if llm_request.config.max_output_tokens:
                request_params["max_tokens"] = llm_request.config.max_output_tokens

            if llm_request.config.temperature is not None:
                request_params["temperature"] = llm_request.config.temperature

            # Add additional OpenAI API parameters if available
            # Use getattr to safely check for attributes that may not exist in all config versions
            top_p = getattr(llm_request.config, "top_p", None)
            if top_p is not None:
                request_params["top_p"] = top_p

            frequency_penalty = getattr(
                llm_request.config, "frequency_penalty", None
            )
            if frequency_penalty is not None:
                request_params["frequency_penalty"] = frequency_penalty

            presence_penalty = getattr(llm_request.config, "presence_penalty", None)
            if presence_penalty is not None:
                request_params["presence_penalty"] = presence_penalty

            logit_bias = getattr(llm_request.config, "logit_bias", None)
            if logit_bias is not None:
                request_params["logit_bias"] = logit_bias

            seed = getattr(llm_request.config, "seed", None)
            if seed is not None:
                request_params["seed"] = seed

            n = getattr(llm_request.config, "n", None)
            if n is not None:
                request_params["n"] = n

            stop = getattr(llm_request.config, "stop", None)
            if stop is not None:
                # stop can be a string or list of strings
                request_params["stop"] = stop

            logprobs = getattr(llm_request.config, "logprobs", None)
            if logprobs is not None:
                request_params["logprobs"] = logprobs

            top_logprobs = getattr(llm_request.config, "top_logprobs", None)
            if top_logprobs is not None:
                request_params["top_logprobs"] = top_logprobs

            user = getattr(llm_request.config, "user", None)
            if user is not None:
                request_params["user"] = user

            # Handle structured output / response format
            # OpenAI supports two types of structured outputs:
            # 1. JSON mode: {"type": "json_object"}
            # 2. Structured outputs with schema: {"type": "json_schema", "json_schema": {...}}
            
            # Check for response_format in config (if available in ADK)
            # Use getattr with None default to safely check for attribute
            # Check it's actually a dict with expected structure to avoid MagicMock issues
            try:
                response_format_set = False
                # Try to get response_format from config
                response_format = getattr(llm_request.config, "response_format", None)
                # Check if it's a dict and has the expected structure (not a MagicMock)
                if (
                    response_format is not None
                    and isinstance(response_format, dict)
                    and "type" in response_format
                ):
                    request_params["response_format"] = response_format
                    response_format_set = True
                
                if not response_format_set:
                    # Check for response_schema (JSON schema for structured outputs)
                    response_schema = getattr(
                        llm_request.config, "response_schema", None
                    )
                    if response_schema is not None:
                        # Convert ADK schema to OpenAI JSON schema format
                        if isinstance(response_schema, dict):
                            request_params["response_format"] = {
                                "type": "json_schema",
                                "json_schema": response_schema,
                            }
                        else:
                            # If it's a Schema object, convert it
                            schema_dict = None
                            try:
                                # Check if it's a Pydantic model class (not an instance)
                                # First check if it's a class
                                is_pydantic_class = False
                                if inspect.isclass(response_schema):
                                    # Check if it has Pydantic-specific methods (model_json_schema or schema)
                                    # or if it's a subclass of BaseModel
                                    has_pydantic_methods = (
                                        hasattr(response_schema, "model_json_schema")
                                        or hasattr(response_schema, "schema")
                                    )
                                    # Check if any base class is BaseModel
                                    is_base_model_subclass = False
                                    try:
                                        for base in response_schema.__mro__:
                                            if hasattr(base, "__name__") and base.__name__ == "BaseModel":
                                                is_base_model_subclass = True
                                                break
                                    except (AttributeError, TypeError):
                                        pass
                                    
                                    is_pydantic_class = has_pydantic_methods or is_base_model_subclass
                                
                                if is_pydantic_class:
                                    # It's a Pydantic model class, get JSON schema from the class
                                    # Try model_json_schema (Pydantic v2)
                                    if hasattr(response_schema, "model_json_schema"):
                                        schema_dict = response_schema.model_json_schema()
                                    # Try schema() method (Pydantic v1)
                                    elif hasattr(response_schema, "schema"):
                                        schema_dict = response_schema.schema()
                                    else:
                                        # Fallback: try to instantiate and get schema
                                        # This might not work if required fields are missing
                                        try:
                                            # Create a minimal instance if possible
                                            instance = response_schema()
                                            if hasattr(instance, "model_json_schema"):
                                                schema_dict = instance.model_json_schema()
                                            elif hasattr(instance, "schema"):
                                                schema_dict = instance.schema()
                                        except Exception:
                                            pass
                                else:
                                    # It's an instance or other object
                                    # Try model_dump (Pydantic v2 instance)
                                    if hasattr(response_schema, "model_dump"):
                                        schema_dict = response_schema.model_dump(
                                            exclude_none=True
                                        )
                                    # Try dict() method (Pydantic v1 instance)
                                    elif hasattr(response_schema, "dict"):
                                        schema_dict = response_schema.dict(
                                            exclude_none=True
                                        )
                                    # Try model_json_schema (Pydantic v2 instance)
                                    elif hasattr(response_schema, "model_json_schema"):
                                        schema_dict = response_schema.model_json_schema()
                                    # Try schema() method (Pydantic v1 instance)
                                    elif hasattr(response_schema, "schema"):
                                        schema_dict = response_schema.schema()
                                    # Try __dict__ attribute
                                    elif hasattr(response_schema, "__dict__"):
                                        schema_dict = {
                                            k: v
                                            for k, v in response_schema.__dict__.items()
                                            if v is not None
                                        }
                                    # Try converting to dict directly
                                    else:
                                        schema_dict = dict(response_schema)
                            except (AttributeError, TypeError, ValueError) as e:
                                logger.warning(
                                    f"Could not convert response_schema to OpenAI format: {e}. "
                                    f"Schema type: {type(response_schema)}, "
                                    f"Schema value: {response_schema}"
                                )
                                schema_dict = None
                            
                            if schema_dict is not None:
                                request_params["response_format"] = {
                                    "type": "json_schema",
                                    "json_schema": schema_dict,
                                }
                    else:
                        # Also check for response_mime_type which might indicate JSON mode
                        response_mime_type = getattr(
                            llm_request.config, "response_mime_type", None
                        )
                        if (
                            response_mime_type is not None
                            and isinstance(response_mime_type, str)
                            and response_mime_type == "application/json"
                        ):
                            request_params["response_format"] = {"type": "json_object"}
                        elif response_mime_type and isinstance(
                            response_mime_type, str
                        ):
                            # For other structured formats, could be extended
                            logger.debug(
                                f"Response MIME type {response_mime_type} "
                                "not directly mapped to OpenAI response_format"
                            )
            except Exception as e:
                # If there's any issue accessing config attributes, log and continue
                logger.debug(f"Error checking structured output config: {e}")
                # Re-raise to see what's happening in tests
                if os.getenv("DEBUG_STRUCTURED_OUTPUT"):
                    raise

        logger.info(
            "Sending request to OpenAI, model: %s, stream: %s",
            request_params["model"],
            stream,
        )
        logger.debug(f"OpenAI request: {request_params}")

        try:
            if stream:
                # Handle streaming response
                stream_response = await client.chat.completions.create(**request_params)

                # Accumulate content and tool calls across chunks
                accumulated_text = ""
                accumulated_tool_calls: Dict[str, Dict[str, Any]] = {}
                finish_reason = None
                usage_metadata = None
                model_name = None
                system_fingerprint = None

                async for chunk in stream_response:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        delta = choice.delta if hasattr(choice, "delta") else None

                        # Track model name and system fingerprint from first chunk
                        if model_name is None and hasattr(chunk, "model"):
                            model_name = chunk.model
                        if (
                            system_fingerprint is None
                            and hasattr(chunk, "system_fingerprint")
                        ):
                            system_fingerprint = chunk.system_fingerprint

                        # Handle text content deltas
                        if delta and hasattr(delta, "content") and delta.content:
                            accumulated_text += delta.content
                            # Create partial response with accumulated text
                            partial_content = types.Content(
                                role="model",
                                parts=[types.Part(text=accumulated_text)],
                            )
                            yield LlmResponse(
                                content=partial_content,
                                partial=True,
                                turn_complete=False,
                            )

                        # Handle tool call deltas
                        if delta and hasattr(delta, "tool_calls") and delta.tool_calls:
                            for tool_call_delta in delta.tool_calls:
                                tool_call_id = (
                                    tool_call_delta.id
                                    if hasattr(tool_call_delta, "id")
                                    else None
                                )
                                if tool_call_id:
                                    if tool_call_id not in accumulated_tool_calls:
                                        accumulated_tool_calls[tool_call_id] = {
                                            "id": tool_call_id,
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""},
                                        }
                                    tool_call = accumulated_tool_calls[tool_call_id]

                                    # Update function name if present
                                    if (
                                        hasattr(tool_call_delta, "function")
                                        and hasattr(tool_call_delta.function, "name")
                                        and tool_call_delta.function.name
                                    ):
                                        tool_call["function"]["name"] = (
                                            tool_call_delta.function.name
                                        )

                                    # Accumulate function arguments
                                    if (
                                        hasattr(tool_call_delta, "function")
                                        and hasattr(
                                            tool_call_delta.function, "arguments"
                                        )
                                        and tool_call_delta.function.arguments
                                    ):
                                        tool_call["function"]["arguments"] += (
                                            tool_call_delta.function.arguments
                                        )

                        # Track finish reason
                        if hasattr(choice, "finish_reason") and choice.finish_reason:
                            finish_reason = to_google_genai_finish_reason(
                                choice.finish_reason
                            )

                    # Track usage metadata if available
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_metadata = types.GenerateContentResponseUsageMetadata(
                            prompt_token_count=chunk.usage.prompt_tokens or 0,
                            candidates_token_count=chunk.usage.completion_tokens or 0,
                            total_token_count=chunk.usage.total_tokens or 0,
                        )

                # Build final complete response with all accumulated data
                final_parts = []
                if accumulated_text:
                    final_parts.append(types.Part(text=accumulated_text))

                # Add accumulated tool calls
                if accumulated_tool_calls:
                    for tool_call_data in accumulated_tool_calls.values():
                        function_args = {}
                        if tool_call_data["function"]["arguments"]:
                            try:
                                function_args = json.loads(
                                    tool_call_data["function"]["arguments"]
                                )
                            except json.JSONDecodeError:
                                function_args = {
                                    "arguments": tool_call_data["function"]["arguments"]
                                }

                        final_parts.append(
                            types.Part(
                                function_call=types.FunctionCall(
                                    id=tool_call_data["id"],
                                    name=tool_call_data["function"]["name"],
                                    args=function_args,
                                )
                            )
                        )

                final_content = (
                    types.Content(role="model", parts=final_parts)
                    if final_parts
                    else types.Content(role="model", parts=[])
                )

                # Create final response with all metadata
                final_response = LlmResponse(
                    content=final_content,
                    usage_metadata=usage_metadata,
                    finish_reason=finish_reason,
                    turn_complete=True,
                )

                # Add model and system_fingerprint if available (as metadata)
                # Note: LlmResponse may not have these fields directly, but we can log them
                if model_name:
                    logger.debug(f"Streaming response from model: {model_name}")
                if system_fingerprint:
                    logger.debug(f"System fingerprint: {system_fingerprint}")

                yield final_response
            else:
                # Handle non-streaming response
                response = await client.chat.completions.create(**request_params)
                llm_response = openai_response_to_llm_response(response)
                yield llm_response

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            yield LlmResponse(error_code="OPENAI_API_ERROR", error_message=str(e))

    def _get_file_extension(self, mime_type: str) -> str:
        """Get file extension from MIME type."""
        mime_to_ext = {
            "application/pdf": ".pdf",
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "text/plain": ".txt",
            "text/markdown": ".md",
            "application/json": ".json",
            "application/msword": ".doc",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        }
        return mime_to_ext.get(mime_type, ".bin")

    async def _upload_file_to_openai(
        self, file_data: bytes, mime_type: str, display_name: Optional[str] = None
    ) -> str:
        """Upload a file to OpenAI's Files API and return the file ID.

        This is a fully async implementation that uploads files to OpenAI's Files API.
        Azure: Files API exists but may differ in accepted purposes or API-version;
        consult Azure docs if you plan to upload files to an Azure OpenAI resource.

        Args:
            file_data: The file data as bytes.
            mime_type: The MIME type of the file.
            display_name: Optional display name for the file.

        Returns:
            The file ID from OpenAI.

        Raises:
            ValueError: If use_files_api is False.
        """
        if not self.use_files_api:
            raise ValueError(
                "Files API is disabled. Set use_files_api=True to enable file uploads."
            )

        client = self._get_openai_client()

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=self._get_file_extension(mime_type)
        ) as temp_file:
            temp_file.write(file_data)
            temp_file_path = temp_file.name

        try:
            # Use the client.files.create API (supported both by OpenAI and Azure OpenAI REST endpoints)
            with open(temp_file_path, "rb") as f:
                uploaded_file = await client.files.create(
                    file=f, purpose="assistants"
                )

            logger.info(
                f"Uploaded file to OpenAI: {uploaded_file.id} ({display_name or 'unnamed'})"
            )
            return uploaded_file.id

        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    async def _handle_file_data(self, part: types.Part) -> Union[Dict[str, Any], str]:
        """Handle file data by uploading to OpenAI Files API or converting to base64."""
        if part.inline_data:
            # Handle inline data
            mime_type = part.inline_data.mime_type or "application/octet-stream"
            data = part.inline_data.data
            display_name = part.inline_data.display_name

            if self.use_files_api and mime_type in [
                "application/pdf",
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ]:
                # Upload documents to Files API
                try:
                    file_id = await self._upload_file_to_openai(
                        data, mime_type, display_name
                    )
                    return {
                        "type": "text",
                        "text": f"[File uploaded to OpenAI: {display_name or 'unnamed file'} (ID: {file_id})]",
                    }
                except Exception as e:
                    logger.warning(
                        f"Failed to upload file to OpenAI Files API: {e}. Falling back to base64 encoding."
                    )

            # For images or when Files API is disabled, use base64 encoding
            if mime_type.startswith("image/"):
                data_b64 = base64.b64encode(data).decode()
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{data_b64}"},
                }
            else:
                # For other file types, convert to base64 for vision models
                data_b64 = base64.b64encode(data).decode()
                return {
                    "type": "image_url",  # OpenAI treats documents as images for vision models
                    "image_url": {"url": f"data:{mime_type};base64,{data_b64}"},
                }

        elif part.file_data:
            # Handle file references (URIs)
            file_uri = part.file_data.file_uri or "unknown"
            display_name = part.file_data.display_name or "unnamed file"
            mime_type = part.file_data.mime_type or "unknown"

            logger.warning(
                f"OpenAI Chat API does not support file references. "
                f"File '{display_name}' ({file_uri}) converted to text description. "
                f"Consider uploading the file directly or using OpenAI's Files API."
            )

            return {
                "type": "text",
                "text": f"[FILE REFERENCE: {display_name}]\n"
                f"URI: {file_uri}\n"
                f"Type: {mime_type}\n"
                f"Note: OpenAI Chat API does not support file references. "
                f"Consider uploading the file directly or using OpenAI's Files API.",
            }

        return {"type": "text", "text": str(part)}

    async def _preprocess_request(self, llm_request: LlmRequest):
        """Preprocesses the request before sending to OpenAI."""
        # Set model if not specified
        if not llm_request.model:
            llm_request.model = self.model
