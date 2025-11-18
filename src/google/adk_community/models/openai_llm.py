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
from typing import List
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
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Converts ADK Content to OpenAI message format.
    
    Returns a single message dict, or a list of messages if there are multiple
    function responses that need separate tool messages.
    """
    message_content = []
    tool_calls = []
    function_responses = []  # Collect all function responses
    regular_parts = []  # Collect regular content parts

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
            # Collect function responses - each needs its own tool message
            function_responses.append(part.function_response)
        else:
            # Collect regular content parts
            regular_parts.append(part)

    # If we have function responses, create tool messages for each
    if function_responses:
        tool_messages = []
        for func_response in function_responses:
            # For tool messages, content should be a string
            response_text = (
                json.dumps(func_response.response)
                if isinstance(func_response.response, dict)
                else str(func_response.response)
            )
            
            # Validate that we have a tool_call_id
            if not func_response.id:
                logger.warning(
                    f"Function response missing id, cannot create tool message. "
                    f"Response: {response_text[:100]}"
                )
                continue
            
            tool_message = {
                "role": "tool",
                "content": response_text,
                "tool_call_id": func_response.id,
            }
            tool_messages.append(tool_message)
            logger.debug(
                f"Created tool message for tool_call_id={func_response.id}, "
                f"content_length={len(response_text)}"
            )
        
        # If there are also regular parts, we need to create a regular message too
        # But typically function responses are in separate Content objects
        if regular_parts:
            # Process regular parts
            for part in regular_parts:
                openai_content = await part_to_openai_content(part, openai_instance)
                if isinstance(openai_content, dict):
                    message_content.append(openai_content)
                else:
                    message_content.append({"type": "text", "text": openai_content})
            
            # Create regular message and prepend it to tool messages
            regular_message = {
                "role": to_openai_role(content.role),
                "content": message_content,
            }
            if tool_calls:
                regular_message["tool_calls"] = tool_calls
            
            return [regular_message] + tool_messages
        else:
            # Only function responses, return list of tool messages
            return tool_messages
    else:
        # No function responses, create regular message
        for part in regular_parts:
            openai_content = await part_to_openai_content(part, openai_instance)
            if isinstance(openai_content, dict):
                message_content.append(openai_content)
            else:
                message_content.append({"type": "text", "text": openai_content})
        
        message = {"role": to_openai_role(content.role), "content": message_content}
        
        if tool_calls:
            message["tool_calls"] = tool_calls
        
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
    - max_tokens/max_completion_tokens (mapped from max_output_tokens, auto-selected based on model),
      temperature: Set via GenerateContentConfig in the request
    - Additional parameters: top_p, frequency_penalty, presence_penalty, seed,
      candidate_count (maps to OpenAI's n), stop_sequences, logprobs can be set via GenerateContentConfig
    - Note: logit_bias, top_logprobs, and user are OpenAI-specific and not supported in Google GenAI
    - Structured outputs: Supported via response_schema or response_mime_type in GenerateContentConfig
    - Tool choice: Configurable via tool_config.function_calling_config.mode in GenerateContentConfig
      (uses FunctionCallingConfigMode: NONE, AUTO, ANY, VALIDATED)

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

    def _requires_max_completion_tokens(self, model: str) -> bool:
        """Check if the model requires max_completion_tokens instead of max_tokens.
        
        o1-series models (o1-preview, o1-mini, etc.) require max_completion_tokens.
        All other models use max_tokens.
        
        Args:
            model: The model name to check
            
        Returns:
            True if the model requires max_completion_tokens, False otherwise
        """
        return model.startswith("o1-") if model else False

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
        self._preprocess_request(llm_request)
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
        # Track tool_call_ids from assistant messages to ensure they're all responded to
        pending_tool_call_ids = set()
        
        for content in llm_request.contents:
            # content_to_openai_message may return a list if there are multiple function responses
            message_or_messages = await content_to_openai_message(content, self)
            
            if isinstance(message_or_messages, list):
                for msg in message_or_messages:
                    messages.append(msg)
                    # Track tool calls and responses
                    if msg.get("role") == "assistant" and "tool_calls" in msg:
                        # Add all tool_call_ids to pending set
                        for tool_call in msg["tool_calls"]:
                            pending_tool_call_ids.add(tool_call["id"])
                    elif msg.get("role") == "tool" and "tool_call_id" in msg:
                        # Remove from pending when we get a tool response
                        pending_tool_call_ids.discard(msg["tool_call_id"])
            else:
                messages.append(message_or_messages)
                # Track tool calls and responses
                if message_or_messages.get("role") == "assistant" and "tool_calls" in message_or_messages:
                    # Add all tool_call_ids to pending set
                    for tool_call in message_or_messages["tool_calls"]:
                        pending_tool_call_ids.add(tool_call["id"])
                elif message_or_messages.get("role") == "tool" and "tool_call_id" in message_or_messages:
                    # Remove from pending when we get a tool response
                    pending_tool_call_ids.discard(message_or_messages["tool_call_id"])
        
        # Validate message sequence: after assistant messages with tool_calls, 
        # the next messages must be tool messages
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                tool_call_ids = {tc["id"] for tc in msg["tool_calls"]}
                # Check that the following messages are tool messages for these tool_call_ids
                j = i + 1
                found_tool_call_ids = set()
                while j < len(messages) and messages[j].get("role") == "tool":
                    tool_call_id = messages[j].get("tool_call_id")
                    if tool_call_id:
                        found_tool_call_ids.add(tool_call_id)
                    j += 1
                
                missing_ids = tool_call_ids - found_tool_call_ids
                if missing_ids:
                    logger.warning(
                        f"Assistant message at index {i} has tool_calls, but missing tool responses "
                        f"for tool_call_ids: {missing_ids}. Found responses for: {found_tool_call_ids}"
                    )
                    # Log the actual message sequence for debugging
                    logger.debug(
                        f"Message sequence around index {i}: "
                        f"{[{'idx': idx, 'role': m.get('role'), 'tool_call_id': m.get('tool_call_id'), 'has_tool_calls': 'tool_calls' in m} for idx, m in enumerate(messages[max(0, i-1):min(len(messages), i+5)])]}"
                    )
        
        # Log warning if there are pending tool calls without responses
        if pending_tool_call_ids:
            logger.warning(
                f"Found {len(pending_tool_call_ids)} tool call(s) without responses: {pending_tool_call_ids}. "
                "This may cause OpenAI API errors."
            )

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
            # Support configurable tool_choice using Google GenAI types only
            # Use tool_config.function_calling_config.mode (strict Google GenAI format)
            tool_choice = "auto"  # Default
            if llm_request.config:
                # Check for tool_config (Google GenAI format: ToolConfig containing FunctionCallingConfig)
                tool_config = getattr(llm_request.config, "tool_config", None)
                if tool_config is not None:
                    # Extract function_calling_config from ToolConfig
                    function_calling_config = getattr(tool_config, "function_calling_config", None)
                    if function_calling_config is not None:
                        # Extract mode from FunctionCallingConfig
                        if hasattr(function_calling_config, "mode") and function_calling_config.mode is not None:
                            mode = function_calling_config.mode
                            # Get mode name/value for comparison (handle both enum and string)
                            mode_name = None
                            if hasattr(mode, "name"):
                                mode_name = mode.name
                            elif hasattr(mode, "value"):
                                mode_name = str(mode.value)
                            elif isinstance(mode, str):
                                mode_name = mode.upper()
                            
                            logger.debug(f"Function calling config mode: {mode}, mode_name: {mode_name}")
                            
                            # Map Google GenAI FunctionCallingConfigMode to OpenAI tool_choice
                            if mode_name == "NONE" or mode == types.FunctionCallingConfigMode.NONE:
                                tool_choice = "none"
                            elif mode_name == "AUTO" or mode == types.FunctionCallingConfigMode.AUTO:
                                tool_choice = "auto"
                            elif mode_name in ("ANY", "VALIDATED") or mode in (types.FunctionCallingConfigMode.ANY, types.FunctionCallingConfigMode.VALIDATED):
                                tool_choice = "required"  # ANY/VALIDATED means tools should be called, similar to required
                            
                            # Check for allowed_function_names for specific function selection
                            if hasattr(function_calling_config, "allowed_function_names") and function_calling_config.allowed_function_names:
                                # OpenAI supports function-specific tool_choice as a dict
                                if len(function_calling_config.allowed_function_names) == 1:
                                    tool_choice = {
                                        "type": "function",
                                        "function": {"name": function_calling_config.allowed_function_names[0]}
                                    }
                        else:
                            logger.debug("Function calling config found but mode is None or missing")
                    else:
                        logger.debug("Tool config found but function_calling_config is None or missing")
                # No fallback - only use Google GenAI tool_config format
            request_params["tool_choice"] = tool_choice
            # Force parallel_tool_calls to True - all modern models support parallel tool calls
            # This parameter exists only for backward compatibility with old API specs
            request_params["parallel_tool_calls"] = True

        # Get max_tokens/max_completion_tokens and temperature from config
        if llm_request.config:
            if llm_request.config.max_output_tokens:
                # Map Google GenAI max_output_tokens to OpenAI's parameter
                # o1-series models require max_completion_tokens, others use max_tokens
                if self._requires_max_completion_tokens(request_model):
                    request_params["max_completion_tokens"] = llm_request.config.max_output_tokens
                else:
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

            # logit_bias is OpenAI-specific and not available in Google GenAI - removed
            
            seed = getattr(llm_request.config, "seed", None)
            if seed is not None:
                request_params["seed"] = seed

            # Use candidate_count (Google GenAI format) instead of n
            candidate_count = getattr(llm_request.config, "candidate_count", None)
            if candidate_count is not None:
                # Map Google GenAI candidate_count to OpenAI's n parameter
                request_params["n"] = candidate_count

            # Use stop_sequences (Google GenAI format) - strict Google GenAI types only
            stop_sequences = getattr(llm_request.config, "stop_sequences", None)
            if stop_sequences is not None:
                # stop_sequences can be a list of strings
                request_params["stop"] = stop_sequences

            logprobs = getattr(llm_request.config, "logprobs", None)
            if logprobs is not None:
                request_params["logprobs"] = logprobs

            # top_logprobs is OpenAI-specific and not available in Google GenAI - removed
            # response_logprobs exists in Google GenAI but is a boolean flag, not equivalent to top_logprobs
            
            # user is OpenAI-specific and not available in Google GenAI - removed

            # Handle structured output / response format
            # Priority: Google GenAI types (response_schema, response_mime_type) first
            # Then check for response_format (OpenAI-specific, for direct API compatibility)
            # OpenAI supports two types of structured outputs:
            # 1. JSON mode: {"type": "json_object"}
            # 2. Structured outputs with schema: {"type": "json_schema", "json_schema": {...}}
            
            try:
                response_format_set = False
                # First, check for Google GenAI response_schema or response_mime_type
                # (These are handled below in the response_schema section)
                
                # Then check for response_format (OpenAI-specific, for direct API compatibility)
                # Use getattr with None default to safely check for attribute
                # Check it's actually a dict with expected structure to avoid MagicMock issues
                if not response_format_set:
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
                        # OpenAI requires: {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}
                        schema_name = None
                        schema_dict = None
                        
                        if isinstance(response_schema, dict):
                            # If it's already a dict, check if it's already in OpenAI format
                            if "name" in response_schema and "schema" in response_schema:
                                # Already in OpenAI format
                                request_params["response_format"] = {
                                    "type": "json_schema",
                                    "json_schema": response_schema,
                                }
                            else:
                                # It's a raw JSON schema dict, need to wrap it
                                schema_dict = response_schema
                                schema_name = response_schema.get("title", "ResponseSchema")
                        else:
                            # If it's a Schema object, convert it
                            try:
                                # Get schema name from the class/object
                                if inspect.isclass(response_schema):
                                    schema_name = getattr(response_schema, "__name__", "ResponseSchema")
                                else:
                                    schema_name = getattr(
                                        response_schema, "__class__", type(response_schema)
                                    ).__name__
                                    if schema_name == "dict":
                                        schema_name = "ResponseSchema"
                                
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
                            
                        # Wrap schema in OpenAI's required format
                        if schema_dict is not None:
                            # Use schema title if available, otherwise use the name we extracted
                            if isinstance(schema_dict, dict) and "title" in schema_dict:
                                schema_name = schema_dict.get("title", schema_name or "ResponseSchema")
                            
                            request_params["response_format"] = {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": schema_name or "ResponseSchema",
                                    "schema": schema_dict,
                                },
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
        # Log message sequence for debugging tool call issues
        if messages:
            logger.debug(f"Message sequence ({len(messages)} messages):")
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                tool_calls = msg.get("tool_calls")
                tool_call_id = msg.get("tool_call_id")
                logger.debug(
                    f"  [{i}] role={role}, "
                    f"tool_calls={len(tool_calls) if tool_calls else 0}, "
                    f"tool_call_id={tool_call_id}"
                )
                if tool_calls:
                    for tc in tool_calls:
                        logger.debug(f"    tool_call: id={tc.get('id')}, name={tc.get('function', {}).get('name')}")
        logger.debug(f"OpenAI request params (excluding messages): { {k: v for k, v in request_params.items() if k != 'messages'} }")

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

    def _preprocess_request(self, llm_request: LlmRequest):
        """Preprocesses the request before sending to OpenAI."""
        # Set model if not specified
        if not llm_request.model:
            llm_request.model = self.model
