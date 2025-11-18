"""OpenAI integration for GPT models in the ADK framework.

This file uses OpenAI's Responses API which supports file inputs including PDFs
and other document formats. The Responses API provides a unified interface for
assistants functionality.

Key features:
- Uses Responses API (client.responses.create)
- Supports file inputs via Files API for documents (PDFs, Word docs, etc.)
- Detects Azure when `azure_endpoint` (or an Azure-looking `base_url`) is set
- Use `AsyncAzureOpenAI` when targeting Azure, `AsyncOpenAI` otherwise

References: OpenAI Responses API, Azure/OpenAI client usage patterns.
"""

from __future__ import annotations

import base64
import inspect
import json
import logging
import os
import re
import tempfile
from urllib.parse import urlparse
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
            tool_call_id = part.function_call.id
            if not tool_call_id:
                tool_call_id = f"call_{len(tool_calls)}"
                logger.warning(
                    "Function call part is missing an 'id'. A temporary one '%s' was generated. "
                    "This may cause issues if a function_response needs to reference it.",
                    tool_call_id,
                )
            tool_calls.append(
                {
                    "id": tool_call_id,
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
                func_response.response
                if isinstance(func_response.response, str)
                else json.dumps(func_response.response)
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


def _convert_param_type(type_value: Any) -> str:
    """Converts a parameter type value to OpenAI format string.
    
    Handles enum types, string representations, and maps common type names
    to OpenAI-compatible type strings.
    
    Args:
        type_value: The type value to convert (can be enum, string, etc.)
        
    Returns:
        A lowercase string representing the OpenAI-compatible type
    """
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
        return type_mapping[type_str]
    elif type_str in [
        "string",
        "number",
        "integer",
        "boolean",
        "array",
        "object",
    ]:
        return type_str
    else:
        # If type is not recognized, log a warning and default to string
        logger.warning(
            f"Unknown type '{original_type}', defaulting to 'string'"
        )
        return "string"


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
                    value_dict["type"] = _convert_param_type(value_dict["type"])
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


async def _fetch_url_content(url: str, timeout: float = 30.0) -> Optional[bytes]:
    """Fetches content from a URL asynchronously.
    
    Args:
        url: The URL to fetch content from
        timeout: Timeout in seconds for the HTTP request (default: 30.0)
        
    Returns:
        The content as bytes, or None if fetching failed
    """
    try:
        # Try to use aiohttp if available (optional dependency)
        try:
            import aiohttp  # type: ignore[import-untyped]
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    else:
                        logger.warning(f"Failed to fetch URL {url}: HTTP {resp.status}")
                        return None
        except ImportError:
            # Fallback to httpx if aiohttp is not available
            try:
                import httpx
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return resp.content
                    else:
                        logger.warning(f"Failed to fetch URL {url}: HTTP {resp.status_code}")
                        return None
            except ImportError:
                # If neither is available, log warning and return None
                logger.warning(
                    "Neither aiohttp nor httpx is available. "
                    "Cannot fetch image URLs. Install aiohttp or httpx to enable image URL fetching."
                )
                return None
    except Exception as e:
        logger.warning(f"Error fetching URL {url}: {e}")
        return None


async def _convert_media_content_to_part(
    content_part: Any,
    timeout: float = 30.0,
) -> Optional[types.Part]:
    """Converts OpenAI media content (image_url, etc.) to Google GenAI Part.
    
    Args:
        content_part: The OpenAI content part with media content
        timeout: Timeout in seconds for fetching external URLs (default: 30.0)
        
    Returns:
        A Google GenAI Part with inline_data, or None if conversion failed
    """
    if not hasattr(content_part, "type"):
        return None
    
    # Handle image_url content
    if content_part.type == "image_url" and hasattr(content_part, "image_url"):
        image_url_obj = content_part.image_url
        url = None
        
        # Extract URL from image_url object (can be string or object with url field)
        if isinstance(image_url_obj, str):
            url = image_url_obj
        elif hasattr(image_url_obj, "url"):
            url = image_url_obj.url
        elif isinstance(image_url_obj, dict) and "url" in image_url_obj:
            url = image_url_obj["url"]
        
        if not url:
            logger.warning("image_url content part has no valid URL")
            return None
        
        # Check if it's a data URI
        if url.startswith("data:"):
            # Parse data URI: data:[<mediatype>][;base64],<data>
            try:
                header, data = url.split(",", 1)
                if ";base64" in header:
                    # Base64 encoded data
                    image_data = base64.b64decode(data)
                    # Extract mime type from header
                    mime_type = "image/png"  # Default
                    if "data:" in header:
                        mime_part = header.split("data:")[1].split(";")[0].strip()
                        if mime_part:
                            mime_type = mime_part
                    return types.Part(
                        inline_data=types.Blob(mime_type=mime_type, data=image_data)
                    )
                else:
                    # URL-encoded data (less common)
                    from urllib.parse import unquote
                    image_data = unquote(data).encode("utf-8")
                    mime_type = "image/png"  # Default
                    if "data:" in header:
                        mime_part = header.split("data:")[1].strip()
                        if mime_part:
                            mime_type = mime_part
                    return types.Part(
                        inline_data=types.Blob(mime_type=mime_type, data=image_data)
                    )
            except Exception as e:
                logger.warning(f"Failed to parse data URI: {e}")
                return None
        else:
            # It's an actual URL - fetch the content
            image_data = await _fetch_url_content(url, timeout=timeout)
            if image_data:
                # Try to determine MIME type from URL or content
                mime_type = "image/png"  # Default
                parsed_url = urlparse(url)
                path = parsed_url.path.lower()
                if path.endswith(".jpg") or path.endswith(".jpeg"):
                    mime_type = "image/jpeg"
                elif path.endswith(".png"):
                    mime_type = "image/png"
                elif path.endswith(".gif"):
                    mime_type = "image/gif"
                elif path.endswith(".webp"):
                    mime_type = "image/webp"
                # Could also check Content-Type header, but for now use file extension
                
                return types.Part(
                    inline_data=types.Blob(mime_type=mime_type, data=image_data)
                )
            else:
                logger.warning(f"Failed to fetch image from URL: {url}")
                return None
    
    # Handle other media types if needed in the future
    return None


async def openai_response_to_llm_response(
    response: Any,
    url_fetch_timeout: float = 30.0,
) -> LlmResponse:
    """Converts OpenAI Responses API response to ADK LlmResponse.
    
    Supports Responses API format (output list).
    Handles multimodal responses including images and other media.
    
    Args:
        response: The OpenAI Responses API response object
        url_fetch_timeout: Timeout in seconds for fetching external image URLs (default: 30.0)
    """
    logger.info("Received response from OpenAI.")
    logger.debug(f"OpenAI response: {response}")

    # Extract content from response
    content_parts = []

    # Parse Responses API format (has 'output' field)
    if hasattr(response, "output"):
        try:
            output_value = response.output
            if output_value:
                # Responses API format
                for output_item in output_value:
                    # Handle text messages
                    if hasattr(output_item, "type") and output_item.type == "message":
                        if hasattr(output_item, "content") and output_item.content:
                            # content is a list of content parts
                            for content_part in output_item.content:
                                if hasattr(content_part, "type"):
                                    if content_part.type == "text" and hasattr(content_part, "text"):
                                        content_parts.append(types.Part(text=content_part.text))
                                    elif content_part.type == "image_url":
                                        # Handle image_url and other media content
                                        media_part = await _convert_media_content_to_part(content_part, timeout=url_fetch_timeout)
                                        if media_part:
                                            content_parts.append(media_part)
                                        else:
                                            logger.debug(
                                                f"Could not convert media content to Part: {getattr(content_part, 'type', 'unknown')}"
                                            )
                                    else:
                                        # Log unknown content part types for debugging
                                        logger.debug(
                                            f"Unknown content part type in response: {getattr(content_part, 'type', 'unknown')}"
                                        )
                    
                    # Handle function tool calls
                    elif hasattr(output_item, "type") and output_item.type == "function":
                        function_args = {}
                        if hasattr(output_item, "arguments") and output_item.arguments:
                            if isinstance(output_item.arguments, str):
                                try:
                                    function_args = json.loads(output_item.arguments)
                                except json.JSONDecodeError:
                                    function_args = {"arguments": output_item.arguments}
                            elif isinstance(output_item.arguments, dict):
                                function_args = output_item.arguments
                        
                        call_id = getattr(output_item, "call_id", None) or getattr(output_item, "id", None)
                        function_name = getattr(output_item, "name", "")
                        
                        content_parts.append(
                            types.Part(
                                function_call=types.FunctionCall(
                                    id=call_id,
                                    name=function_name,
                                    args=function_args,
                                )
                            )
                        )
                    else:
                        # Log unknown output item types for debugging
                        logger.debug(
                            f"Unknown output item type in Responses API: {getattr(output_item, 'type', 'unknown')}"
                        )
        except (AttributeError, TypeError) as e:
            # output exists but is not accessible or is None/empty
            logger.debug(f"Could not parse Responses API output format: {e}")

    # Create content
    content = (
        types.Content(role="model", parts=content_parts) if content_parts else None
    )

    # Extract usage metadata
    usage_metadata = None
    if hasattr(response, "usage") and response.usage:
        # Responses API format
        if hasattr(response.usage, "input_tokens"):
            usage_metadata = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=getattr(response.usage, "input_tokens", 0),
                candidates_token_count=getattr(response.usage, "output_tokens", 0),
                total_token_count=getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0),
            )

    # Extract finish reason
    finish_reason = None
    if hasattr(response, "incomplete_details"):
        # Responses API format - check if response is incomplete
        if response.incomplete_details:
            finish_reason = "MAX_TOKENS"  # Response was truncated
        else:
            finish_reason = "STOP"  # Response completed normally
    elif hasattr(response, "error") and response.error:
        # Responses API format - error occurred
        finish_reason = "FINISH_REASON_UNSPECIFIED"
        # Extract error details if available
        error_code = None
        error_message = None
        if hasattr(response.error, "code"):
            error_code = str(response.error.code)
        if hasattr(response.error, "message"):
            error_message = str(response.error.message)
        # Return error response early if error details are available
        if error_code or error_message:
            return LlmResponse(
                content=None,
                usage_metadata=usage_metadata,
                finish_reason=finish_reason,
                error_code=error_code,
                error_message=error_message,
            )
    else:
        # Default to STOP if no finish reason available
        finish_reason = "STOP"

    # Extract model name if available
    model_name = None
    if hasattr(response, "model"):
        model_name = response.model
        logger.debug(f"Response from model: {model_name}")

    # Extract system fingerprint if available
    system_fingerprint = None
    if hasattr(response, "system_fingerprint"):
        system_fingerprint = response.system_fingerprint
        logger.debug(f"System fingerprint: {system_fingerprint}")

    return LlmResponse(
        content=content, usage_metadata=usage_metadata, finish_reason=finish_reason
    )


class OpenAI(BaseLlm):
    """Integration with OpenAI GPT models using the Responses API.

    This implementation uses OpenAI's Responses API (introduced in 2025), which
    supports file inputs including PDFs and other document formats. The Responses
    API provides a unified interface for assistants functionality.

    Configuration is read from environment variables and the GenerateContentConfig:
    - API keys: OPENAI_API_KEY or AZURE_OPENAI_API_KEY
    - Base URL: OPENAI_BASE_URL (for OpenAI) or AZURE_OPENAI_ENDPOINT (for Azure)
    - Azure API version: AZURE_OPENAI_API_VERSION (defaults to "2024-02-15-preview")
    - max_output_tokens: Set via GenerateContentConfig (Responses API uses max_output_tokens directly),
      temperature: Set via GenerateContentConfig in the request
    - Additional parameters: top_p, frequency_penalty, presence_penalty, seed,
      candidate_count (maps to OpenAI's n), stop_sequences, logprobs can be set via GenerateContentConfig
    - Note: logit_bias, top_logprobs, and user are OpenAI-specific and not supported in Google GenAI
    - Structured outputs: Supported via response_schema or response_mime_type in GenerateContentConfig
    - Tool choice: Configurable via tool_config.function_calling_config.mode in GenerateContentConfig
      (uses FunctionCallingConfigMode: NONE, AUTO, ANY, VALIDATED)

    File Input Support:
    - Documents (PDFs, Word docs, etc.) are uploaded via Files API when use_files_api=True
    - Images use base64 encoding
    - The Responses API supports file inputs directly in message content

    Structured Output Support:
    - JSON mode: Set response_mime_type="application/json" or response_format={"type": "json_object"}
    - Structured outputs with schema: Set response_schema with a JSON schema dict or Schema object
      This will be converted to OpenAI's response_format with type "json_schema"

    Attributes:
      model: The name of the OpenAI model or (for Azure) the deployment name.
      use_files_api: Whether to use OpenAI's Files API for file uploads (default: True).
                     The Responses API supports file inputs, so Files API is enabled by default.
      url_fetch_timeout: Timeout in seconds for fetching external image URLs (default: 30.0).
                         This is used when converting image_url content parts from external URLs.
    """

    model: str = "gpt-4o"
    use_files_api: bool = True
    url_fetch_timeout: float = 30.0

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
                # If AsyncAzureOpenAI is not available, raise an error as the sync client is not compatible.
                raise ImportError(
                    "The installed `openai` version is too old for Azure async support. "
                    "Please upgrade to `openai>=2.7.2`."
                )

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

        # Convert request to Responses API format
        messages = []
        instructions = None

        # Extract system instruction for Responses API (uses 'instructions' parameter)
        if llm_request.config and llm_request.config.system_instruction:
            if isinstance(llm_request.config.system_instruction, str):
                instructions = llm_request.config.system_instruction

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
        # Responses API uses 'input' instead of 'messages'
        # Input is a list of messages for the Responses API
        request_params = {
            "model": request_model,
            "input": messages,  # Responses API accepts messages list in input
            "stream": stream,
        }
        
        # Add instructions if present (Responses API uses 'instructions' for system prompt)
        if instructions:
            request_params["instructions"] = instructions

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

        # Get max_output_tokens and temperature from config
        if llm_request.config:
            if llm_request.config.max_output_tokens:
                # Responses API uses max_output_tokens directly
                request_params["max_output_tokens"] = llm_request.config.max_output_tokens

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
            # Handle response_logprobs: if True, enable logprobs if not already set
            response_logprobs = getattr(llm_request.config, "response_logprobs", None)
            if response_logprobs is True and logprobs is None:
                # If response_logprobs is True but logprobs not set, use default value
                # OpenAI Responses API: logprobs defaults to None (disabled), but if we want logprobs
                # we need to set it to a positive integer (number of top logprobs to return)
                # Default to 5 if response_logprobs is True
                logprobs = 5
                logger.debug("response_logprobs=True but logprobs not set, defaulting to logprobs=5")
            if logprobs is not None:
                request_params["logprobs"] = logprobs

            # top_logprobs is OpenAI-specific and not available in Google GenAI - removed
            
            # user is OpenAI-specific and not available in Google GenAI - removed
            
            # Handle caching_config: Map to OpenAI Responses API cache parameters if available
            caching_config = getattr(llm_request.config, "caching_config", None)
            if caching_config is not None:
                # OpenAI Responses API may support cache parameters
                # Check if caching_config has enable_cache or similar
                enable_cache = getattr(caching_config, "enable_cache", None)
                if enable_cache is not None:
                    # Try to map to OpenAI cache parameter (if it exists)
                    # Note: OpenAI Responses API cache behavior may be automatic
                    # If explicit cache control is needed, it might be via a 'cache' parameter
                    # For now, log that caching_config was provided
                    logger.debug(f"caching_config.enable_cache={enable_cache} - OpenAI Responses API cache behavior is automatic")
                # Check for other cache-related fields
                cache_ttl = getattr(caching_config, "ttl", None)
                if cache_ttl is not None:
                    logger.debug(f"caching_config.ttl={cache_ttl} - OpenAI Responses API does not support explicit TTL")
            
            # Handle grounding_config: Map to web search tool if applicable
            grounding_config = getattr(llm_request.config, "grounding_config", None)
            if grounding_config is not None:
                # Check if grounding_config enables web search
                # Google GenAI grounding_config may have web_search or similar fields
                web_search_enabled = getattr(grounding_config, "web_search", None)
                if web_search_enabled is True:
                    # OpenAI Responses API supports web search via tools
                    # Check if web search tool is already in tools list
                    web_search_tool_exists = False
                    current_tools = request_params.get("tools") or tools
                    if current_tools:
                        for tool in current_tools:
                            if isinstance(tool, dict) and tool.get("type") == "web_search":
                                web_search_tool_exists = True
                                break
                            elif hasattr(tool, "type") and tool.type == "web_search":
                                web_search_tool_exists = True
                                break
                    
                    if not web_search_tool_exists:
                        # Add web search tool to tools list
                        current_tools.append({"type": "web_search"})
                        request_params["tools"] = current_tools
                        # Update tools variable for consistency
                        tools = current_tools
                        logger.debug("grounding_config.web_search=True - added web_search tool to request")
                else:
                    # Other grounding features might not have direct equivalents
                    logger.debug(f"grounding_config provided but web_search not enabled - other grounding features may not be supported")
            
            # Handle safety_settings: OpenAI Responses API may have moderation features
            safety_settings = getattr(llm_request.config, "safety_settings", None)
            if safety_settings is not None:
                # Google GenAI safety_settings is a list of SafetySetting objects
                # OpenAI Responses API may have moderation parameters
                # Check if OpenAI supports explicit moderation settings
                # For now, note that OpenAI has built-in moderation that cannot be disabled
                # but specific safety thresholds might not be configurable
                if isinstance(safety_settings, list) and len(safety_settings) > 0:
                    logger.debug(f"safety_settings provided with {len(safety_settings)} settings - OpenAI Responses API uses built-in moderation (not configurable)")
                else:
                    logger.debug("safety_settings provided - OpenAI Responses API uses built-in moderation (not configurable)")

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
                                    schema_name = response_schema.__name__
                                else:
                                    schema_name = type(response_schema).__name__
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
                                    except (AttributeError, TypeError) as e:
                                        logger.debug(f"Could not inspect MRO for schema: {e}")
                                    
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
                                        except Exception as e:
                                            logger.debug(f"Could not get schema from instantiated model: {e}")
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
            except (AttributeError, TypeError, ValueError) as e:
                # If there's any issue accessing config attributes, log and continue
                logger.warning(f"Error checking structured output config: {e}")

        logger.info(
            "Sending request to OpenAI Responses API, model: %s, stream: %s",
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
        logger.debug(f"OpenAI request params (excluding input): { {k: v for k, v in request_params.items() if k != 'input'} }")

        try:
            if stream:
                # Handle streaming response using Responses API
                stream_response = await client.responses.create(**request_params)

                # Accumulate content and tool calls across events
                accumulated_text = ""
                accumulated_tool_calls: Dict[str, Dict[str, Any]] = {}
                finish_reason = None
                usage_metadata = None
                model_name = None
                response_id = None
                system_fingerprint = None

                async for event in stream_response:
                    # Track model name and response ID from response.created event
                    if hasattr(event, "type"):
                        event_type = getattr(event, "type", None)
                        
                        # Handle response.created event
                        if event_type == "response.created":
                            if hasattr(event, "response") and hasattr(event.response, "model"):
                                model_name = event.response.model
                            if hasattr(event, "response") and hasattr(event.response, "id"):
                                response_id = event.response.id
                        
                        # Handle content part added events (text content)
                        elif event_type == "response.content_part.added":
                            if hasattr(event, "part"):
                                part = event.part
                                if hasattr(part, "type"):
                                    if part.type == "text" and hasattr(part, "text"):
                                        accumulated_text += part.text
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
                        
                        # Handle reasoning text done events
                        elif event_type == "response.reasoning_text.done":
                            if hasattr(event, "text"):
                                accumulated_text += event.text
                                partial_content = types.Content(
                                    role="model",
                                    parts=[types.Part(text=accumulated_text)],
                                )
                                yield LlmResponse(
                                    content=partial_content,
                                    partial=True,
                                    turn_complete=False,
                                )
                        
                        # Handle function tool call events
                        elif event_type == "response.function_call.added":
                            if hasattr(event, "function_call"):
                                func_call = event.function_call
                                call_id = getattr(func_call, "call_id", None) or getattr(func_call, "id", None)
                                if call_id:
                                    accumulated_tool_calls[call_id] = {
                                        "id": call_id,
                                        "type": "function",
                                        "function": {
                                            "name": getattr(func_call, "name", ""),
                                            "arguments": getattr(func_call, "arguments", ""),
                                        },
                                    }
                        
                        # Handle function arguments delta events
                        elif event_type == "response.function_call.arguments_delta":
                            if hasattr(event, "delta") and hasattr(event, "call_id"):
                                call_id = event.call_id
                                if call_id in accumulated_tool_calls:
                                    delta_text = getattr(event.delta, "text", "")
                                    accumulated_tool_calls[call_id]["function"]["arguments"] += delta_text
                        
                        # Handle response done event
                        elif event_type == "response.done":
                            if hasattr(event, "response"):
                                resp = event.response
                                # Extract usage metadata
                                if hasattr(resp, "usage") and resp.usage:
                                    if hasattr(resp.usage, "input_tokens"):
                                        usage_metadata = types.GenerateContentResponseUsageMetadata(
                                            prompt_token_count=getattr(resp.usage, "input_tokens", 0),
                                            candidates_token_count=getattr(resp.usage, "output_tokens", 0),
                                            total_token_count=getattr(resp.usage, "input_tokens", 0) + getattr(resp.usage, "output_tokens", 0),
                                        )
                                
                                # Extract finish reason
                                if hasattr(resp, "incomplete_details") and resp.incomplete_details:
                                    finish_reason = "MAX_TOKENS"
                                elif hasattr(resp, "error") and resp.error:
                                    finish_reason = "FINISH_REASON_UNSPECIFIED"
                                else:
                                    finish_reason = "STOP"

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
                # Handle non-streaming response using Responses API
                response = await client.responses.create(**request_params)
                llm_response = await openai_response_to_llm_response(response, url_fetch_timeout=self.url_fetch_timeout)
                yield llm_response

        except openai.APIError as e:
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

        This is used with the Responses API which supports file inputs.

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
            mode="wb", delete=False, suffix=self._get_file_extension(mime_type)
        ) as temp_file:
            temp_file.write(file_data)
            temp_file_path = temp_file.name

        try:
            # Use the client.files.create API
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
        """Handle file data using Files API (for documents) or base64 encoding (for images)."""
        if part.inline_data:
            # Handle inline data
            mime_type = part.inline_data.mime_type or "application/octet-stream"
            data = part.inline_data.data
            display_name = part.inline_data.display_name

            # For documents (PDFs, Word docs, etc.), use Files API if enabled
            if self.use_files_api and mime_type in [
                "application/pdf",
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ]:
                try:
                    file_id = await self._upload_file_to_openai(
                        data, mime_type, display_name
                    )
                    # Responses API supports file inputs in message content
                    return {
                        "type": "file",
                        "file_id": file_id,
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
                f"OpenAI Responses API does not support file references. "
                f"File '{display_name}' ({file_uri}) converted to text description."
            )

            return {
                "type": "text",
                "text": f"[FILE REFERENCE: {display_name}]\n"
                f"URI: {file_uri}\n"
                f"Type: {mime_type}\n"
                f"Note: OpenAI Responses API does not support file references.",
            }

        return {"type": "text", "text": str(part)}

    def _preprocess_request(self, llm_request: LlmRequest):
        """Preprocesses the request before sending to OpenAI."""
        # Set model if not specified
        if not llm_request.model:
            llm_request.model = self.model
