"""OpenAI integration for GPT models in the ADK framework.

This file uses OpenAI's Responses API which supports file inputs including PDFs
and other document formats. The Responses API provides a unified interface for
assistants functionality.

Key features:
- Uses Responses API (client.responses.create)
- Supports file inputs via Files API for documents (PDFs, Word docs, etc.)
- Detects Azure when `azure_endpoint` (or an Azure-looking `base_url`) is set
- Use `AsyncAzureOpenAI` when targeting Azure, `AsyncOpenAI` otherwise

Content Type Mapping (Google GenAI ↔ OpenAI Responses API):
- Google GenAI → OpenAI:
  - text → input_text (user/system) or output_text (assistant/model)
  - inline_data (images) → input_image
  - file_data → input_file
  
- OpenAI → Google GenAI:
  - input_text/output_text → text (Part)
  - input_image/computer_screenshot → inline_data (Part with Blob)
  - input_file → file_data (Part) or text placeholder if file_id not mappable
  - refusal → text (Part) with [REFUSAL] prefix
  - summary_text → text (Part) with [SUMMARY] prefix

Note: Some OpenAI-specific types (refusal, computer_screenshot, summary_text)
are converted to text with semantic markers, as Google GenAI doesn't have direct
equivalents. File references using OpenAI's file_id may need special handling.

References: OpenAI Responses API, Azure/OpenAI client usage patterns.
"""

from __future__ import annotations

import base64
import copy
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


def _convert_content_type_for_responses_api(
    content: Dict[str, Any], role: Optional[str]
) -> Dict[str, Any]:
    """Converts content type to Responses API format based on role.
    
    Responses API requires specific content types:
    - input_text, input_image, input_file for user/system messages
    - output_text, refusal, summary_text for assistant/model messages
    - computer_screenshot (can be input or output)
    
    Mapping from Google GenAI to OpenAI Responses API:
    - text → input_text (user/system) or output_text (assistant/model)
    - image_url → input_image (user/system) or output_text (assistant/model)
    - file → input_file (user/system)
    
    Important: Responses API expects input_image to have a URL string directly,
    not an object with a 'url' field like Chat Completions API.
    
    Args:
        content: Content dict with 'type' field
        role: The role of the message ('user', 'system', 'model', 'assistant')
    
    Returns:
        Content dict with updated type and structure for Responses API
    """
    if not isinstance(content, dict) or "type" not in content:
        return content
    
    content_type = content["type"]
    is_input = role in ("user", "system")
    
    # Map content types for Responses API
    if content_type == "text":
        content["type"] = "input_text" if is_input else "output_text"
    elif content_type == "image_url":
        # Responses API expects input_image with URL string directly, not object
        content["type"] = "input_image"
        # Convert image_url object to URL string
        if "image_url" in content:
            image_url_obj = content.pop("image_url")
            # Extract URL from image_url object (can be string or object with url field)
            if isinstance(image_url_obj, str):
                url = image_url_obj
            elif isinstance(image_url_obj, dict) and "url" in image_url_obj:
                url = image_url_obj["url"]
            elif hasattr(image_url_obj, "url"):
                url = image_url_obj.url
            else:
                logger.warning(f"Could not extract URL from image_url object: {image_url_obj}")
                url = None
            
            if url:
                # Responses API expects the URL directly in the content dict
                content["image_url"] = url
            else:
                logger.error("Failed to extract image URL for Responses API")
        # If image_url is already a string in the content, keep it
    elif content_type == "file":
        content["type"] = "input_file"  # Files are input
    # Other types like "function" should remain as-is
    
    return content


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
            
            # Validate that we have a tool_call_id - REQUIRED for Responses API
            if not func_response.id:
                raise ValueError(
                    f"Function response missing required 'id' field. "
                    f"Cannot create tool message without tool_call_id. "
                    f"Response content: {response_text[:200]}"
                )
            
            # Ensure tool_call_id is a string (OpenAI expects string IDs)
            try:
                tool_call_id = str(func_response.id)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Function response id cannot be converted to string. "
                    f"ID: {func_response.id}, Type: {type(func_response.id).__name__}, Error: {e}"
                )
            
            if not tool_call_id or not tool_call_id.strip():
                raise ValueError(
                    f"Function response id is empty after conversion to string. "
                    f"Original ID: {func_response.id}"
                )
            
            tool_message = {
                "role": "tool",
                "content": response_text,
                "tool_call_id": tool_call_id,  # Must be a string matching the function_call id
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
                    # Convert content type for Responses API
                    converted_content = _convert_content_type_for_responses_api(
                        openai_content.copy(), content.role
                    )
                    message_content.append(converted_content)
                else:
                    # Convert string content to dict and apply Responses API type
                    text_content = {"type": "text", "text": openai_content}
                    converted_content = _convert_content_type_for_responses_api(
                        text_content, content.role
                    )
                    message_content.append(converted_content)
            
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
                # Convert content type for Responses API
                converted_content = _convert_content_type_for_responses_api(
                    openai_content.copy(), content.role
                )
                message_content.append(converted_content)
            else:
                # Convert string content to dict and apply Responses API type
                text_content = {"type": "text", "text": openai_content}
                converted_content = _convert_content_type_for_responses_api(
                    text_content, content.role
                )
                message_content.append(converted_content)
        
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
    # Validate that function name is present and not empty
    # Try multiple ways to access the name attribute
    function_name = None
    if hasattr(function_declaration, "name"):
        function_name = getattr(function_declaration, "name", None)
    elif hasattr(function_declaration, "__dict__") and "name" in function_declaration.__dict__:
        function_name = function_declaration.__dict__["name"]
    
    # Validate name
    if not function_name:
        raise ValueError(
            f"FunctionDeclaration must have a non-empty 'name' field. "
            f"Got: {function_name!r} (type: {type(function_name).__name__ if function_name else 'None'}), "
            f"FunctionDeclaration attributes: {dir(function_declaration)}"
        )
    
    # Convert to string and strip whitespace
    if isinstance(function_name, str):
        function_name = function_name.strip()
        if not function_name:
            raise ValueError(
                f"FunctionDeclaration 'name' field is empty or whitespace only. "
                f"Got: {repr(function_declaration.name)}"
            )
    else:
        # Convert non-string to string
        function_name = str(function_name).strip()
        if not function_name:
            raise ValueError(
                f"FunctionDeclaration 'name' field cannot be converted to non-empty string. "
                f"Got: {repr(getattr(function_declaration, 'name', None))}"
            )
    
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
        "name": function_name,
        "description": function_declaration.description or "",
        "parameters": {
            "type": "object",
            "properties": properties,
        },
    }

    if required_params:
        function_schema["parameters"]["required"] = required_params

    tool = {"type": "function", "function": function_schema}
    
    # Validate the final tool structure
    if not tool.get("function", {}).get("name"):
        raise ValueError(
            f"Tool conversion failed: 'name' field missing in function schema. "
            f"FunctionDeclaration.name: {function_name!r}, "
            f"Tool structure: {tool}"
        )
    
    return tool


def convert_tools_to_responses_api_format(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert tools from Chat Completions format to Responses API format.
    
    The Responses API expects a FLATTENED format where name, description, and parameters
    are at the top level of the tool object, NOT nested in a "function" object.
    
    Chat Completions format:
        {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    
    Responses API format:
        {"type": "function", "name": "...", "description": "...", "parameters": {...}}
    
    Args:
        tools: List of tools in Chat Completions format
    
    Returns:
        List of tools in Responses API format (flattened structure)
    """
    converted_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            logger.warning(f"Skipping invalid tool (not a dict): {tool}")
            continue
        
        tool_type = tool.get("type")
        function_obj = tool.get("function", {})
        
        if tool_type == "function" and function_obj:
            function_name = function_obj.get("name")
            if not function_name:
                logger.error(f"Skipping tool with missing function.name: {tool}")
                continue
            
            # Responses API expects FLATTENED structure:
            # {type: "function", name: "...", description: "...", parameters: {...}}
            # NOT nested in "function" object!
            converted_tool = {
                "type": "function",
                "name": function_name,
                "description": function_obj.get("description", ""),
                "parameters": function_obj.get("parameters", {"type": "object", "properties": {}})
            }
            converted_tools.append(converted_tool)
            logger.debug(f"Converted tool for Responses API: name={function_name} (flattened structure)")
        else:
            # For non-function tools, pass through as-is but log a warning
            logger.warning(f"Tool with type '{tool_type}' may not be supported in Responses API: {tool}")
            converted_tools.append(tool)
    
    return converted_tools


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


def _ensure_strict_json_schema(
    json_schema: Dict[str, Any],
    path: tuple = (),
) -> Dict[str, Any]:
    """Recursively ensures all object types in a JSON schema have additionalProperties: false.
    
    This is required by OpenAI's Responses API when using strict mode.
    Mutates the schema in place to add additionalProperties: false to all object types.
    
    Args:
        json_schema: The JSON schema dictionary to process
        path: Current path in the schema (for error reporting)
    
    Returns:
        The modified schema dictionary
    """
    if not isinstance(json_schema, dict):
        return json_schema
    
    # Process $defs (JSON Schema draft 2020-12)
    if "$defs" in json_schema and isinstance(json_schema["$defs"], dict):
        for def_name, def_schema in json_schema["$defs"].items():
            if isinstance(def_schema, dict):
                _ensure_strict_json_schema(def_schema, path=(*path, "$defs", def_name))
    
    # Process definitions (JSON Schema draft 4/7)
    if "definitions" in json_schema and isinstance(json_schema["definitions"], dict):
        for definition_name, definition_schema in json_schema["definitions"].items():
            if isinstance(definition_schema, dict):
                _ensure_strict_json_schema(definition_schema, path=(*path, "definitions", definition_name))
    
    # Process properties - recursively ensure nested objects are strict
    if "properties" in json_schema and isinstance(json_schema["properties"], dict):
        for prop_name, prop_schema in json_schema["properties"].items():
            if isinstance(prop_schema, dict):
                _ensure_strict_json_schema(prop_schema, path=(*path, "properties", prop_name))
    
    # Process items in arrays (for array of objects)
    if "items" in json_schema:
        items_schema = json_schema["items"]
        if isinstance(items_schema, dict):
            _ensure_strict_json_schema(items_schema, path=(*path, "items"))
        elif isinstance(items_schema, list):
            # Array of schemas (tuple validation)
            for i, item_schema in enumerate(items_schema):
                if isinstance(item_schema, dict):
                    _ensure_strict_json_schema(item_schema, path=(*path, "items", i))
    
    # Process anyOf, oneOf, allOf - recursively process each schema
    for keyword in ["anyOf", "oneOf", "allOf"]:
        if keyword in json_schema and isinstance(json_schema[keyword], list):
            for i, sub_schema in enumerate(json_schema[keyword]):
                if isinstance(sub_schema, dict):
                    _ensure_strict_json_schema(sub_schema, path=(*path, keyword, i))
    
    # Add additionalProperties: false to all object types
    schema_type = json_schema.get("type")
    if schema_type == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False
        logger.debug(f"Added additionalProperties: false to object at path: {path if path else 'root'}")
    
    return json_schema


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

    # Parse Responses API format (has 'output' field and potentially 'output_text')
    # IMPORTANT: Both output_text and output array may contain the same text content.
    # We should prioritize the output array if present, and only use output_text as fallback
    # to avoid duplicate messages.
    
    # Check if output array exists and has items
    has_output_array = hasattr(response, "output") and response.output and len(response.output) > 0
    
    # Only use output_text if output array is not present or empty
    # This prevents duplicate messages when both output_text and output array contain the same text
    if not has_output_array:
        if hasattr(response, "output_text") and response.output_text:
            logger.debug("Found output_text in Responses API response (no output array present)")
            content_parts.append(types.Part(text=response.output_text))
    elif hasattr(response, "output_text") and response.output_text:
        logger.debug(
            "Both output_text and output array present - using output array to avoid duplicates. "
            f"output_text will be ignored: {response.output_text[:100]}..."
        )
    
    # Parse Responses API format (has 'output' field - list of items)
    if hasattr(response, "output"):
        try:
            output_value = response.output
            if output_value:
                logger.debug(f"Parsing Responses API output with {len(output_value)} item(s)")
                # Responses API format - output is a list of items
                for output_item in output_value:
                    item_type = getattr(output_item, "type", None)
                    logger.debug(f"Processing output item type: {item_type}")
                    
                    # Handle function tool calls (Responses API uses "function_call" not "function")
                    if item_type == "function_call":
                        function_args = {}
                        if hasattr(output_item, "arguments") and output_item.arguments:
                            if isinstance(output_item.arguments, str):
                                try:
                                    function_args = json.loads(output_item.arguments)
                                except json.JSONDecodeError:
                                    logger.warning(f"Could not parse function arguments as JSON: {output_item.arguments}")
                                    function_args = {"arguments": output_item.arguments}
                            elif isinstance(output_item.arguments, dict):
                                function_args = output_item.arguments
                        
                        call_id = getattr(output_item, "call_id", None) or getattr(output_item, "id", None)
                        function_name = getattr(output_item, "name", "")
                        
                        logger.debug(f"Function call: name={function_name}, call_id={call_id}, args={function_args}")
                        
                        content_parts.append(
                            types.Part(
                                function_call=types.FunctionCall(
                                    id=call_id,
                                    name=function_name,
                                    args=function_args,
                                )
                            )
                        )
                    
                    # Handle text messages
                    elif item_type == "message":
                        if hasattr(output_item, "content") and output_item.content:
                            # content is a list of content parts
                            for content_part in output_item.content:
                                if hasattr(content_part, "type"):
                                    content_type = content_part.type
                                    
                                    # Handle text-based content types
                                    if content_type in ("text", "input_text", "output_text", "refusal", "summary_text"):
                                        text_content = getattr(content_part, "text", "")
                                        if text_content:
                                            # For refusal and summary_text, preserve semantic meaning in text
                                            if content_type == "refusal":
                                                # Mark refusal content for potential safety handling
                                                text_content = f"[REFUSAL] {text_content}"
                                            elif content_type == "summary_text":
                                                # Mark summary content
                                                text_content = f"[SUMMARY] {text_content}"
                                            content_parts.append(types.Part(text=text_content))
                                    
                                    # Handle image content types
                                    elif content_type in ("image_url", "input_image", "computer_screenshot"):
                                        # Handle image_url and other media content
                                        media_part = await _convert_media_content_to_part(content_part, timeout=url_fetch_timeout)
                                        if media_part:
                                            content_parts.append(media_part)
                                        else:
                                            logger.debug(
                                                f"Could not convert media content to Part: {getattr(content_part, 'type', 'unknown')}"
                                            )
                                    
                                    # Handle file content
                                    elif content_type == "input_file":
                                        # OpenAI Responses API file content
                                        file_id = getattr(content_part, "file_id", None)
                                        if file_id:
                                            # Convert file reference to Google GenAI format
                                            # Note: Google GenAI uses file_data with file_uri, not file_id
                                            # We'll need to handle this differently - for now, log a warning
                                            logger.warning(
                                                f"OpenAI file_id ({file_id}) cannot be directly mapped to Google GenAI file_data. "
                                                f"File references may need to be handled separately."
                                            )
                                            # Create a text placeholder indicating file reference
                                            content_parts.append(
                                                types.Part(text=f"[FILE_REFERENCE: {file_id}]")
                                            )
                                        else:
                                            logger.debug("input_file content part missing file_id")
                                    
                                    else:
                                        # Log unknown content part types for debugging
                                        logger.debug(
                                            f"Unknown content part type in response: {getattr(content_part, 'type', 'unknown')}"
                                        )
                    
                    # Handle function_call_output (tool execution results)
                    elif item_type == "function_call_output":
                        # This is a tool execution result, not a function call request
                        # In Google ADK, this would be a function_response
                        call_id = getattr(output_item, "call_id", None)
                        output_data = getattr(output_item, "output", "")
                        
                        # Parse output if it's a JSON string
                        if isinstance(output_data, str):
                            try:
                                output_data = json.loads(output_data)
                            except json.JSONDecodeError:
                                pass  # Keep as string if not JSON
                        
                        logger.debug(f"Function call output: call_id={call_id}, output={output_data}")
                        
                        # Convert to function_response format
                        content_parts.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    id=call_id,
                                    response=output_data,
                                )
                            )
                        )
                    
                    else:
                        # Log unknown output item types for debugging
                        logger.debug(
                            f"Unknown output item type in Responses API: {item_type}. "
                            f"Item attributes: {dir(output_item) if hasattr(output_item, '__dict__') else 'N/A'}"
                        )
        except (AttributeError, TypeError) as e:
            # output exists but is not accessible or is None/empty
            logger.warning(f"Could not parse Responses API output format: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    # Create content with all parts (including function calls)
    # Note: GenerateContentResponse.function_calls is a computed property that extracts
    # function calls from candidates[0].content.parts automatically
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

    # Create a GenerateContentResponse object matching Google GenAI's structure
    # IMPORTANT: GenerateContentResponse.function_calls is a computed property that
    # automatically extracts function calls from candidates[0].content.parts
    # We don't need to (and can't) set function_calls directly - it's read-only
    # The function calls are already in content.parts, so the property will extract them
    try:
        # Always create a GenerateContentResponse to match Google GenAI's structure
        # This ensures LlmResponse.create() can properly extract all fields
        genai_response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=content,
                    finish_reason=finish_reason,
                )
            ],
            usage_metadata=usage_metadata,
        )
        
        # Use LlmResponse.create() to properly extract all fields including function_calls
        # This matches how google_llm.py works: LlmResponse.create(response)
        return LlmResponse.create(genai_response)
    except Exception as e:
        logger.warning(
            f"Could not create GenerateContentResponse: {e}. "
            f"Falling back to direct LlmResponse creation. "
            f"Function calls may not be visible in frontend."
        )
        import traceback
        logger.debug(traceback.format_exc())
        # Fallback: create LlmResponse directly (function calls are still in content parts)
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
                        try:
                            # Log the function declaration for debugging
                            logger.debug(
                                f"Converting FunctionDeclaration to OpenAI tool: "
                                f"name={func_decl.name!r}, "
                                f"description={func_decl.description!r}"
                            )
                            
                            openai_tool = function_declaration_to_openai_tool(func_decl)
                            
                            # Validate the tool structure
                            function_obj = openai_tool.get("function", {})
                            tool_name = function_obj.get("name")
                            
                            if not tool_name:
                                logger.error(
                                    f"Tool conversion produced invalid structure: missing 'name' in function. "
                                    f"Tool: {openai_tool}, "
                                    f"FunctionDeclaration.name: {func_decl.name!r}"
                                )
                                continue
                            
                            logger.debug(
                                f"Successfully converted tool: name={tool_name}, "
                                f"type={openai_tool.get('type')}"
                            )
                            tools.append(openai_tool)
                        except ValueError as e:
                            logger.error(
                                f"Failed to convert FunctionDeclaration to OpenAI tool: {e}. "
                                f"FunctionDeclaration: name={func_decl.name!r}, "
                                f"description={func_decl.description!r}"
                            )
                            # Skip this tool but continue with others
                            continue
                        except Exception as e:
                            logger.error(
                                f"Unexpected error converting FunctionDeclaration to OpenAI tool: {e}. "
                                f"FunctionDeclaration: {func_decl}"
                            )
                            continue
        
        # Log final tools structure for debugging
        if tools:
            logger.debug(f"Prepared {len(tools)} tool(s) for OpenAI API")
            for i, tool in enumerate(tools):
                tool_name = tool.get("function", {}).get("name", "UNKNOWN")
                logger.debug(f"Tool {i}: name={tool_name}, structure={tool}")
        elif llm_request.config and llm_request.config.tools:
            logger.warning(
                "No valid tools were prepared from the provided tool configuration. "
                "This may cause API errors if tools are required."
            )

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
        # Input is a list of items (messages, function_call, function_call_output, etc.)
        # IMPORTANT: Responses API doesn't support 'tool_calls' nested in messages
        # Tool calls must be separate items with type="function_call"
        # Tool results must be separate items with type="function_call_output"
        # IMPORTANT: Google GenAI uses 'id' in tool_calls and 'tool_call_id' in tool messages,
        # but OpenAI Responses API uses 'call_id' for both function_call and function_call_output items.
        # We convert Google GenAI's 'id'/'tool_call_id' to OpenAI's 'call_id' during conversion.
        input_items = []
        logger.debug(f"Converting {len(messages)} message(s) to Responses API input format")
        for msg in messages:
            msg_role = msg.get("role")
            
            # Handle tool messages - convert to function_call_output items
            if msg_role == "tool":
                # Extract tool_call_id from Google GenAI format and convert to call_id for OpenAI Responses API
                # Google GenAI uses 'tool_call_id' in tool messages, OpenAI Responses API uses 'call_id'
                # Check multiple possible field names for compatibility
                tool_call_id = msg.get("tool_call_id") or msg.get("call_id") or msg.get("id")
                
                # Validate that tool_call_id exists and is not empty - FAIL if missing
                if not tool_call_id:
                    raise ValueError(
                        f"Tool message missing required 'tool_call_id' field. "
                        f"Cannot create function_call_output item without call_id. "
                        f"Message keys: {list(msg.keys())}, "
                        f"Message: {msg}"
                    )
                
                # Ensure tool_call_id is a string (OpenAI expects string IDs)
                if not isinstance(tool_call_id, str):
                    try:
                        tool_call_id = str(tool_call_id)
                        logger.debug(f"Converted tool_call_id to string: {tool_call_id}")
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"Tool message tool_call_id cannot be converted to string. "
                            f"tool_call_id: {msg.get('tool_call_id')}, Type: {type(msg.get('tool_call_id')).__name__}, Error: {e}"
                        )
                
                if not tool_call_id.strip():
                    raise ValueError(
                        f"Tool message tool_call_id is empty after conversion. "
                        f"Original tool_call_id: {msg.get('tool_call_id')}"
                    )
                
                # Final validation: ensure tool_call_id is still valid after all processing
                if not tool_call_id or not isinstance(tool_call_id, str) or not tool_call_id.strip():
                    raise ValueError(
                        f"Tool message tool_call_id is invalid after processing. "
                        f"tool_call_id={tool_call_id!r}, type={type(tool_call_id).__name__}, "
                        f"original_message={msg}"
                    )
                
                # Convert tool message to function_call_output item
                # Convert Google GenAI format (tool_call_id) to OpenAI Responses API format (call_id)
                # Responses API requires: type="function_call_output", call_id (string), output (string or object)
                function_call_output_item = {
                    "type": "function_call_output",
                    "call_id": tool_call_id.strip(),  # OpenAI Responses API uses 'call_id' (Google GenAI uses 'tool_call_id')
                    "output": msg.get("content", "")
                }
                
                # Double-check the item before appending
                if not function_call_output_item.get("call_id"):
                    raise ValueError(
                        f"function_call_output item missing call_id after creation. "
                        f"Item: {function_call_output_item}, original_message: {msg}"
                    )
                
                input_items.append(function_call_output_item)
                logger.debug(
                    f"Converted tool message to function_call_output: call_id={tool_call_id!r}, "
                    f"output_type={type(msg.get('content', '')).__name__}"
                )
            else:
                # Regular message - remove tool_calls if present and add as message item
                cleaned_msg = msg.copy()
                
                # Extract tool_calls if present and create separate function_call items
                if "tool_calls" in cleaned_msg:
                    tool_calls = cleaned_msg.pop("tool_calls")
                    logger.debug(
                        f"Extracting {len(tool_calls)} tool_call(s) from {msg_role} message "
                        f"to separate function_call items"
                    )
                    
                    # First add the message without tool_calls
                    # Ensure message has type="message" for Responses API
                    message_item = {
                        "type": "message",
                        "role": msg_role,
                        "content": cleaned_msg.get("content", [])
                    }
                    input_items.append(message_item)
                    
                    # Then add each tool_call as a separate function_call item
                    # Track all function_call IDs we create so we can validate tool messages match them
                    function_call_ids = []
                    for tool_call in tool_calls:
                        # Extract 'id' from Google GenAI format and convert to 'call_id' for OpenAI Responses API
                        # Google GenAI uses 'id' in tool_calls, but OpenAI Responses API uses 'call_id'
                        call_id = tool_call.get("id")
                        if not call_id:
                            raise ValueError(
                                f"Tool call missing required 'id' field (Google GenAI format). "
                                f"Cannot create function_call item without id. "
                                f"Tool call: {tool_call}"
                            )
                        
                        # Ensure call_id is a string (OpenAI expects string IDs)
                        if not isinstance(call_id, str):
                            try:
                                call_id = str(call_id)
                                logger.debug(f"Converted function_call id to string: {call_id}")
                            except (TypeError, ValueError) as e:
                                raise ValueError(
                                    f"Tool call id cannot be converted to string. "
                                    f"ID: {tool_call.get('id')}, Type: {type(tool_call.get('id')).__name__}, Error: {e}"
                                )
                        
                        if not call_id.strip():
                            raise ValueError(
                                f"Tool call id is empty after conversion. "
                                f"Original id: {tool_call.get('id')}"
                            )
                        
                        call_id = call_id.strip()
                        function_call_ids.append(call_id)
                        
                        function_name = tool_call.get("function", {}).get("name", "")
                        function_arguments = tool_call.get("function", {}).get("arguments", "{}")
                        
                        # Convert Google GenAI format (id) to OpenAI Responses API format (call_id)
                        function_call_item = {
                            "type": "function_call",
                            "call_id": call_id,  # OpenAI Responses API uses 'call_id' (Google GenAI uses 'id')
                            "name": function_name,
                            "arguments": function_arguments
                        }
                        input_items.append(function_call_item)
                        logger.debug(
                            f"Added function_call item: call_id={call_id!r}, name={function_name}"
                        )
                else:
                    # Regular message without tool_calls
                    # Ensure message has type="message" for Responses API
                    message_item = {
                        "type": "message",
                        "role": msg_role,
                        "content": cleaned_msg.get("content", [])
                    }
                    input_items.append(message_item)
        
        # Log final input items structure for debugging and validate
        logger.debug(f"Converted to {len(input_items)} input item(s) for Responses API")
        for i, item in enumerate(input_items):
            item_type = item.get("type", "unknown")
            if item_type == "function_call_output":
                call_id = item.get("call_id")
                if not call_id:
                    raise ValueError(
                        f"Input item {i} is a function_call_output but missing required 'call_id' field. "
                        f"Item: {item}"
                    )
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(
                        f"Input item {i} has invalid call_id: {call_id!r} (type: {type(call_id).__name__}). "
                        f"call_id must be a non-empty string. Item: {item}"
                    )
                logger.debug(
                    f"Input item {i}: type={item_type}, call_id={call_id!r} "
                    f"(required for function_call_output)"
                )
            elif item_type == "function_call":
                call_id = item.get("call_id")
                if not call_id:
                    raise ValueError(
                        f"Input item {i} is a function_call but missing required 'call_id' field. "
                        f"Item: {item}"
                    )
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(
                        f"Input item {i} has invalid call_id: {call_id!r} (type: {type(call_id).__name__}). "
                        f"call_id must be a non-empty string. Item: {item}"
                    )
                logger.debug(
                    f"Input item {i}: type={item_type}, call_id={call_id!r} "
                    f"(must match call_id in corresponding function_call_output)"
                )
            else:
                logger.debug(f"Input item {i}: type={item_type}")
        
        # Create a clean copy of input_items to ensure no mutations affect the request
        # Also do a final validation pass on the copy
        clean_input_items = []
        for i, item in enumerate(input_items):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Input item {i} is not a dict: {type(item).__name__}, value: {item}"
                )
            
            # Make a copy to avoid mutations
            item_copy = dict(item)
            
            # Validate function_call_output items
            if item_copy.get("type") == "function_call_output":
                if "call_id" not in item_copy:
                    raise ValueError(
                        f"Input item {i} (function_call_output) missing 'call_id' key. "
                        f"Item keys: {list(item_copy.keys())}, Original item: {item}"
                    )
                call_id = item_copy.get("call_id")
                if call_id is None:
                    raise ValueError(
                        f"Input item {i} (function_call_output) has 'call_id' key but value is None. "
                        f"Original item: {item}"
                    )
                if not isinstance(call_id, str):
                    raise ValueError(
                        f"Input item {i} (function_call_output) call_id is not a string: "
                        f"{type(call_id).__name__}, value: {call_id!r}, Original item: {item}"
                    )
                if not call_id.strip():
                    raise ValueError(
                        f"Input item {i} (function_call_output) call_id is empty/whitespace: "
                        f"{call_id!r}, Original item: {item}"
                    )
                # Ensure call_id is clean (no whitespace)
                item_copy["call_id"] = call_id.strip()
            
            # Validate function_call items
            elif item_copy.get("type") == "function_call":
                if "call_id" not in item_copy:
                    raise ValueError(
                        f"Input item {i} (function_call) missing 'call_id' key. "
                        f"Item keys: {list(item_copy.keys())}, Original item: {item}"
                    )
                call_id = item_copy.get("call_id")
                if call_id is None:
                    raise ValueError(
                        f"Input item {i} (function_call) has 'call_id' key but value is None. "
                        f"Original item: {item}"
                    )
                if not isinstance(call_id, str):
                    raise ValueError(
                        f"Input item {i} (function_call) call_id is not a string: "
                        f"{type(call_id).__name__}, value: {call_id!r}, Original item: {item}"
                    )
                if not call_id.strip():
                    raise ValueError(
                        f"Input item {i} (function_call) call_id is empty/whitespace: "
                        f"{call_id!r}, Original item: {item}"
                    )
                # Ensure call_id is clean (no whitespace)
                item_copy["call_id"] = call_id.strip()
            
            clean_input_items.append(item_copy)
        
        request_params = {
            "model": request_model,
            "input": clean_input_items,  # Use cleaned and validated input items
            "stream": stream,
        }
        
        # Add instructions if present (Responses API uses 'instructions' for system prompt)
        if instructions:
            request_params["instructions"] = instructions

        if tools:
            # Validate tools structure before adding to request
            validated_tools = []
            for i, tool in enumerate(tools):
                # Check if tool has the required structure
                if not isinstance(tool, dict):
                    logger.error(f"Tool {i} is not a dict: {tool}")
                    continue
                
                tool_type = tool.get("type")
                function_obj = tool.get("function", {})
                function_name = function_obj.get("name")
                
                if not tool_type:
                    logger.error(f"Tool {i} missing 'type' field: {tool}")
                    continue
                
                if not function_obj:
                    logger.error(f"Tool {i} missing 'function' field: {tool}")
                    continue
                
                if not function_name:
                    logger.error(
                        f"Tool {i} missing 'name' in function object. "
                        f"Tool structure: {tool}, "
                        f"Function object: {function_obj}"
                    )
                    continue
                
                # Ensure name is a non-empty string (not None, not empty)
                if function_name is None:
                    logger.error(
                        f"Tool {i} has None as function.name. "
                        f"Tool structure: {tool}"
                    )
                    continue
                
                if not isinstance(function_name, str) or not function_name.strip():
                    logger.error(
                        f"Tool {i} has invalid function.name: {function_name!r} (type: {type(function_name).__name__}). "
                        f"Tool structure: {tool}"
                    )
                    continue
                
                # Double-check the tool structure is correct before adding
                if "function" not in tool or "name" not in tool["function"]:
                    logger.error(
                        f"Tool {i} structure validation failed after checks. "
                        f"Tool: {tool}"
                    )
                    continue
                
                # Log the tool structure for debugging
                logger.debug(
                    f"Validated tool {i}: type={tool_type}, "
                    f"name={function_name}, "
                    f"name_type={type(function_name).__name__}, "
                    f"full_structure={tool}"
                )
                validated_tools.append(tool)
            
            if not validated_tools:
                logger.error(
                    "No valid tools after validation. Original tools count: "
                    f"{len(tools)}. This will cause API errors."
                )
            elif len(validated_tools) < len(tools):
                logger.warning(
                    f"Only {len(validated_tools)} out of {len(tools)} tools passed validation. "
                    "Some tools may have been skipped."
                )
            
            request_params["tools"] = validated_tools
            logger.debug(f"Added {len(validated_tools)} validated tool(s) to request params")
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
            # Responses API uses tool_choice (same as Chat Completions)
            request_params["tool_choice"] = tool_choice
            # Responses API supports parallel_tool_calls
            # Default to True for better performance (all modern models support it)
            request_params["parallel_tool_calls"] = True

        # Map Google GenAI GenerateContentConfig parameters to Responses API
        # Currently mapped parameters:
        # - max_output_tokens → max_output_tokens (direct)
        # - temperature → temperature (direct)
        # - top_p → top_p (direct)
        # - frequency_penalty → frequency_penalty (direct)
        # - presence_penalty → presence_penalty (direct)
        # - seed → seed (direct)
        # - candidate_count → n (direct)
        # - stop_sequences → stop (direct)
        # - logprobs → logprobs (direct)
        # - top_logprobs → top_logprobs (direct, if available in config)
        # - user → user (direct, if available in config)
        # - max_tool_calls → max_tool_calls (direct, if available in config)
        # - metadata → metadata (direct, if available in config)
        # - prompt_cache_key → prompt_cache_key (direct, if available in config)
        # - safety_identifier → safety_identifier (direct, if available in config)
        # - truncation → truncation (direct, if available in config)
        # - store → store (direct, if available in config)
        # - background → background (direct, if available in config)
        # - service_tier → service_tier (direct, if available in config)
        # - system_instruction → instructions (direct)
        # - response_schema/response_mime_type → text (converted format)
        # - tool_config.function_calling_config.mode → tool_choice (mapped)
        # - tools → tools (converted to Responses API format)
        # - parallel_tool_calls → parallel_tool_calls (default: True)
        # - reasoning_effort → reasoning (mapped: reasoning_effort → reasoning={"effort": ...})
        #
        # Responses API features not yet mapped (no GenAI equivalent):
        # - conversation: Built-in conversation state management
        # - previous_response_id: Stateless multi-turn conversations
        # - include: Response inclusion options
        # - stream_options: Enhanced streaming options
        # - prompt: Alternative prompt format
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

            # top_logprobs - Responses API supports this
            top_logprobs = getattr(llm_request.config, "top_logprobs", None)
            if top_logprobs is not None:
                request_params["top_logprobs"] = top_logprobs
            
            # user - Responses API supports this (for tracking/abuse prevention)
            user = getattr(llm_request.config, "user", None)
            if user is not None:
                request_params["user"] = user
            
            # max_tool_calls - Responses API supports limiting tool calls
            max_tool_calls = getattr(llm_request.config, "max_tool_calls", None)
            if max_tool_calls is not None:
                request_params["max_tool_calls"] = max_tool_calls
            
            # metadata - Responses API supports metadata for tracking
            metadata = getattr(llm_request.config, "metadata", None)
            if metadata is not None:
                request_params["metadata"] = metadata
            
            # prompt_cache_key - Responses API supports prompt caching
            prompt_cache_key = getattr(llm_request.config, "prompt_cache_key", None)
            if prompt_cache_key is not None:
                request_params["prompt_cache_key"] = prompt_cache_key
            
            # safety_identifier - Responses API supports safety tracking
            safety_identifier = getattr(llm_request.config, "safety_identifier", None)
            if safety_identifier is not None:
                request_params["safety_identifier"] = safety_identifier
            
            # truncation - Responses API supports truncation control
            truncation = getattr(llm_request.config, "truncation", None)
            if truncation is not None:
                request_params["truncation"] = truncation
            
            # store - Responses API supports storing responses
            store = getattr(llm_request.config, "store", None)
            if store is not None:
                request_params["store"] = store
            
            # background - Responses API supports async/background processing
            background = getattr(llm_request.config, "background", None)
            if background is not None:
                request_params["background"] = background
            
            # service_tier - Responses API supports different service tiers
            service_tier = getattr(llm_request.config, "service_tier", None)
            if service_tier is not None:
                request_params["service_tier"] = service_tier
            
            # reasoning_effort - Map Google GenAI reasoning_effort to OpenAI Responses API reasoning parameter
            # OpenAI Responses API uses: reasoning={"effort": "low"|"medium"|"high"}
            # Google GenAI uses: reasoning_effort="low"|"medium"|"high"
            reasoning_effort = getattr(llm_request.config, "reasoning_effort", None)
            if reasoning_effort is not None:
                # Validate and normalize the effort value
                if isinstance(reasoning_effort, str):
                    effort_lower = reasoning_effort.lower()
                    # OpenAI Responses API accepts: "low", "medium", "high"
                    if effort_lower in ("low", "medium", "high"):
                        request_params["reasoning"] = {"effort": effort_lower}
                        logger.debug(f"Mapped reasoning_effort={reasoning_effort} to reasoning={{'effort': '{effort_lower}'}}")
                    else:
                        logger.warning(
                            f"reasoning_effort value '{reasoning_effort}' is not a valid Responses API effort value. "
                            f"Expected: 'low', 'medium', or 'high'. Skipping reasoning parameter."
                        )
                else:
                    logger.warning(
                        f"reasoning_effort must be a string, got {type(reasoning_effort).__name__}. "
                        f"Skipping reasoning parameter."
                    )
            
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
                        # Note: OpenAI Responses API may require web_search as a function tool with a name
                        # For now, we'll skip adding it as a tool since it may not be supported in this format
                        # or it needs to be handled differently
                        logger.warning(
                            "web_search tool requested but may not be supported in current OpenAI API format. "
                            "Skipping web_search tool addition to avoid API errors. "
                            "If web_search is needed, it should be added as a proper function tool with a name."
                        )
                        # If web_search needs to be added, it should have the structure:
                        # {"type": "function", "function": {"name": "web_search", "description": "..."}}
                        # For now, we skip it to prevent API errors
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
                                # Already in OpenAI Chat Completions format (nested json_schema)
                                # This will be converted to Responses API format (flattened) later
                                request_params["response_format"] = {
                                    "type": "json_schema",
                                    "json_schema": response_schema,  # Chat Completions format: nested
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
        
        # Final validation: check all function_call_output items before API call
        if "input" in request_params:
            input_array = request_params["input"]
            for i, item in enumerate(input_array):
                if item.get("type") == "function_call_output":
                    if "call_id" not in item:
                        raise ValueError(
                            f"Input item {i} is function_call_output but 'call_id' key is missing! "
                            f"Item keys: {list(item.keys())}, Item: {item}"
                        )
                    call_id = item.get("call_id")
                    if call_id is None:
                        raise ValueError(
                            f"Input item {i} has 'call_id' key but value is None! "
                            f"Item: {item}"
                        )
                    if not isinstance(call_id, str):
                        raise ValueError(
                            f"Input item {i} has call_id but it's not a string! "
                            f"call_id={call_id!r}, type={type(call_id).__name__}, Item: {item}"
                        )
                    if not call_id.strip():
                        raise ValueError(
                            f"Input item {i} has call_id but it's empty/whitespace! "
                            f"call_id={call_id!r}, Item: {item}"
                        )
        
        # Final validation of tools before API call (in case tools were added after initial validation)
        if "tools" in request_params:
            import json
            # Ensure tools is a list
            if not isinstance(request_params["tools"], list):
                logger.error(
                    f"CRITICAL: 'tools' parameter is not a list! Type: {type(request_params['tools'])}, "
                    f"Value: {request_params['tools']}. Removing tools to avoid API errors."
                )
                request_params.pop("tools", None)
            elif len(request_params["tools"]) == 0:
                logger.warning("Tools list is empty. Removing 'tools' from request.")
                request_params.pop("tools", None)
            else:
                # Final validation: filter out any invalid tools
                validated_final_tools = []
                for i, tool in enumerate(request_params["tools"]):
                    
                    # Validate tool structure
                    if not isinstance(tool, dict):
                        logger.error(f"Tool {i} is not a dict, skipping: {tool}")
                        continue
                    
                    tool_type = tool.get("type")
                    function_obj = tool.get("function", {})
                    function_name = function_obj.get("name") if function_obj else None
                    
                    # Check for required fields
                    if not tool_type:
                        logger.error(f"Tool {i} missing 'type' field, skipping: {tool}")
                        continue
                    
                    if tool_type == "function":
                        # Function tools must have a function object with a name
                        if not function_obj:
                            logger.error(f"Tool {i} (type=function) missing 'function' field, skipping: {tool}")
                            continue
                        
                        if not function_name or not isinstance(function_name, str) or not function_name.strip():
                            logger.error(
                                f"Tool {i} (type=function) missing or invalid 'name' in function object, skipping. "
                                f"Tool: {tool}, function_name: {function_name!r}"
                            )
                            continue
                    elif tool_type == "web_search":
                        # web_search might be a special tool type, but OpenAI API may still require a name
                        # Skip it for now to avoid API errors
                        logger.error(
                            f"Tool {i} has type='web_search' which is not supported in this format. Skipping to avoid API errors. "
                            f"Tool structure: {tool}"
                        )
                        continue
                    else:
                        # Unknown tool type - check if it has required structure
                        # OpenAI API requires tools[].name, so even unknown types need proper structure
                        if not function_obj or not function_name:
                            logger.error(
                                f"Tool {i} has unknown type '{tool_type}' and missing required 'function.name' field. "
                                f"Skipping to avoid API errors. Tool: {tool}"
                            )
                            continue
                        logger.warning(f"Tool {i} has unknown type '{tool_type}'. Proceeding but may cause API errors.")
                    
                    validated_final_tools.append(tool)
                
                # Update request_params with validated tools
                if len(validated_final_tools) != len(request_params["tools"]):
                    logger.error(
                        f"FILTERED OUT {len(request_params['tools']) - len(validated_final_tools)} INVALID TOOL(S)! "
                        f"Original count: {len(request_params['tools'])}, Valid count: {len(validated_final_tools)}"
                    )
                    
                    if not validated_final_tools:
                        # No valid tools left - remove tools from request to avoid API errors
                        logger.error("No valid tools remaining after final validation. Removing 'tools' from request.")
                        request_params.pop("tools", None)
                    else:
                        request_params["tools"] = validated_final_tools

        # Convert tools to Responses API format before sending
        if "tools" in request_params and request_params["tools"]:
            request_params["tools"] = convert_tools_to_responses_api_format(request_params["tools"])
        
        # Convert response_format to Responses API 'text' parameter
        # Responses API uses 'text' instead of 'response_format'
        # The 'text' parameter structure:
        # - For json_object: text={"format": {"type": "json_object"}}
        # - For json_schema: text={"format": {"type": "json_schema", "name": "...", "schema": {...}, "strict": true}}
        # IMPORTANT: Responses API uses FLATTENED structure - name, schema, strict are at same level as type
        # NOT nested under json_schema like Chat Completions API
        if "response_format" in request_params:
            response_format = request_params.pop("response_format")
            logger.debug(f"Converting response_format to Responses API 'text' parameter: {response_format}")
            
            try:
                if isinstance(response_format, dict):
                    format_type = response_format.get("type")
                    
                    if format_type == "json_object":
                        # Responses API format: text={"format": {"type": "json_object"}}
                        request_params["text"] = {
                            "format": {"type": "json_object"}
                        }
                        logger.debug("Converted response_format to text: json_object format")
                    
                    elif format_type == "json_schema":
                        # Responses API format: FLATTENED structure
                        # text={"format": {"type": "json_schema", "name": "...", "schema": {...}, "strict": true}}
                        # NOT: text={"format": {"type": "json_schema", "json_schema": {...}}}
                        json_schema_obj = response_format.get("json_schema", {})
                        if json_schema_obj:
                            # Extract name and schema from the nested json_schema object
                            # Chat Completions format: {"name": "...", "schema": {...}}
                            # Responses API format (flattened): {"type": "json_schema", "name": "...", "schema": {...}}
                            if isinstance(json_schema_obj, dict):
                                # Extract name (required)
                                schema_name = json_schema_obj.get("name", "ResponseSchema")
                                
                                # Extract schema (required)
                                if "schema" in json_schema_obj:
                                    schema_dict = json_schema_obj["schema"]
                                else:
                                    # If schema is missing, this is an error
                                    raise ValueError(
                                        f"json_schema object missing required 'schema' field. "
                                        f"Expected structure: {{'name': '...', 'schema': {{...}}}}, "
                                        f"got: {json_schema_obj}"
                                    )
                                
                                # Check if strict mode is enabled
                                strict = json_schema_obj.get("strict", False)
                                if strict:
                                    # When strict mode is enabled, ensure all object types have additionalProperties: false
                                    # This is required by OpenAI's Responses API for strict mode
                                    if isinstance(schema_dict, dict):
                                        # Make a deep copy to avoid mutating the original
                                        schema_dict = copy.deepcopy(schema_dict)
                                        _ensure_strict_json_schema(schema_dict)
                                        logger.debug(
                                            f"Ensured strict JSON schema compliance: added additionalProperties: false "
                                            f"to all object types in schema"
                                        )
                                    else:
                                        logger.warning(
                                            f"Schema is not a dict, cannot ensure strict compliance. "
                                            f"Type: {type(schema_dict).__name__}"
                                        )
                                
                                # Validate schema_name (must be string, max 64 chars, only a-z, A-Z, 0-9, _, -)
                                if not isinstance(schema_name, str):
                                    schema_name = str(schema_name)
                                # Truncate if too long
                                if len(schema_name) > 64:
                                    schema_name = schema_name[:64]
                                    logger.warning(f"Schema name truncated to 64 characters: {schema_name}")
                                # Validate characters (only a-z, A-Z, 0-9, _, -)
                                # Note: re is imported at module level
                                if not re.match(r'^[a-zA-Z0-9_-]+$', schema_name):
                                    # Replace invalid characters
                                    schema_name = re.sub(r'[^a-zA-Z0-9_-]', '_', schema_name)
                                    logger.warning(f"Schema name contains invalid characters, replaced with underscores: {schema_name}")
                                
                                # Build the FLATTENED format structure for Responses API
                                format_dict = {
                                    "type": "json_schema",
                                    "name": schema_name,
                                    "schema": schema_dict
                                }
                                
                                # Add strict if present (optional, defaults to False in Responses API)
                                if strict is not None:
                                    format_dict["strict"] = bool(strict)
                                
                                request_params["text"] = {
                                    "format": format_dict
                                }
                                logger.debug(
                                    f"Converted response_format to text: json_schema format with "
                                    f"name={schema_name}, strict={format_dict.get('strict', False)}"
                                )
                            else:
                                raise ValueError(
                                    f"json_schema must be a dict, got {type(json_schema_obj).__name__}. "
                                    f"Expected structure: {{'name': '...', 'schema': {{...}}}}"
                                )
                        else:
                            raise ValueError(
                                "response_format has json_schema type but missing json_schema field. "
                                "Expected structure: {'type': 'json_schema', 'json_schema': {'name': '...', 'schema': {...}}}"
                            )
                    
                    else:
                        # Unknown format type, log warning and don't convert
                        logger.warning(
                            f"Unknown response_format type '{format_type}' - not converting to Responses API 'text' parameter. "
                            f"Supported types: 'json_object', 'json_schema'."
                        )
                else:
                    logger.warning(
                        f"response_format is not a dict, cannot convert to Responses API 'text' parameter. "
                        f"Type: {type(response_format)}, Value: {response_format}"
                    )
            except Exception as e:
                # If conversion fails, log and skip
                logger.warning(
                    f"Failed to convert response_format to Responses API 'text' parameter: {e}. "
                    "Skipping conversion to avoid API errors."
                )
                import traceback
                logger.debug(traceback.format_exc())
        
        # Log and validate 'text' parameter if present
        if "text" in request_params:
            import json
            text_param = request_params["text"]
            logger.info(f"TEXT PARAMETER BEING SENT: {json.dumps(text_param, indent=2, default=str)}")
            # If text is a dict with 'type', it might be causing the 'text.type' error
            # The Responses API might expect a different structure
            if isinstance(text_param, dict) and "type" in text_param:
                logger.warning(
                    "text parameter is a dict with 'type' field. "
                    "This might cause 'text.type' parameter error. "
                    "Consider using SDK classes or different structure."
                )
        
        # Validate tools format before API call
        if "tools" in request_params:
            for i, tool in enumerate(request_params["tools"]):
                # Validate Responses API format - ensure name is at top level
                if not tool.get("name"):
                    logger.error(
                        f"Tool {i} missing top-level 'name' field (Responses API format)! "
                        f"Tool: {tool}"
                    )

        try:
            if stream:
                # Handle streaming response using Responses API
                logger.info(f"Calling OpenAI Responses API with stream=True, tools count: {len(request_params.get('tools', []))}")
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
                function_calls_list = []
                
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

                        function_call = types.FunctionCall(
                            id=tool_call_data["id"],
                            name=tool_call_data["function"]["name"],
                            args=function_args,
                        )
                        final_parts.append(types.Part(function_call=function_call))
                        function_calls_list.append(function_call)

                final_content = (
                    types.Content(role="model", parts=final_parts)
                    if final_parts
                    else None
                )

                # Create GenerateContentResponse to match Google GenAI structure
                # IMPORTANT: function_calls is a computed property that extracts from content.parts
                # We don't need to set it directly - function calls are already in final_parts
                try:
                    genai_response = types.GenerateContentResponse(
                        candidates=[
                            types.Candidate(
                                content=final_content,
                                finish_reason=finish_reason,
                            )
                        ],
                        usage_metadata=usage_metadata,
                    )
                    
                    # Use LlmResponse.create() to match Google GenAI behavior
                    final_response = LlmResponse.create(genai_response)
                    # Ensure turn_complete is set for streaming
                    final_response.turn_complete = True
                except Exception as e:
                    logger.warning(
                        f"Could not create GenerateContentResponse for streaming: {e}. "
                        f"Falling back to direct LlmResponse creation."
                    )
                    # Fallback: create LlmResponse directly
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
                logger.info(f"Calling OpenAI Responses API with stream=False, tools count: {len(request_params.get('tools', []))}")
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
