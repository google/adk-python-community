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
from openai.types import ReasoningEffort
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
                # For data URIs (base64), keep the full data URI string
                content["image_url"] = url
                logger.debug(f"Converted image_url to Responses API format: {url[:100]}...")
            else:
                # Failed to extract URL - revert type change and add error marker
                logger.error("Failed to extract image URL for Responses API")
                content["type"] = "input_text"  # Fall back to text type
                content["text"] = "[IMAGE ERROR: Could not extract image URL]"
        # If image_url is already a string in the content, keep it
    elif content_type == "file":
        content["type"] = "input_file"  # Files are input
        # Preserve file_id for Responses API
        if "file_id" in content:
            # Responses API expects file_id in the content
            # Keep file_id as-is
            pass
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
    """Recursively normalise JSON Schema for OpenAI Responses strict mode.

    OpenAI's Responses API applies additional constraints on JSON Schema when
    using strict structured outputs:
    - All object types MUST have ``additionalProperties: false``
    - All object types MUST have a ``required`` array that includes *every*
      key in ``properties`` (even if fields are optional in the original schema)

    This helper mutates the schema in place to satisfy those constraints.

    Args:
        json_schema: The JSON schema dictionary to process
        path: Current path in the schema (for debugging)

    Returns:
        The modified schema dictionary.
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
    
    # Normalise object schemas for Responses strict mode
    schema_type = json_schema.get("type")
    if schema_type == "object":
        # 1) additionalProperties: false (if not explicitly set)
        if "additionalProperties" not in json_schema:
            json_schema["additionalProperties"] = False
            logger.debug(
                f"Added additionalProperties: false to object at path: "
                f"{path if path else 'root'}"
            )

        # 2) required: must include *all* properties keys (OpenAI requirement)
        props = json_schema.get("properties")
        if isinstance(props, dict) and props:
            prop_keys = list(props.keys())
            # Always overwrite with the full set of property keys to satisfy
            # "required must include every key in properties"
            json_schema["required"] = prop_keys
            logger.debug(
                f"Set required={prop_keys} for object at path: "
                f"{path if path else 'root'}"
            )

    return json_schema


async def openai_response_to_llm_response(
    response: Any,
    url_fetch_timeout: float = 30.0,
    pydantic_model: Optional[Any] = None,
) -> LlmResponse:
    """Converts OpenAI Responses API response to ADK LlmResponse.
    
    Supports Responses API format (output list).
    Handles multimodal responses including images and other media.
    Also handles parse() method responses which return parsed Pydantic models.
    
    Args:
        response: The OpenAI Responses API response object or parsed Pydantic model
        url_fetch_timeout: Timeout in seconds for fetching external image URLs (default: 30.0)
        pydantic_model: Optional Pydantic model class if this is a parse() response
    """
    logger.info("Received response from OpenAI.")
    logger.debug(f"OpenAI response: {response}")

    # Extract content from response
    content_parts = []

    def _serialize_parsed_content(parsed_obj: Any) -> Optional[str]:
        """Convert parsed content (Pydantic model, dict, etc.) into a JSON/text string."""
        if parsed_obj is None:
            return None
        try:
            if hasattr(parsed_obj, "model_dump_json"):
                # Pydantic v2 helper
                return parsed_obj.model_dump_json()
            if hasattr(parsed_obj, "model_dump"):
                return json.dumps(parsed_obj.model_dump())
            if hasattr(parsed_obj, "dict"):
                return json.dumps(parsed_obj.dict())
            if isinstance(parsed_obj, (dict, list)):
                return json.dumps(parsed_obj)
            if isinstance(parsed_obj, bytes):
                return parsed_obj.decode("utf-8", errors="replace")
            return str(parsed_obj)
        except Exception as exc:
            logger.warning(f"Failed to serialize parsed content: {exc}")
            return str(parsed_obj)

    # Check if this is a parse() response (returns parsed Pydantic model directly)
    # Parse() responses might be:
    # 1. A Pydantic model instance directly
    # 2. A response object with a 'parsed' field containing the model
    # 3. A response object with the parsed data in output_text or output array
    
    # Check if response is a Pydantic model instance (has model_dump or dict method)
    is_pydantic_instance = False
    parsed_data = None
    is_parse_response = False
    pydantic_instance_obj: Optional[Any] = None
    
    # First check if it's a Pydantic model instance directly
    if hasattr(response, "model_dump") or hasattr(response, "dict"):
        # Check if it looks like a Pydantic model (has __class__ and model fields)
        # and doesn't have response-like attributes
        if not (hasattr(response, "output") or hasattr(response, "output_text") or hasattr(response, "usage")):
            is_pydantic_instance = True
            pydantic_instance_obj = response
            if hasattr(response, "model_dump"):
                logger.debug(f"Detected Pydantic v2 model instance: {type(response).__name__}")
            else:
                logger.debug(f"Detected Pydantic v1 model instance: {type(response).__name__}")
            is_parse_response = True
    elif hasattr(response, "parsed"):
        # Response object with parsed field
        parsed_data = response.parsed
        if hasattr(parsed_data, "model_dump"):
            parsed_data = parsed_data.model_dump()
        elif hasattr(parsed_data, "dict"):
            parsed_data = parsed_data.dict()
        logger.debug(f"Found parsed field in response: {type(parsed_data)}")
        is_parse_response = True
    
    # Also check if response has output_text but no output array (parse() might return it this way)
    # Only if we have a pydantic_model hint
    if pydantic_model is not None and not is_parse_response:
        if hasattr(response, "output_text") and response.output_text:
            # Check if output_text contains JSON that matches the Pydantic model
            try:
                parsed_json = json.loads(response.output_text)
                # If it's a dict and we have a pydantic model, treat it as parsed data
                if isinstance(parsed_json, dict):
                    parsed_data = parsed_json
                    is_parse_response = True
                    logger.debug(f"Found parse() response data in output_text for model {pydantic_model.__name__}")
            except (json.JSONDecodeError, AttributeError) as e:
                logger.debug(f"Could not parse output_text as JSON for pydantic model: {e}")
    
    if is_parse_response:
        # This is a parse() response - convert parsed model to JSON/text
        json_text: Optional[str] = None

        if parsed_data is not None:
            json_text = _serialize_parsed_content(parsed_data)
        elif is_pydantic_instance:
            json_text = _serialize_parsed_content(pydantic_instance_obj)
        elif hasattr(response, "output_text") and response.output_text:
            json_text = response.output_text
        else:
            json_text = _serialize_parsed_content(response)

        if json_text:
            logger.debug(f"Converted parse() response to JSON text (length: {len(json_text)})")
            content_parts.append(types.Part(text=json_text))
        else:
            logger.warning("Parse() response serialization produced empty text.")
    else:
        # Regular Responses API format (has 'output' field and potentially 'output_text')
        # IMPORTANT: Both output_text and output array may contain the same text content.
        # We should prioritize the output array if present, and only use output_text as fallback
        # to avoid duplicate messages.
        
        # Check if output array exists and has items
        has_output_array = hasattr(response, "output") and response.output and len(response.output) > 0
        
        # Only use output_text if output array is not present or empty
        # This prevents duplicate messages when both output_text and output array contain the same text
        if not has_output_array:
            if hasattr(response, "output_text") and response.output_text:
                # Ensure output_text is a string (not a MagicMock or other type)
                output_text_value = response.output_text
                if isinstance(output_text_value, str) and output_text_value:
                    logger.debug("Found output_text in Responses API response (no output array present)")
                    content_parts.append(types.Part(text=output_text_value))
        elif hasattr(response, "output_text") and response.output_text:
            # Ensure output_text is a string before slicing
            output_text_value = response.output_text
            if isinstance(output_text_value, str):
                logger.debug(
                    "Both output_text and output array present - using output array to avoid duplicates. "
                    f"output_text will be ignored: {output_text_value[:100]}..."
                )
    
    # Parse Responses API format (has 'output' field - list of items)
    # Skip this if we already handled a parse() response
    if not is_parse_response and hasattr(response, "output"):
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
                                    elif content_type == "parsed":
                                        parsed_content = getattr(content_part, "parsed", None)
                                        if parsed_content is None and hasattr(content_part, "content"):
                                            parsed_content = content_part.content
                                        serialized = _serialize_parsed_content(parsed_content)
                                        if serialized:
                                            content_parts.append(types.Part(text=serialized))
                                        else:
                                            logger.debug("Parsed content part missing data after serialization.")
                                    
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
    usage_metadata: Optional[types.GenerateContentResponseUsageMetadata] = None
    if hasattr(response, "usage") and response.usage:
        # Responses API format
        if hasattr(response.usage, "input_tokens"):
            usage_metadata = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=getattr(response.usage, "input_tokens", 0),
                candidates_token_count=getattr(response.usage, "output_tokens", 0),
                total_token_count=getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0),
            )

    # Extract finish reason
    finish_reason: Optional[str] = None
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
        error_code: Optional[str] = None
        error_message: Optional[str] = None
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
    model_name: Optional[str] = None
    if hasattr(response, "model"):
        model_name = response.model
        logger.debug(f"Response from model: {model_name}")

    # Extract system fingerprint if available
    system_fingerprint: Optional[str] = None
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
      candidate_count (maps to OpenAI's n), stop_sequences, logprobs, response_logprobs
      can be set via GenerateContentConfig
    - Note: Some OpenAI-specific parameters (logit_bias, top_logprobs, user, max_tool_calls,
      metadata, prompt_cache_key, safety_identifier, truncation, store, background,
      service_tier, reasoning_effort, caching_config, grounding_config, safety_settings)
      are not available in GenerateContentConfig and cannot be mapped
    - Structured outputs: Supported via response_schema or response_mime_type in GenerateContentConfig
    - Tool choice: Uses smart defaults based on conversation state (auto/required).
      NOTE: tool_config cannot be mapped (not in GenerateContentConfig)

    File Input Support:
    - Documents (PDFs, Word docs, etc.) are uploaded via Files API when use_files_api=True
    - Images use base64 encoding
    - The Responses API supports file inputs directly in message content

    Structured Output Support:
    - JSON mode: Set response_mime_type="application/json"
    - Structured outputs with schema: Set response_schema with a JSON schema dict or Schema object
      This will be converted to Responses API's text parameter with format type "json_schema"

    Attributes:
      model: The name of the OpenAI model or (for Azure) the deployment name.
      use_files_api: Whether to use OpenAI's Files API for file uploads (default: True).
                     The Responses API supports file inputs, so Files API is enabled by default.
      url_fetch_timeout: Timeout in seconds for fetching external image URLs (default: 30.0).
                         This is used when converting image_url content parts from external URLs.
    """

    model: str = "gpt-5.1"
    use_files_api: bool = True
    url_fetch_timeout: float = 30.0
    reasoning_effort: Optional[ReasoningEffort] = None

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
        # Also track which assistant message (by index) made each tool call for proper placeholder insertion
        pending_tool_call_ids = set()
        tool_call_to_assistant_idx: Dict[str, int] = {}  # tool_call_id -> assistant message index
        has_pending_tool_calls = False  # Track if we're in the middle of a tool-calling sequence
        
        for content in llm_request.contents:
            # content_to_openai_message may return a list if there are multiple function responses
            message_or_messages = await content_to_openai_message(content, self)
            
            if isinstance(message_or_messages, list):
                for msg in message_or_messages:
                    messages.append(msg)
                    msg_idx = len(messages) - 1
                    # Track tool calls and responses
                    if msg.get("role") == "assistant" and "tool_calls" in msg:
                        # Add all tool_call_ids to pending set and track their source assistant message
                        for tool_call in msg["tool_calls"]:
                            tool_call_id = tool_call["id"]
                            pending_tool_call_ids.add(tool_call_id)
                            tool_call_to_assistant_idx[tool_call_id] = msg_idx
                        has_pending_tool_calls = len(pending_tool_call_ids) > 0
                    elif msg.get("role") == "tool" and "tool_call_id" in msg:
                        # Remove from pending when we get a tool response
                        tool_call_id = msg["tool_call_id"]
                        pending_tool_call_ids.discard(tool_call_id)
                        tool_call_to_assistant_idx.pop(tool_call_id, None)
                        has_pending_tool_calls = len(pending_tool_call_ids) > 0
            else:
                messages.append(message_or_messages)
                msg_idx = len(messages) - 1
                # Track tool calls and responses
                if message_or_messages.get("role") == "assistant" and "tool_calls" in message_or_messages:
                    # Add all tool_call_ids to pending set and track their source assistant message
                    for tool_call in message_or_messages["tool_calls"]:
                        tool_call_id = tool_call["id"]
                        pending_tool_call_ids.add(tool_call_id)
                        tool_call_to_assistant_idx[tool_call_id] = msg_idx
                    has_pending_tool_calls = len(pending_tool_call_ids) > 0
                elif message_or_messages.get("role") == "tool" and "tool_call_id" in message_or_messages:
                    # Remove from pending when we get a tool response
                    tool_call_id = message_or_messages["tool_call_id"]
                    pending_tool_call_ids.discard(tool_call_id)
                    tool_call_to_assistant_idx.pop(tool_call_id, None)
                    has_pending_tool_calls = len(pending_tool_call_ids) > 0
        
        # Calculate final state of pending tool calls after processing all messages
        has_pending_tool_calls = len(pending_tool_call_ids) > 0
        
        # Check if the last message was a tool response (indicates we just got tool results)
        last_message_was_tool_response = False
        if messages:
            last_message = messages[-1]
            if last_message.get("role") == "tool":
                last_message_was_tool_response = True
                logger.debug("Last message was a tool response - model can provide final answer or call more tools")
        
        # Check if we have any tool calls in the conversation history (indicates we're in a tool-using workflow)
        has_tool_calls_in_history = False
        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                has_tool_calls_in_history = True
                break
        
        if has_pending_tool_calls:
            logger.debug(f"Pending tool call IDs after message processing: {pending_tool_call_ids}")
        
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
        
        # Handle pending tool calls without responses - inject placeholder responses
        # OpenAI API requires every tool_call to have a corresponding tool response
        # IMPORTANT: Placeholders must be inserted immediately after their corresponding assistant message,
        # not appended at the end, to maintain proper message sequence
        if pending_tool_call_ids:
            logger.warning(
                f"Found {len(pending_tool_call_ids)} tool call(s) without responses: {pending_tool_call_ids}. "
                "Injecting placeholder tool responses to prevent API errors."
            )
            
            # Group pending tool calls by their source assistant message index
            # so we can insert all placeholders for each assistant message together
            assistant_idx_to_tool_calls: Dict[int, List[str]] = {}
            for tool_call_id in pending_tool_call_ids:
                assistant_idx = tool_call_to_assistant_idx.get(tool_call_id)
                if assistant_idx is not None:
                    if assistant_idx not in assistant_idx_to_tool_calls:
                        assistant_idx_to_tool_calls[assistant_idx] = []
                    assistant_idx_to_tool_calls[assistant_idx].append(tool_call_id)
                else:
                    # Fallback: if we lost track of the assistant message, log error
                    # This shouldn't happen but handle it gracefully
                    logger.error(
                        f"Lost track of assistant message for tool_call_id: {tool_call_id}. "
                        f"Will append placeholder at end (may cause API errors)."
                    )
            
            # Insert placeholders in reverse order of assistant message index
            # to avoid index shifting issues when inserting
            for assistant_idx in sorted(assistant_idx_to_tool_calls.keys(), reverse=True):
                tool_call_ids_for_assistant = assistant_idx_to_tool_calls[assistant_idx]
                
                # Find the insertion point: right after all existing tool responses for this assistant message
                # Start from the message after the assistant message
                insert_idx = assistant_idx + 1
                
                # Skip past any existing tool responses that immediately follow
                while insert_idx < len(messages) and messages[insert_idx].get("role") == "tool":
                    insert_idx += 1
                
                # Insert placeholders for all missing tool calls from this assistant message
                for tool_call_id in tool_call_ids_for_assistant:
                    placeholder_response = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": "[Tool execution was interrupted or response was lost. Please retry the operation.]"
                    }
                    messages.insert(insert_idx, placeholder_response)
                    logger.debug(
                        f"Injected placeholder tool response for tool_call_id: {tool_call_id} "
                        f"at index {insert_idx} (after assistant message at index {assistant_idx})"
                    )
                    insert_idx += 1  # Move insertion point for next placeholder
            
            # Handle any tool calls where we lost track of the assistant message (fallback)
            orphan_tool_calls = [
                tc_id for tc_id in pending_tool_call_ids 
                if tool_call_to_assistant_idx.get(tc_id) is None
            ]
            for tool_call_id in orphan_tool_calls:
                placeholder_response = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "[Tool execution was interrupted or response was lost. Please retry the operation.]"
                }
                messages.append(placeholder_response)
                logger.warning(
                    f"Appended orphan placeholder tool response for tool_call_id: {tool_call_id} at end of messages"
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
        
        # Track Pydantic model for structured output (used with parse() method)
        pydantic_model_for_parse = None
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
        if self.reasoning_effort:
            request_params["reasoning"] = {"effort": self.reasoning_effort}

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
            # Smart default strategy for tool_choice in agentic environments:
            # NOTE: tool_config cannot be mapped (not in GenerateContentConfig), so we use smart defaults
            # - If we have pending tool calls, we're waiting for tool results, so use "auto" to allow final answer
            # - If the last message was a tool response, we just got tool results, so use "auto" to allow final answer
            # - If we have tool calls in history, we're in a tool-using workflow, so use "auto" (model can decide to continue or finish)
            # - Otherwise (fresh conversation with tools available), use "required" to ensure tools are called when available
            # This ensures tools are used in agentic environments while preventing loops
            if has_pending_tool_calls:
                # We're in the middle of a tool-calling sequence (waiting for tool results)
                tool_choice = "auto"
                logger.debug("Pending tool calls detected - using 'auto' to allow final answer after tool results")
            elif last_message_was_tool_response:
                # We just received tool results - allow model to provide final answer or call more tools
                tool_choice = "auto"
                logger.debug("Last message was tool response - using 'auto' to allow final answer or additional tool calls")
            elif has_tool_calls_in_history:
                # We're in a tool-using workflow - let model decide whether to continue or finish
                tool_choice = "auto"
                logger.debug("Tool calls in history - using 'auto' to let model decide next step")
            else:
                # Fresh conversation with tools available - use "required" to ensure tools are called
                # This is important for agentic behavior where tools should be used when available
                tool_choice = "required"
                logger.debug("Fresh conversation with tools - using 'required' to ensure tools are called")
            
            # NOTE: tool_config cannot be mapped (not in GenerateContentConfig).
            # If tool_config is needed, it must be added to the upstream GenAI SDK first.
            # Tool choice is determined by the smart default strategy above.
            # Responses API uses tool_choice (same as Chat Completions)
            request_params["tool_choice"] = tool_choice
            # Responses API supports parallel_tool_calls
            # Default to True for better performance (all modern models support it)
            request_params["parallel_tool_calls"] = True
        # Map Google GenAI GenerateContentConfig parameters to Responses API
        # NOTE: Only access attributes that exist on google.genai.types.GenerateContentConfig.
        # The ADK SDK re-exports those autogenerated Pydantic models and we *must* keep
        # them untouched. If a new OpenAI parameter is needed, add it to the upstream
        # GenAI types first, regenerate, then reference it here. Adding ad-hoc fields
        # directly on the config object will fail validation.
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
        # - response_logprobs → logprobs (enables logprobs if True)
        # - system_instruction → instructions (direct)
        # - response_schema/response_mime_type → text (converted format for Responses API)
        # - tools → tools (converted to Responses API format)
        # - parallel_tool_calls → parallel_tool_calls (default: True)
        # - tool_choice: Uses smart defaults (auto/required) based on conversation state
        #
        # Parameters that cannot be mapped (not in GenerateContentConfig):
        # - top_logprobs, user, max_tool_calls, metadata, prompt_cache_key, safety_identifier,
        #   truncation, store, background, service_tier, reasoning_effort, caching_config,
        #   grounding_config, safety_settings, tool_config
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

            # NOTE: The following OpenAI Responses API parameters cannot be mapped because
            # they have no equivalent in google.genai.types.GenerateContentConfig:
            # - top_logprobs: Cannot be mapped (not in GenerateContentConfig)
            # - user: Cannot be mapped (not in GenerateContentConfig)
            # - max_tool_calls: Cannot be mapped (not in GenerateContentConfig)
            # - metadata: Cannot be mapped (not in GenerateContentConfig)
            # - prompt_cache_key: Cannot be mapped (not in GenerateContentConfig)
            # - safety_identifier: Cannot be mapped (not in GenerateContentConfig)
            # - truncation: Cannot be mapped (not in GenerateContentConfig)
            # - store: Cannot be mapped (not in GenerateContentConfig)
            # - background: Cannot be mapped (not in GenerateContentConfig)
            # - service_tier: Cannot be mapped (not in GenerateContentConfig)
            # - reasoning_effort: Cannot be mapped (not in GenerateContentConfig)
            # - caching_config: Cannot be mapped (not in GenerateContentConfig)
            # - grounding_config: Cannot be mapped (not in GenerateContentConfig)
            # - safety_settings: Cannot be mapped (not in GenerateContentConfig)
            # If these are needed, they must be added to the upstream GenAI SDK first.

            # Handle structured output / text format
            # IMPORTANT: Skip text format entirely when tools are present
            # because text format (even without strict mode) conflicts with:
            # 1. Function calls (tools) - can't make function calls with JSON schema format
            # 2. Planner text format (e.g., PlanReActPlanner needs /*FINAL_ANSWER*/ tags)
            # 
            # Priority: Google GenAI types (response_schema, response_mime_type) first
            # Responses API uses 'text' parameter with 'format' for structured output:
            # 1. JSON mode: {"format": {"type": "json_object"}}
            # 2. Structured outputs with schema: {"format": {"type": "json_schema", "name": "...", "schema": {...}, "strict": true}}
            
            # Check if tools are present - if so, skip text format entirely
            has_tools_for_text_format = "tools" in request_params and request_params.get("tools")
            if has_tools_for_text_format:
                logger.debug(
                    "Tools are present - skipping text format to allow function calls and planner text format"
                )
            else:
                try:
                    text_format_set = False
                    # First, check for Google GenAI response_schema or response_mime_type
                    # (These are handled below in the response_schema section)
                    
                    if not text_format_set:
                        # Check for response_schema (JSON schema for structured outputs)
                        response_schema = getattr(
                            llm_request.config, "response_schema", None
                        )
                        if response_schema is not None:
                            # Convert ADK schema to Responses API text format
                            # Responses API requires: {"type": "json_schema", "name": "...", "schema": {...}, "strict": true}
                            schema_name = None
                            schema_dict = None
                            
                            if isinstance(response_schema, dict):
                                # If it's already a dict, check if it's already in OpenAI format
                                if "name" in response_schema and "schema" in response_schema:
                                    # Already in OpenAI format (nested json_schema)
                                    # Convert to Responses API text format - flatten the structure
                                    request_params["text"] = {
                                        "format": {
                                            "type": "json_schema",
                                            "name": response_schema.get("name", "ResponseSchema"),
                                            "schema": response_schema.get("schema", {}),
                                            "strict": response_schema.get("strict", True),
                                        }
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
                                        # It's a Pydantic model class - store it for use with parse() method
                                        # Responses API parse() method accepts Pydantic models directly
                                        pydantic_model_for_parse = response_schema
                                        
                                        # Also get JSON schema for logging/debugging
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
                            
                            # For Pydantic models, we'll use create() with response_format (strict schema)
                            # For dict schemas, we still need to create response_format structure
                            if pydantic_model_for_parse is not None:
                                # Pydantic model detected - will use create() with response_format (strict schema)
                                logger.info(
                                    f"Pydantic model detected for structured output: {pydantic_model_for_parse.__name__}. "
                                    f"Will use create() with response_format (strict schema)."
                                )
                                # Don't set response_format here - we'll set it later with strict schema
                            elif schema_dict is not None:
                                # Dict-based schema - create text format structure for Responses API
                                # Use schema title if available, otherwise use the name we extracted
                                if isinstance(schema_dict, dict) and "title" in schema_dict:
                                    schema_name = schema_dict.get("title", schema_name or "ResponseSchema")
                                
                                # Ensure strict mode compliance: add additionalProperties: false to all object types
                                # Make a deep copy to avoid mutating the original
                                strict_schema_dict = copy.deepcopy(schema_dict)
                                _ensure_strict_json_schema(strict_schema_dict)
                                
                                # Responses API uses 'text' parameter with 'format' for structured output
                                request_params["text"] = {
                                    "format": {
                                        "type": "json_schema",
                                        "name": schema_name or "ResponseSchema",
                                        "schema": strict_schema_dict,
                                        "strict": True  # Always enable strict mode for structured outputs
                                    }
                                }
                                logger.info(
                                    f"Converted response_schema to text format: "
                                    f"name={schema_name or 'ResponseSchema'}, "
                                    f"strict=True, "
                                    f"schema_keys={list(strict_schema_dict.keys()) if isinstance(strict_schema_dict, dict) else 'N/A'}"
                                    )
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
                                # Responses API uses 'text' parameter for JSON mode
                                request_params["text"] = {"format": {"type": "json_object"}}
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
        
        # Handle text format for structured output (Responses API)
        # IMPORTANT: Skip text format if tools are present, as format conflicts with tools
        # Responses API uses 'text' parameter with 'format' for structured output
        # The text format structure:
        # - For json_object: {"format": {"type": "json_object"}}
        # - For json_schema: {"format": {"type": "json_schema", "name": "...", "schema": {...}, "strict": true}}
        if "text" in request_params:
            # Check if tools are present - if so, remove text format to allow function calls
            if "tools" in request_params and request_params.get("tools"):
                logger.warning(
                    "text format is set but tools are also present. "
                    "Removing text format to allow function calls and planner text format. "
                    "Text format conflicts with tools/function calls."
                )
                request_params.pop("text")
            else:
                # For json_schema format, ensure strict mode and additionalProperties: false
                text_param = request_params["text"]
                if isinstance(text_param, dict) and "format" in text_param:
                    format_obj = text_param["format"]
                    if isinstance(format_obj, dict) and format_obj.get("type") == "json_schema":
                        # Schema is directly in format_obj (Responses API structure)
                        if "schema" in format_obj and isinstance(format_obj["schema"], dict):
                            schema_dict = format_obj["schema"]
                            # Make a deep copy to avoid mutating the original
                            schema_dict = copy.deepcopy(schema_dict)
                            _ensure_strict_json_schema(schema_dict)
                            # Update the schema in the text format
                            format_obj["schema"] = schema_dict
                            # Ensure strict mode is enabled
                            format_obj["strict"] = True
                            logger.info(
                                f"Ensured strict JSON schema compliance for text format: "
                                f"name={format_obj.get('name', 'ResponseSchema')}, strict=True"
                            )
                    
        # Log text format if present (for structured outputs)
        if "text" in request_params:
            text_param = request_params["text"]
            logger.info(f"TEXT FORMAT BEING SENT: {json.dumps(text_param, indent=2, default=str)}")
        
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
                
                # For structured outputs with Pydantic models, always use create() with response_format
                # This ensures we can apply strict schema validation (additionalProperties: false)
                if pydantic_model_for_parse is not None:
                    # Convert Pydantic model to response_format with strict schema
                    logger.info(
                        f"Pydantic model detected (streaming): {pydantic_model_for_parse.__name__}. "
                        f"Using create() with response_format (strict schema)."
                    )
                    # Get JSON schema from Pydantic model
                    if hasattr(pydantic_model_for_parse, "model_json_schema"):
                        schema_dict = pydantic_model_for_parse.model_json_schema()
                    elif hasattr(pydantic_model_for_parse, "schema"):
                        schema_dict = pydantic_model_for_parse.schema()
                    else:
                        # Fallback: try to instantiate and get schema
                        try:
                            instance = pydantic_model_for_parse()
                            if hasattr(instance, "model_json_schema"):
                                schema_dict = instance.model_json_schema()
                            elif hasattr(instance, "schema"):
                                schema_dict = instance.schema()
                            else:
                                schema_dict = None
                        except Exception as e:
                            logger.debug(f"Could not extract schema from Pydantic model (streaming): {e}")
                            schema_dict = None
                    
                    if schema_dict is not None:
                        # Ensure strict mode compliance (additionalProperties: false for all objects)
                        strict_schema_dict = copy.deepcopy(schema_dict)
                        _ensure_strict_json_schema(strict_schema_dict)
                        
                        # Use schema title if available, otherwise use model name
                        schema_name = strict_schema_dict.get("title", pydantic_model_for_parse.__name__)
                        
                        # Responses API uses 'text' parameter with 'format' for structured output
                        request_params["text"] = {
                            "format": {
                                "type": "json_schema",
                                "name": schema_name,
                                "schema": strict_schema_dict,
                                "strict": True,  # Enable strict mode
                            }
                        }
                        logger.info(
                            f"Using create() (streaming) with strict JSON schema for {pydantic_model_for_parse.__name__}, "
                            f"tools count: {len(request_params.get('tools', []))}"
                        )
                    else:
                        logger.warning(
                            f"Could not extract JSON schema from {pydantic_model_for_parse.__name__}, "
                            f"falling back to regular create() call"
                        )
                
                # Use create() method for all requests (streaming)
                logger.info(f"Calling OpenAI Responses API create() with stream=True, tools count: {len(request_params.get('tools', []))}")
                stream_response = await client.responses.create(**request_params)

                # Accumulate content and tool calls across events
                accumulated_text = ""
                accumulated_tool_calls: Dict[str, Dict[str, Any]] = {}
                finish_reason: Optional[str] = None
                usage_metadata: Optional[types.GenerateContentResponseUsageMetadata] = None
                model_name: Optional[str] = None
                system_fingerprint: Optional[str] = None

                async for event in stream_response:
                    # Track model name and response ID from response.created event
                    event_type = getattr(event, "type", None)
                    if not event_type:
                        continue
                    
                    # Handle response.created event
                    if event_type == "response.created":
                        if hasattr(event, "response") and hasattr(event.response, "model"):
                            model_name = event.response.model
                        # Note: response.id is available but not currently used
                        # Could be used for conversation state management in future
                    
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

                        function_call = types.FunctionCall(
                            id=tool_call_data["id"],
                            name=tool_call_data["function"]["name"],
                            args=function_args,
                        )
                        final_parts.append(types.Part(function_call=function_call))

                final_content = (
                    types.Content(role="model", parts=final_parts)
                    if final_parts
                    else None
                )

                # Create GenerateContentResponse to match Google GenAI structure
                # IMPORTANT: function_calls is a computed property that extracts from content.parts
                # We don't need to set it directly - function calls are already in final_parts
                final_response: LlmResponse
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
                # For structured outputs with Pydantic models, always use create() with response_format
                # This ensures we can apply strict schema validation (additionalProperties: false)
                if pydantic_model_for_parse is not None:
                    # Convert Pydantic model to response_format with strict schema
                    logger.info(
                        f"Pydantic model detected: {pydantic_model_for_parse.__name__}. "
                        f"Using create() with response_format (strict schema)."
                    )
                    # Get JSON schema from Pydantic model
                    if hasattr(pydantic_model_for_parse, "model_json_schema"):
                        schema_dict = pydantic_model_for_parse.model_json_schema()
                    elif hasattr(pydantic_model_for_parse, "schema"):
                        schema_dict = pydantic_model_for_parse.schema()
                    else:
                        # Fallback: try to instantiate and get schema
                        try:
                            instance = pydantic_model_for_parse()
                            if hasattr(instance, "model_json_schema"):
                                schema_dict = instance.model_json_schema()
                            elif hasattr(instance, "schema"):
                                schema_dict = instance.schema()
                            else:
                                schema_dict = None
                        except Exception as e:
                            logger.debug(f"Could not extract schema from Pydantic model: {e}")
                            schema_dict = None
                    
                    if schema_dict is not None:
                        # Ensure strict mode compliance (additionalProperties: false for all objects)
                        strict_schema_dict = copy.deepcopy(schema_dict)
                        _ensure_strict_json_schema(strict_schema_dict)
                        
                        # Use schema title if available, otherwise use model name
                        schema_name = strict_schema_dict.get("title", pydantic_model_for_parse.__name__)
                        
                        # Responses API uses 'text' parameter with 'format' for structured output
                        request_params["text"] = {
                            "format": {
                                "type": "json_schema",
                                "name": schema_name,
                                "schema": strict_schema_dict,
                                "strict": True,  # Enable strict mode
                            }
                        }
                        logger.info(
                            f"Using create() with strict JSON schema for {pydantic_model_for_parse.__name__}, "
                            f"tools count: {len(request_params.get('tools', []))}"
                        )
                    else:
                        logger.warning(
                            f"Could not extract JSON schema from {pydantic_model_for_parse.__name__}, "
                            f"falling back to regular create() call"
                        )
                
                # Use create() method for all requests
                logger.info(f"Calling OpenAI Responses API create() with stream=False, tools count: {len(request_params.get('tools', []))}")
                response = await client.responses.create(**request_params)
                
                # Pass pydantic_model for response parsing (create() returns structured output in response_format)
                llm_response = await openai_response_to_llm_response(
                    response, 
                    url_fetch_timeout=self.url_fetch_timeout,
                    pydantic_model=pydantic_model_for_parse
                )
                yield llm_response

        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            yield LlmResponse(error_code="RATE_LIMIT_ERROR", error_message=str(e))
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {e}")
            yield LlmResponse(error_code="AUTHENTICATION_ERROR", error_message=str(e))
        except openai.BadRequestError as e:
            logger.error(f"OpenAI bad request: {e}")
            yield LlmResponse(error_code="BAD_REQUEST_ERROR", error_message=str(e))
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API connection error: {e}")
            yield LlmResponse(error_code="CONNECTION_ERROR", error_message=str(e))
        except openai.APITimeoutError as e:
            logger.error(f"OpenAI API timeout: {e}")
            yield LlmResponse(error_code="TIMEOUT_ERROR", error_message=str(e))
        except openai.APIError as e:
            logger.error(f"Error calling OpenAI API: {e}")
            yield LlmResponse(error_code="OPENAI_API_ERROR", error_message=str(e))

    def _is_image_mime_type(self, mime_type: str) -> bool:
        """Check if MIME type is an image type."""
        return mime_type.startswith("image/")

    def _is_supported_files_api_mime_type(self, mime_type: str) -> bool:
        """Check if MIME type is supported by OpenAI Files API for context stuffing.
        
        Supported formats: .art, .bat, .brf, .c, .cls, .css, .diff, .eml, .es, .h, .hs, 
        .htm, .html, .ics, .ifb, .java, .js, .json, .ksh, .ltx, .mail, .markdown, .md, 
        .mht, .mhtml, .mjs, .nws, .patch, .pdf, .pl, .pm, .pot, .py, .scala, .sh, .shtml, 
        .srt, .sty, .tex, .text, .txt, .vcf, .vtt, .xml, .yaml, .yml
        
        Note: Images are NOT supported and must use base64 encoding.
        """
        # Map of supported MIME types to their extensions
        supported_mime_types = {
            # Text files
            "text/plain": True,
            "text/markdown": True,
            "text/html": True,
            "text/css": True,
            "text/xml": True,
            "text/yaml": True,
            "text/x-yaml": True,
            "text/x-python": True,
            "text/x-java": True,
            "text/x-javascript": True,
            "text/x-scala": True,
            "text/x-shellscript": True,
            "text/x-perl": True,
            "text/x-latex": True,
            "text/x-tex": True,
            "text/x-diff": True,
            "text/x-patch": True,
            "text/vcard": True,
            "text/vtt": True,
            "text/srt": True,
            "text/x-mail": True,
            "message/rfc822": True,  # .eml
            # Documents
            "application/pdf": True,
            "application/json": True,
            "application/xml": True,
            "application/yaml": True,
            "application/x-yaml": True,
            "application/javascript": True,
            "application/x-javascript": True,
            "application/x-httpd-php": True,
            # Code files (by extension mapping)
            "text/x-c": True,
            "text/x-c++": True,
            "text/x-csharp": True,
            "text/x-go": True,
            "text/x-rust": True,
            # Other document types
            "application/x-httpd-erb": True,
            "application/x-sh": True,
            "application/x-perl": True,
            "application/x-python": True,
            "application/x-java": True,
            "application/x-scala": True,
            "application/x-latex": True,
            "application/x-tex": True,
            "application/x-diff": True,
            "application/x-patch": True,
            "application/x-vcard": True,
            "application/x-subrip": True,  # .srt
            "application/x-ics": True,  # .ics
            "application/x-ifb": True,  # .ifb
            "application/x-mht": True,  # .mht
            "application/x-mhtml": True,  # .mhtml
            "application/x-news": True,  # .nws
            "application/x-art": True,  # .art
            "application/x-bat": True,  # .bat
            "application/x-brf": True,  # .brf
            "application/x-cls": True,  # .cls
            "application/x-es": True,  # .es
            "application/x-h": True,  # .h
            "application/x-hs": True,  # .hs
            "application/x-ksh": True,  # .ksh
            "application/x-ltx": True,  # .ltx
            "application/x-mail": True,  # .mail
            "application/x-mjs": True,  # .mjs
            "application/x-pl": True,  # .pl
            "application/x-pm": True,  # .pm
            "application/x-pot": True,  # .pot
            "application/x-shtml": True,  # .shtml
            "application/x-sty": True,  # .sty
            "application/x-vcf": True,  # .vcf
        }
        
        # Check exact match first
        if mime_type in supported_mime_types:
            return True
        
        # Check if it's a text/* type (many text files are supported)
        if mime_type.startswith("text/"):
            # Exclude image types that might be misclassified
            if not self._is_image_mime_type(mime_type):
                return True
        
        return False

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
        """Handle file data using Files API (for supported documents only) or base64 encoding (for images and unsupported types).
        
        Note: Images must use base64 encoding as they are not supported by Files API for context stuffing.
        Only supported document types (.pdf, .txt, .md, .json, etc.) can use Files API.
        """
        if part.inline_data:
            # Handle inline data
            mime_type = part.inline_data.mime_type or "application/octet-stream"
            data = part.inline_data.data
            display_name = part.inline_data.display_name

            # Images must always use base64 encoding (Files API doesn't support images for context stuffing)
            if self._is_image_mime_type(mime_type):
                data_b64 = base64.b64encode(data).decode()
                data_uri = f"data:{mime_type};base64,{data_b64}"
                
                logger.debug(
                    f"Encoding image {mime_type} as base64 data URI "
                    f"(Images must use base64, not Files API, size: {len(data)} bytes)"
                )
                
                return {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                }

            # For non-image files, try Files API if enabled and supported
            if self.use_files_api and self._is_supported_files_api_mime_type(mime_type):
                try:
                    file_id = await self._upload_file_to_openai(
                        data, mime_type, display_name
                    )
                    # Responses API supports file inputs in message content
                    logger.info(f"File uploaded successfully to Files API, file_id: {file_id}, mime_type: {mime_type}")
                    return {
                        "type": "file",
                        "file_id": file_id,
                    }
                except Exception as e:
                    logger.warning(
                        f"Failed to upload file to OpenAI Files API: {e}. Falling back to base64 encoding."
                    )
                    # Fall through to base64 encoding only if upload failed
            elif self.use_files_api:
                logger.debug(
                    f"MIME type {mime_type} is not supported by Files API for context stuffing. "
                    f"Using base64 encoding instead."
                )

            # Use base64 if Files API is disabled, not supported, or upload failed
            # For non-image files, return as text with data URI since Responses API
            # doesn't support non-image base64 data URIs in the same way as images
            data_b64 = base64.b64encode(data).decode()
            data_uri = f"data:{mime_type};base64,{data_b64}"
            
            logger.debug(
                f"Encoding {mime_type} as base64 data URI "
                f"(Files API disabled/unsupported/failed, size: {len(data)} bytes, base64: {len(data_b64)} chars)"
            )
            
            # Non-image files should be returned as text with data URI
            # (Images are already handled and returned earlier, so we never reach here for images)
            return {
                "type": "text",
                "text": data_uri,
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

    def _preprocess_request(self, llm_request: LlmRequest) -> None:
        """Preprocesses the request before sending to OpenAI."""
        # Set model if not specified
        if not llm_request.model:
            llm_request.model = self.model
