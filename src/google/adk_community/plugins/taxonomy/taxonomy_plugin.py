# Copyright 2026 Google LLC
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

"""TaxonomyPlugin — ADK BasePlugin for pluggable taxonomy policy enforcement."""

from __future__ import annotations

import logging
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Optional

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.skills import prompt
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from .policy import DefaultSkillPolicy
from .policy import SkillPolicy
from .policy import TaxonomyResolver
from .taxonomy_config import TaxonomyRegistry

logger = logging.getLogger("google_adk_community." + __name__)

_ACTIVE_TAXONOMIES_STATE_KEY = "_active_taxonomies"

_SKILL_GATE_TOOLS = frozenset({
    "list_skills",
    "load_skill",
    "load_skill_resource",
    "run_skill_script",
})


class TaxonomyPlugin(BasePlugin):
  """Native ADK Plugin enforcing pluggable taxonomy policies."""

  def __init__(
      self,
      name: str = "taxonomy_plugin",
      *,
      taxonomy_registry: Optional[TaxonomyRegistry] = None,
      resolver: Optional[TaxonomyResolver] = None,
      policy: Optional[SkillPolicy] = None,
  ):
    super().__init__(name)
    self.taxonomy_registry = taxonomy_registry or TaxonomyRegistry()
    self.resolver = resolver
    self.policy = policy or DefaultSkillPolicy(self.taxonomy_registry)
    if self.policy and getattr(self.policy, "registry", None) is None:
      try:
        self.policy.registry = self.taxonomy_registry
      except Exception:
        pass

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """Resolves active taxonomies and stores them in session state."""
    if not self.resolver:
      return None

    active_taxonomies = await self.resolver.resolve_taxonomies(
        callback_context, llm_request
    )
    callback_context.state[_ACTIVE_TAXONOMIES_STATE_KEY] = active_taxonomies

    logger.debug(
        "[%s] Resolved active taxonomies: %s", self.name, active_taxonomies
    )

    if self.policy:
      orig_instructions = llm_request.config.system_instruction or ""
      shaped_instructions = self.policy.shape_system_instruction(
          callback_context, active_taxonomies, orig_instructions
      )
      if shaped_instructions != orig_instructions:
        logger.debug(
            "[%s] Active taxonomy dynamic system prompt shaping applied.",
            self.name,
        )
        llm_request.config.system_instruction = shaped_instructions

    return None

  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[dict]:
    """Intercepts skill tools to enforce taxonomy policy and path validation."""
    if tool.name not in _SKILL_GATE_TOOLS:
      return None

    assert tool is not None, "Intercepted tool cannot be None"
    assert isinstance(tool_args, dict), "tool_args must be a dictionary"
    assert tool_context is not None, "tool_context cannot be None"

    active_taxonomies = (
        tool_context.state.get(_ACTIVE_TAXONOMIES_STATE_KEY) or []
    )

    if tool.name == "list_skills":
      return self._filter_list_skills(tool, tool_context, active_taxonomies)

    skill_name = tool_args.get("skill_name")
    if not skill_name:
      return None

    # Inline path validation (avoids importing private _validate_path_segment)
    if (
        not skill_name
        or "\x00" in skill_name
        or "/" in skill_name
        or "\\" in skill_name
        or skill_name in (".", "..")
        or ".." in skill_name.split("/")
    ):
      return {
          "error": f"Invalid skill_name parameter: {skill_name!r}",
          "error_code": "INVALID_ARGUMENTS",
      }

    file_path = tool_args.get("file_path")
    if file_path:
      posix_p = PurePosixPath(file_path)
      win_p = PureWindowsPath(file_path)
      
      # Block absolute paths or presence of a drive letter
      if posix_p.is_absolute() or win_p.is_absolute() or win_p.drive:
        return {
            "error": f"Absolute path blocked: {file_path}",
            "error_code": "INVALID_ARGUMENTS",
        }
      
      # Block traversal segments
      if ".." in posix_p.parts or ".." in win_p.parts:
        return {
            "error": f"Path traversal attempt blocked: {file_path}",
            "error_code": "INVALID_ARGUMENTS",
        }

    if self.policy and self.resolver:
      toolset = getattr(tool, "_toolset", None)
      if toolset:
        skill = await toolset._get_or_fetch_skill(
            skill_name, tool_context.invocation_id
        )
        if skill and not self.policy.is_skill_allowed(
            skill, tool_context, active_taxonomies
        ):
          logger.warning(
              "[%s] Skill '%s' blocked by policy. Active taxonomies: %s",
              self.name,
              skill_name,
              active_taxonomies,
          )
          return {
              "error": (
                  f"Access to skill '{skill_name}' is not permitted"
                  " under active policy constraints."
              ),
              "error_code": "SKILL_NOT_PERMITTED",
          }

    return None

  def _filter_list_skills(
      self, tool: BaseTool, tool_context: ToolContext, active_taxonomies: list[str]
  ) -> Optional[dict]:
    """Filters the list_skills result to only show policy-permitted skills."""
    if not self.policy or not self.resolver:
      return None

    toolset = getattr(tool, "_toolset", None)
    if not toolset:
      return None

    all_skills = toolset._list_skills()
    allowed_skills = [
        skill
        for skill in all_skills
        if self.policy.is_skill_allowed(skill, tool_context, active_taxonomies)
    ]

    # Reorder and prioritize skills dynamically
    prioritized_skills = self.policy.prioritize_skills(
        allowed_skills, tool_context, active_taxonomies
    )

    shaped_skills = []
    for skill in prioritized_skills:
      original_desc = skill.frontmatter.description or ""
      shaped_desc = self.policy.shape_description(skill, tool_context, original_desc)
      new_skill = self.policy.shape_skill(skill, tool_context, shaped_desc)
      shaped_skills.append(new_skill)

    logger.debug(
        "[%s] Filtered skills: %d/%d visible",
        self.name,
        len(shaped_skills),
        len(all_skills),
    )
    return {"result": prompt.format_skills_as_xml(shaped_skills)}


  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict,
  ) -> Optional[dict]:
    """Applies dynamic instruction shaping to load_skill results."""
    if tool.name != "load_skill":
      return None
    if not self.policy or not self.resolver:
      return None
    if not isinstance(result, dict) or "instructions" not in result:
      return None

    skill_name = tool_args.get("skill_name")
    if not skill_name:
      return None

    toolset = getattr(tool, "_toolset", None)
    if not toolset:
      return None

    skill = await toolset._get_or_fetch_skill(
        skill_name, tool_context.invocation_id
    )
    if not skill:
      return None

    shaped_instructions = self.policy.shape_instructions(
        skill, tool_context, result["instructions"]
    )

    if shaped_instructions != result["instructions"]:
      logger.debug(
          "[%s] Shaped instructions for skill '%s'",
          self.name,
          skill_name,
      )

    shaped_result = dict(result)
    shaped_result["instructions"] = shaped_instructions
    return shaped_result
