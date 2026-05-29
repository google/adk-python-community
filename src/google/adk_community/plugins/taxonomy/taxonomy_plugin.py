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
from typing import Any
from typing import Optional

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.skills import prompt
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

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
    self.policy = policy

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
      if ".." in file_path or file_path.startswith(("/", "\\")):
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

    logger.debug(
        "[%s] Filtered skills: %d/%d visible",
        self.name,
        len(allowed_skills),
        len(all_skills),
    )
    return {"result": prompt.format_skills_as_xml(allowed_skills)}

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
