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

"""Abstract interfaces for taxonomy resolution and skill policy enforcement."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any, Optional

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.skills.models import Skill

logger = logging.getLogger("google_adk_community." + __name__)

class TaxonomyResolver(ABC):
  """Abstract base class for taxonomy resolution.
  
  Resolvers analyze context and LLM history to determine which taxonomy
  classification domains (e.g. URI strings) are currently active and relevant.
  """

  @abstractmethod
  async def resolve_taxonomies(
      self, context: ReadonlyContext, llm_request: LlmRequest
  ) -> list[str]:
    """Resolves active taxonomy domain URIs from context and LLM history.
    
    Args:
      context: The current read-only execution context.
      llm_request: The upcoming LLM request holding prompt configurations.
      
    Returns:
      A list of resolved active taxonomy strings/URIs.
    """
    pass


class TaxonomyPipeline(TaxonomyResolver):
  """Executes a sequence of taxonomy resolvers in order (multi-step pipeline).
  
  This implements a composite/pipeline pattern to merge active taxonomy domains
  identified by multiple independent heuristics (e.g. lexical, model-based).
  """

  def __init__(self, resolvers: list[TaxonomyResolver]):
    self.resolvers = resolvers

  async def resolve_taxonomies(
      self, context: ReadonlyContext, llm_request: LlmRequest
  ) -> list[str]:
    # Aggregates unique taxonomy domains across all registered resolvers
    active_domains: set[str] = set()
    for resolver in self.resolvers:
      domains = await resolver.resolve_taxonomies(context, llm_request)
      if domains:
        active_domains.update(domains)
    return list(active_domains)


class DefaultKeywordResolver(TaxonomyResolver):
  """Declarative, configuration-driven keyword/phrase resolver.
  
  Scans user prompt history for triggering phrases defined directly inside each
  taxonomy term's triggers list or alt_labels, resolving active domains natively.
  """

  def __init__(self, registry: Any):
    self.registry = registry

  async def resolve_taxonomies(self, context: ReadonlyContext, llm_request: LlmRequest) -> list[str]:
    active_domains: set[str] = set()
    
    for term_id in self.registry.list_ids():
      term = self.registry.get_term(term_id)
      if term:
        triggers = getattr(term, "triggers", [])
        if not triggers and hasattr(term, "model_extra"):
          triggers = (term.model_extra or {}).get("triggers", [])
        
        # Fall back to alt_labels as secondary keyword triggers
        if not triggers and hasattr(term, "alt_labels"):
          triggers = term.alt_labels
          
        if triggers:
          for turn in llm_request.contents:
            for part in turn.parts:
              if part.text:
                text_upper = part.text.upper()
                if any(str(phrase).upper() in text_upper for phrase in triggers):
                  active_domains.add(term_id)
                  break

    return list(active_domains)


class SkillPolicy(ABC):
  """Abstract policy engine determining skill execution permissions and instruction shaping.
  
  This class defines the interface for two main responsibilities:
  1. Access Control (Authorization): Blocking or permitting skills based on active taxonomies.
  2. Cognitive Steering (Behavioral Shaping): Altering skill instructions, descriptions,
     prioritization, and global system prompts to steer agent execution dynamically.
     
  Implements the Hook Method pattern, providing concrete default pass-throughs
  for steering while keeping authorization and core shaping abstract.
  """

  registry: Optional[Any] = None

  @abstractmethod
  def is_skill_allowed(
      self,
      skill: Skill,
      context: ReadonlyContext,
      active_taxonomies: list[str],
  ) -> bool:
    """Determines if a skill can be loaded/used under the active taxonomies and context.
    
    Args:
      skill: The target Skill model instance.
      context: The read-only interaction context.
      active_taxonomies: The list of currently active taxonomy domains.
      
    Returns:
      True if the skill is permitted to run, False otherwise.
    """
    pass

  @abstractmethod
  def shape_instructions(
      self,
      skill: Skill,
      context: ReadonlyContext,
      original_instructions: str,
  ) -> str:
    """Applies dynamic instruction shaping/guardrails to a skill's instructions.
    
    Use this to append safety restrictions, enforce compliance constraints,
    or adjust operating parameters of a skill before execution.
    """
    pass

  def shape_description(
      self,
      skill: Skill,
      context: ReadonlyContext,
      original_description: str,
  ) -> str:
    """Applies dynamic description shaping before the tool reaches the agent.
    
    This can be used to emphasize specific features of a skill to the LLM or
    prune redundant information to fit within context limits.
    """
    return original_description

  def shape_system_instruction(
      self,
      context: ReadonlyContext,
      active_taxonomies: list[str],
      original_instructions: str,
  ) -> str:
    """Applies dynamic instruction shaping to the global agent system instructions.

    Use this to dynamically inject directives (e.g. telling the LLM to trigger
    certain tools almost by default or prioritize specific workflows) depending
    on the current active taxonomy classification.
    """
    return original_instructions

  def prioritize_skills(
      self,
      skills: list[Skill],
      context: ReadonlyContext,
      active_taxonomies: list[str],
  ) -> list[Skill]:
    """Prioritizes, reorders, or accentuates skills under the active taxonomy.

    Allows the policy to sort key tools to the top of the available_skills XML list
    presented in the prompt, encouraging the LLM to select preferred actions.
    """
    return skills

  def shape_skill(
      self,
      skill: Skill,
      context: ReadonlyContext,
      shaped_description: Optional[str],
  ) -> Skill:
    """Prepares and shapes a skill representation for presentation to the agent.

    Defaults to a secure manual reconstruction to prevent accidental leakage of
    internal developer/business flags to LLM prompts, but can be overridden by
    custom policies to use `model_copy()` or other strategies.
    """
    assert skill is not None, "Skill instance cannot be None"
    
    from google.adk.skills.models import Skill, Frontmatter
    extra = getattr(skill.frontmatter, "model_extra", None) or {}
    return Skill(
        frontmatter=Frontmatter(
            name=skill.frontmatter.name,
            description=shaped_description,
            **extra
        ),
        instructions=skill.instructions
    )


def _get_taxonomy_binds(skill: Skill) -> list[str]:
  """Dynamically extracts taxonomy binds, supporting both modified and unmodified core SDKs.
  
  This utility functions as a robust protocol layer. If the SDK natively supports
  frontmatter taxonomy binds, it reads them directly. Otherwise, it falls back to parsing
  Pydantic extra fields (since core SDK uses `extra="allow"`), handling variations in 
  hyphenation/naming conventions.
  """
  # Direct attribute access check
  if hasattr(skill.frontmatter, "taxonomy_binds"):
    return skill.frontmatter.taxonomy_binds
  
  # Fallback: Read from Pydantic's model_extra dictionary (natively populated because of extra="allow")
  extra = getattr(skill.frontmatter, "model_extra", None) or {}
  binds = extra.get("taxonomy-binds") or extra.get("taxonomy_binds") or []
  if isinstance(binds, str):
    return [binds]
  return list(binds)


def _interpolate_variables(text: str, active_taxonomies: list[str], registry: Optional[Any]) -> str:
  if not text or not registry:
    return text

  import re
  pattern = r"\{taxonomy:([a-zA-Z0-9_-]+)\}"

  def replace(match):
    var_name = match.group(1)
    for tax_id in active_taxonomies:
      term = registry.get_term(tax_id)
      if term:
        variables = getattr(term, "variables", {})
        if not variables and hasattr(term, "model_extra"):
          variables = (term.model_extra or {}).get("variables", {})
        if variables and var_name in variables:
          return str(variables[var_name])
    
    logger.warning("Taxonomy variable %r not found under active taxonomies: %s", var_name, active_taxonomies)
    return ""

  return re.sub(pattern, replace, text)


class DefaultSkillPolicy(SkillPolicy):
  """Default skill policy using taxonomy-bind set-intersection matching.
  
  If a skill has no taxonomy binds defined, it is treated as unrestricted/allowed by default.
  If it has binds, at least one bind must intersect with the active taxonomy set.
  """

  def __init__(self, registry: Optional[Any] = None):
    self.registry = registry

  def is_skill_allowed(
      self,
      skill: Skill,
      context: ReadonlyContext,
      active_taxonomies: list[str],
  ) -> bool:
    binds = _get_taxonomy_binds(skill)
    # Unrestricted skills are always allowed
    if not binds:
      return True
    # Require at least one matching taxonomy between active set and skill binds
    return bool(set(binds) & set(active_taxonomies))

  def shape_instructions(
      self,
      skill: Skill,
      context: ReadonlyContext,
      original_instructions: str,
  ) -> str:
    active_taxonomies = context.state.get("_active_taxonomies") or []
    return _interpolate_variables(original_instructions, active_taxonomies, self.registry)

  def shape_description(
      self,
      skill: Skill,
      context: ReadonlyContext,
      original_description: str,
  ) -> str:
    active_taxonomies = context.state.get("_active_taxonomies") or []
    return _interpolate_variables(original_description, active_taxonomies, self.registry)

  def shape_system_instruction(
      self,
      context: ReadonlyContext,
      active_taxonomies: list[str],
      original_instructions: str,
  ) -> str:
    return _interpolate_variables(original_instructions, active_taxonomies, self.registry)

  def prioritize_skills(
      self,
      skills: list[Skill],
      context: ReadonlyContext,
      active_taxonomies: list[str],
  ) -> list[Skill]:
    # No-op pass-through for default behavior
    return skills
