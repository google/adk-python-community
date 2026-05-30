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

from abc import ABC
from abc import abstractmethod

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.skills.models import Skill


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


class SkillPolicy(ABC):
  """Abstract policy engine determining skill execution permissions and instruction shaping.
  
  This class defines the interface for two main responsibilities:
  1. Access Control (Authorization): Blocking or permitting skills based on active taxonomies.
  2. Cognitive Steering (Behavioral Shaping): Altering skill instructions, descriptions,
     prioritization, and global system prompts to steer agent execution dynamically.
     
  Implements the Hook Method pattern, providing concrete default pass-throughs
  for steering while keeping authorization and core shaping abstract.
  """

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


class DefaultSkillPolicy(SkillPolicy):
  """Default skill policy using taxonomy-bind set-intersection matching.
  
  If a skill has no taxonomy binds defined, it is treated as unrestricted/allowed by default.
  If it has binds, at least one bind must intersect with the active taxonomy set.
  """

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
    # No-op pass-through for default behavior
    return original_instructions

  def shape_system_instruction(
      self,
      context: ReadonlyContext,
      active_taxonomies: list[str],
      original_instructions: str,
  ) -> str:
    # No-op pass-through for default behavior
    return original_instructions

  def prioritize_skills(
      self,
      skills: list[Skill],
      context: ReadonlyContext,
      active_taxonomies: list[str],
  ) -> list[Skill]:
    # No-op pass-through for default behavior
    return skills
