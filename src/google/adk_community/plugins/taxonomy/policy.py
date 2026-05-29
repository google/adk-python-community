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
  """Abstract base class for taxonomy resolution."""

  @abstractmethod
  async def resolve_taxonomies(
      self, context: ReadonlyContext, llm_request: LlmRequest
  ) -> list[str]:
    """Resolves active taxonomy domain URIs from context and LLM history."""
    pass


class TaxonomyPipeline(TaxonomyResolver):
  """Executes a sequence of taxonomy resolvers in order (multi-step pipeline)."""

  def __init__(self, resolvers: list[TaxonomyResolver]):
    self.resolvers = resolvers

  async def resolve_taxonomies(
      self, context: ReadonlyContext, llm_request: LlmRequest
  ) -> list[str]:
    active_domains: set[str] = set()
    for resolver in self.resolvers:
      domains = await resolver.resolve_taxonomies(context, llm_request)
      if domains:
        active_domains.update(domains)
    return list(active_domains)


class SkillPolicy(ABC):
  """Abstract policy engine determining skill execution permissions and instruction shaping."""

  @abstractmethod
  def is_skill_allowed(
      self,
      skill: Skill,
      context: ReadonlyContext,
      active_taxonomies: list[str],
  ) -> bool:
    """Determines if a skill can be loaded/used under the active taxonomies and context."""
    pass

  @abstractmethod
  def shape_instructions(
      self,
      skill: Skill,
      context: ReadonlyContext,
      original_instructions: str,
  ) -> str:
    """Applies dynamic instruction shaping/guardrails to a skill's instructions."""
    pass


def _get_taxonomy_binds(skill: Skill) -> list[str]:
  """Dynamically extracts taxonomy binds, supporting both modified and unmodified core SDKs."""
  if hasattr(skill.frontmatter, "taxonomy_binds"):
    return skill.frontmatter.taxonomy_binds
  
  # Fallback: Read from Pydantic's model_extra dictionary (natively populated because of extra="allow")
  extra = getattr(skill.frontmatter, "model_extra", None) or {}
  binds = extra.get("taxonomy-binds") or extra.get("taxonomy_binds") or []
  if isinstance(binds, str):
    return [binds]
  return list(binds)


class DefaultSkillPolicy(SkillPolicy):
  """Default skill policy using taxonomy-bind set-intersection matching."""

  def is_skill_allowed(
      self,
      skill: Skill,
      context: ReadonlyContext,
      active_taxonomies: list[str],
  ) -> bool:
    binds = _get_taxonomy_binds(skill)
    if not binds:
      return True
    return bool(set(binds) & set(active_taxonomies))

  def shape_instructions(
      self,
      skill: Skill,
      context: ReadonlyContext,
      original_instructions: str,
  ) -> str:
    return original_instructions
