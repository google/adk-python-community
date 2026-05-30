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

"""Unit tests for the Pluggable Policy & Taxonomy Security Engine in Community.

This test suite covers taxonomy classification data loading formats, resolver aggregation,
access-control authorization filtering, path validation/traversal prevention, and
cognitive steering/behavioral shaping mechanisms.
"""

from unittest import mock
import pytest

from google.adk_community.plugins.taxonomy import DefaultSkillPolicy
from google.adk_community.plugins.taxonomy import SkillPolicy
from google.adk_community.plugins.taxonomy import TaxonomyPipeline
from google.adk_community.plugins.taxonomy import TaxonomyPlugin
from google.adk_community.plugins.taxonomy import TaxonomyRegistry
from google.adk_community.plugins.taxonomy import TaxonomyResolver
from google.adk_community.plugins.taxonomy import TaxonomyTerm
from google.adk_community.plugins.taxonomy.policy import _get_taxonomy_binds
from google.adk.skills.models import Frontmatter
from google.adk.skills.models import Skill


def test_taxonomy_term():
  """Tests TaxonomyTerm model instantiation and defaults.
  
  Ensures taxonomy term instances hold core metadata and instantiate with standard
  defaults (like empty alternate labels and no parents).
  """
  term = TaxonomyTerm(id="tech", name="Technology", definition="Tech domain")
  assert term.id == "tech"
  assert term.name == "Technology"
  assert term.definition == "Tech domain"
  assert term.parent_id is None
  assert term.alt_labels == []


def test_registry_flat_json():
  """Tests parsing flat JSON structure into TaxonomyRegistry.
  
  Verifies that a plain list of objects defining IDs and parent IDs are correctly
  loaded and indexed into hierarchical parent-child relationships.
  """
  data = [
      {
          "id": "eng",
          "parentId": None,
          "name": "Engineering",
          "definition": "Eng dept",
      },
      {
          "id": "ml",
          "parentId": "eng",
          "name": "Machine Learning",
          "definition": "ML team",
      },
  ]
  registry = TaxonomyRegistry.from_flat_json(data)
  assert len(registry.list_ids()) == 2
  assert "eng" in registry.list_ids()
  assert "ml" in registry.list_ids()

  term_eng = registry.get_term("eng")
  term_ml = registry.get_term("ml")
  assert term_eng.name == "Engineering"
  assert term_ml.parent_id == "eng"

  children = registry.get_children("eng")
  assert len(children) == 1
  assert children[0].id == "ml"


def test_registry_json_ld():
  """Tests parsing JSON-LD SKOS structure into TaxonomyRegistry.
  
  Validates SKOS standard structure imports, including URI mapping, prefLabel
  mapping, altLabel array conversions, and broader relation parsing.
  """
  data = [
      {
          "@context": "http://w3.org",
          "@type": "Concept",
          "@id": "https://example.com/eng",
          "prefLabel": {"@value": "Engineering", "@language": "en"},
          "definition": {"@value": "Eng dept", "@language": "en"},
      },
      {
          "@context": "http://w3.org",
          "@type": "Concept",
          "@id": "https://example.com/ml",
          "prefLabel": {"@value": "Machine Learning", "@language": "en"},
          "altLabel": [{"@value": "ML", "@language": "en"}],
          "definition": {"@value": "ML team", "@language": "en"},
          "broader": "https://example.com/eng",
      },
  ]
  registry = TaxonomyRegistry.from_json_ld(data)
  assert len(registry.list_ids()) == 2

  term_eng = registry.get_term("https://example.com/eng")
  term_ml = registry.get_term("https://example.com/ml")
  assert term_eng.name == "Engineering"
  assert term_ml.parent_id == "https://example.com/eng"
  assert term_ml.alt_labels == ["ML"]


@pytest.mark.asyncio
async def test_taxonomy_pipeline():
  """Tests pipeline resolution chaining multiple resolvers.
  
  Ensures that the composite pipeline runs each individual resolver and merges
  their outputs into a unique, aggregated active taxonomy list.
  """

  class SimpleResolver(TaxonomyResolver):

    def __init__(self, resolved_domains: list[str]):
      self.resolved_domains = resolved_domains

    async def resolve_taxonomies(self, context, llm_request) -> list[str]:
      return self.resolved_domains

  context = mock.MagicMock()
  llm_request = mock.MagicMock()

  pipeline = TaxonomyPipeline([SimpleResolver(["eng"]), SimpleResolver(["finance"])])
  resolved = await pipeline.resolve_taxonomies(context, llm_request)
  assert sorted(resolved) == ["eng", "finance"]


def test_default_skill_policy():
  """Tests DefaultSkillPolicy filter mechanism.
  
  Checks that the default intersection policy correctly authorizes matching skills,
  blocks skills with non-overlapping binds, and always allows unrestricted skills.
  """
  policy = DefaultSkillPolicy()

  skill_eng = Skill(
      frontmatter=Frontmatter(
          name="eng-skill",
          description="Desc",
          taxonomy_binds=["eng"],
      ),
      instructions="body",
  )
  skill_finance = Skill(
      frontmatter=Frontmatter(
          name="finance-skill",
          description="Desc",
          taxonomy_binds=["finance"],
      ),
      instructions="body",
  )

  context = mock.MagicMock()
  assert policy.is_skill_allowed(skill_eng, context, ["eng"]) is True
  assert policy.is_skill_allowed(skill_finance, context, ["eng"]) is False
  assert policy.is_skill_allowed(skill_finance, context, ["eng", "finance"]) is True

  skill_unrestricted = Skill(
      frontmatter=Frontmatter(name="any-skill", description="Desc"),
      instructions="body",
  )
  assert policy.is_skill_allowed(skill_unrestricted, context, ["marketing"]) is True

  assert policy.shape_instructions(skill_eng, context, "original") == "original"


@pytest.mark.asyncio
async def test_taxonomy_plugin_list_skills():
  """Tests TaxonomyPlugin intercepts and filters skill lists correctly.
  
  Verifies that list_skills tool calls are intercepted in before_tool_callback
  and that the return payload contains only the policy-allowed skills in valid XML format.
  """

  class RestrictedPolicy(SkillPolicy):

    def is_skill_allowed(self, skill: Skill, context, active_taxonomies: list[str]) -> bool:
      binds = _get_taxonomy_binds(skill)
      return "eng" in binds

    def shape_instructions(self, skill: Skill, context, original_instructions: str) -> str:
      return original_instructions

  mock_resolver = mock.MagicMock()
  plugin = TaxonomyPlugin(policy=RestrictedPolicy(), resolver=mock_resolver)

  skills = {
      "skill-1": Skill(
          frontmatter=Frontmatter(
              name="skill-1",
              description="Desc",
              taxonomy_binds=["eng"],
          ),
          instructions="body",
      ),
      "skill-2": Skill(
          frontmatter=Frontmatter(
              name="skill-2",
              description="Desc",
              taxonomy_binds=["finance"],
          ),
          instructions="body",
      ),
  }

  context = mock.MagicMock()
  context.state = {"_active_taxonomies": ["eng"]}

  mock_tool = mock.MagicMock()
  mock_tool.name = "list_skills"
  mock_tool._toolset._list_skills.return_value = list(skills.values())

  # Patch XML formatter to focus purely on verifying taxonomy filtration behavior
  with mock.patch("google.adk_community.plugins.taxonomy.taxonomy_plugin.prompt.format_skills_as_xml") as mock_format:
    mock_format.return_value = "<skills><skill name=\"skill-1\"/></skills>"

    result = await plugin.before_tool_callback(
        tool=mock_tool,
        tool_args={},
        tool_context=context,
    )

    assert isinstance(result, dict)
    assert "result" in result
    assert "skill-1" in result["result"]
    assert "skill-2" not in result["result"]


@pytest.mark.asyncio
async def test_taxonomy_steering_capabilities():
  """Tests prioritizing/sorting skills and injecting global system prompts.
  
  Verifies cognitive steering hooks:
  1. System Instruction Shaping (injecting dynamic instructions into LLM system prompts).
  2. Skill Prioritization (reordering skills in list_skills results).
  """

  class SteeringPolicy(SkillPolicy):

    def is_skill_allowed(self, skill: Skill, context, active_taxonomies: list[str]) -> bool:
      return True

    def shape_instructions(self, skill: Skill, context, original_instructions: str) -> str:
      return original_instructions

    def shape_system_instruction(self, context, active_taxonomies: list[str], original_instructions: str) -> str:
      if "strict" in active_taxonomies:
        return original_instructions + " - MANDATED COMPLIANCE TURN"
      return original_instructions

    def prioritize_skills(self, skills: list[Skill], context, active_taxonomies: list[str]) -> list[Skill]:
      if "strict" in active_taxonomies:
        return sorted(skills, key=lambda s: 0 if s.frontmatter.name == "important" else 1)
      return skills

  class MockResolver(TaxonomyResolver):
    async def resolve_taxonomies(self, context, llm_request) -> list[str]:
      return ["strict"]

  plugin = TaxonomyPlugin(policy=SteeringPolicy(), resolver=MockResolver())

  # 1. Verify before_model_callback system instruction injection
  context = mock.MagicMock()
  context.state = {}
  llm_request = mock.MagicMock()
  llm_request.config = mock.MagicMock()
  llm_request.config.system_instruction = "Original Prompt"

  await plugin.before_model_callback(callback_context=context, llm_request=llm_request)
  assert context.state["_active_taxonomies"] == ["strict"]
  assert llm_request.config.system_instruction == "Original Prompt - MANDATED COMPLIANCE TURN"

  # 2. Verify skill prioritization/sorting in list_skills
  skills = [
      Skill(frontmatter=Frontmatter(name="normal", description="Desc"), instructions="body"),
      Skill(frontmatter=Frontmatter(name="important", description="Desc"), instructions="body"),
  ]

  mock_tool = mock.MagicMock()
  mock_tool.name = "list_skills"
  mock_tool._toolset._list_skills.return_value = skills

  with mock.patch("google.adk_community.plugins.taxonomy.taxonomy_plugin.prompt.format_skills_as_xml") as mock_format:
    await plugin.before_tool_callback(
        tool=mock_tool,
        tool_args={},
        tool_context=context,
    )
    # Check that format_skills_as_xml was called with "important" sorted first
    called_skills = mock_format.call_args[0][0]
    assert called_skills[0].frontmatter.name == "important"
    assert called_skills[1].frontmatter.name == "normal"
