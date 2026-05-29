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

"""Pydantic models for taxonomy configuration parsing."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class TaxonomyTerm(BaseModel):
  """A single taxonomy term with metadata for validation and LLM disambiguation.
  Attributes:
      id: (str)
      parent_id: (Optional[str])
      name: (str)
      definition: (Optional[str])
      alt_labels: (list[str])
  """

  model_config = ConfigDict(populate_by_name=True)

  id: str
  parent_id: Optional[str] = Field(None, alias="parentId")
  name: str
  definition: Optional[str] = None
  alt_labels: list[str] = Field(default_factory=list, alias="altLabels")


class TaxonomyRegistry(BaseModel):
  """Central registry for taxonomy term definitions.

  Supported JSON Schemas:

    **Flat Key-Value JSON** (``from_flat_json``):
        id: str
        parentId: Optional[str]
        name: str
        definition: Optional[str]

    **JSON-LD with SKOS** (``from_json_ld``):
        @context: str
        @type: str
        @id: str
        prefLabel: dict (``{"@value": str, "@language": str}``)
        altLabel: list[dict] (``[{"@value": str, "@language": str}]``)
        definition: dict (``{"@value": str, "@language": str}``)
        broader: Optional[str]
  """

  terms: dict[str, TaxonomyTerm] = {}

  @classmethod
  def from_flat_json(cls, data: list[dict]) -> TaxonomyRegistry:
    """Parse taxonomy terms from flat key-value JSON format."""
    terms = {}
    for item in data:
      term = TaxonomyTerm.model_validate(item)
      terms[term.id] = term
    return cls(terms=terms)

  @classmethod
  def from_json_ld(cls, data: list[dict]) -> TaxonomyRegistry:
    """Parse JSON-LD SKOS format into TaxonomyRegistry."""
    terms = {}
    for item in data:
      term_id = item.get("@id")
      if not term_id:
        continue

      pref_label = item.get("prefLabel", {})
      if isinstance(pref_label, dict):
        pref_label = pref_label.get("@value", "")

      definition_raw = item.get("definition", {})
      if isinstance(definition_raw, dict):
        definition = definition_raw.get("@value") or None
      elif isinstance(definition_raw, str):
        definition = definition_raw or None
      else:
        definition = None

      alt_labels_raw = item.get("altLabel", [])
      if not isinstance(alt_labels_raw, list):
        alt_labels_raw = [alt_labels_raw]
      alt_labels = [
          label.get("@value")
          for label in alt_labels_raw
          if isinstance(label, dict) and label.get("@value")
      ]

      broader = item.get("broader")
      term = TaxonomyTerm(
          id=term_id,
          parent_id=broader,
          name=pref_label,
          definition=definition,
          alt_labels=alt_labels,
      )
      terms[term_id] = term
    return cls(terms=terms)

  def get_term(self, term_id: str) -> Optional[TaxonomyTerm]:
    """Lookup a term by its ID."""
    return self.terms.get(term_id)

  def get_children(self, parent_id: str) -> list[TaxonomyTerm]:
    """Get all direct children of a term."""
    return [t for t in self.terms.values() if t.parent_id == parent_id]

  def list_ids(self) -> list[str]:
    """List all term IDs in the registry."""
    return list(self.terms.keys())
