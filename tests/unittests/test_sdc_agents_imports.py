# Copyright 2025 Google LLC
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

"""Verify that all SDC Agents re-exports resolve correctly."""

import importlib

import pytest


@pytest.fixture(autouse=True)
def _skip_if_not_installed():
  """Skip every test in this module when sdc-agents is not installed."""
  pytest.importorskip("sdc_agents")


class TestSDCAgentsImports:
  """Validate that the community re-export module exposes all toolsets."""

  def test_module_importable(self):
    mod = importlib.import_module("google.adk_community.sdc_agents")
    assert mod is not None

  def test_load_config_exported(self):
    from google.adk_community.sdc_agents import load_config

    assert callable(load_config)

  def test_sdc_agents_config_exported(self):
    from google.adk_community.sdc_agents import SDCAgentsConfig

    assert SDCAgentsConfig is not None

  @pytest.mark.parametrize(
      "name",
      [
          "AssemblyToolset",
          "CatalogToolset",
          "DistributionToolset",
          "GeneratorToolset",
          "IntrospectToolset",
          "KnowledgeToolset",
          "MappingToolset",
          "ValidationToolset",
      ],
  )
  def test_toolset_exported(self, name: str):
    mod = importlib.import_module("google.adk_community.sdc_agents")
    cls = getattr(mod, name, None)
    assert cls is not None, f"{name} not found in sdc_agents module"

  def test_all_list_complete(self):
    from google.adk_community import sdc_agents

    expected = {
        "load_config",
        "SDCAgentsConfig",
        "AssemblyToolset",
        "CatalogToolset",
        "DistributionToolset",
        "GeneratorToolset",
        "IntrospectToolset",
        "KnowledgeToolset",
        "MappingToolset",
        "ValidationToolset",
    }
    assert set(sdc_agents.__all__) == expected
