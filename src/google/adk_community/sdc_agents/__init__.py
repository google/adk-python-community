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

"""SDC Agents -- Purpose-scoped semantic data governance toolsets for ADK.

Eight BaseToolset implementations (32 tools) that transform SQL, CSV, JSON,
and MongoDB data into validated, self-describing SDC4 artifacts with
structured audit trails and enforced agent isolation boundaries.

Install: pip install google-adk-community[sdc-agents]
Docs: https://github.com/SemanticDataCharter/SDC_Agents

Requires sdc-agents >= 4.3.3.
"""

from sdc_agents.common.config import load_config
from sdc_agents.common.config import SDCAgentsConfig
from sdc_agents.toolsets.assembly import AssemblyToolset
from sdc_agents.toolsets.catalog import CatalogToolset
from sdc_agents.toolsets.distribution import DistributionToolset
from sdc_agents.toolsets.generator import GeneratorToolset
from sdc_agents.toolsets.introspect import IntrospectToolset
from sdc_agents.toolsets.knowledge import KnowledgeToolset
from sdc_agents.toolsets.mapping import MappingToolset
from sdc_agents.toolsets.validation import ValidationToolset

__all__ = [
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
]
