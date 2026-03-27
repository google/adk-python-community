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

"""SDC Agents demo -- full data governance pipeline.

Composes five SDC Agents toolsets into a single LlmAgent that can
introspect datasources, discover matching catalog components, map
columns to schemas, and assemble validated data models.

Prerequisites:
    pip install google-adk-community[sdc-agents]
    export SDC_API_KEY="your-sdcstudio-api-key"

Usage:
    adk run .
"""

from google.adk.agents import LlmAgent

from google.adk_community.sdc_agents import (
    CatalogToolset,
    IntrospectToolset,
    MappingToolset,
    AssemblyToolset,
    ValidationToolset,
    SDCAgentsConfig,
)

config = SDCAgentsConfig(
    sdcstudio={
        "base_url": "https://sdcstudio.com",
        "api_key": "${SDC_API_KEY}",
    },
    datasources={
        "sample": {
            "type": "csv",
            "path": "./data/sample.csv",
        },
    },
    cache={"root": ".sdc-cache"},
    audit={"path": ".sdc-cache/audit.jsonl"},
)

root_agent = LlmAgent(
    name="sdc_demo_agent",
    model="gemini-2.0-flash",
    description=(
        "Full data governance pipeline: introspect, discover, map,"
        " and assemble SDC4 data models."
    ),
    instruction=(
        "You help data engineers govern their data. Follow this workflow:\n"
        "1. Introspect the datasource to discover columns and types\n"
        "2. Search the SDC4 catalog for matching published schemas\n"
        "3. Discover catalog components that match the datasource structure\n"
        "4. Map unmatched columns to schema components by similarity\n"
        "5. Propose a cluster hierarchy for the data model\n"
        "6. Assemble the final data model via the Assembly API\n"
        "7. Validate the generated artifacts"
    ),
    tools=[
        IntrospectToolset(config=config),
        CatalogToolset(config=config),
        MappingToolset(config=config),
        AssemblyToolset(config=config),
        ValidationToolset(config=config),
    ],
)
