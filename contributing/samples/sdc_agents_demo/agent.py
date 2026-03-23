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

"""Minimal SDC Agents demo -- catalog search + data introspection.

Composes CatalogToolset and IntrospectToolset into a single LlmAgent
that can discover published SDC4 schemas and introspect datasource
structure.

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
        "Discovers SDC4 schemas and introspects datasource structure."
    ),
    instruction=(
        "You help data engineers govern their data.\n"
        "1. Use catalog tools to search for published SDC4 schemas\n"
        "2. Use introspect tools to analyze datasource columns and types\n"
        "3. Summarize findings clearly"
    ),
    tools=[
        CatalogToolset(config=config),
        IntrospectToolset(config=config),
    ],
)
