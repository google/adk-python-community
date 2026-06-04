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

"""GoogleManagedMcpToolset for simplified Google MCP server connections.

Provides a McpToolset subclass that automates ADC-based authentication
and endpoint URL resolution for Google Managed MCP servers (e.g.,
BigQuery).

Usage:
    from google.adk_community.tools.google_managed_mcp import (
        GoogleManagedMcpToolset,
    )
    from google.adk.agents import Agent

    toolset = GoogleManagedMcpToolset(product="bigquery")

    agent = Agent(
        name="bq_agent",
        model="gemini-2.5-flash",
        instruction="Help users query BigQuery.",
        tools=[toolset],
    )
"""

from google.adk_community.tools.google_managed_mcp.google_managed_mcp_toolset import (
    GoogleManagedMcpToolset,
)

__all__ = [
    "GoogleManagedMcpToolset",
]
