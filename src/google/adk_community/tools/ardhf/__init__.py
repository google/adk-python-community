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

"""ARDHF — HuggingFace Discover (ARD) toolset for ADK.

Wraps HuggingFace Discover
(https://github.com/huggingface/hf-discover) as an ADK BaseToolset,
giving agents the ability to discover agents, skills, MCP servers, and
other agentic resources at runtime.

Usage::

    from google.adk_community.tools.ardhf import AgentFinderToolset
    from google.adk import Agent

    agent = Agent(
        name="discovery_agent",
        instruction="Search for tools when you need a capability.",
        tools=[AgentFinderToolset()],
    )

Prepared for rename to ARD (Agentic Resource Discovery).
"""

from google.adk_community.tools.ardhf.ardhf_toolset import (
    AgentFinderToolset,
)

__all__ = [
    "AgentFinderToolset",
]
