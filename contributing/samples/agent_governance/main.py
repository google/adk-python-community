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

"""Example: Google ADK agent with Agent Governance Toolkit policy enforcement.

Demonstrates:
1. Loading YAML governance policies
2. Evaluating policies before tool calls via the ADK plugin lifecycle
3. Producing tamper-evident audit trails
"""

from pathlib import Path

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk_community.plugins import AgentGovernancePlugin


def create_governed_runner() -> Runner:
    """Create an ADK runner with governance controls."""
    plugin = AgentGovernancePlugin(
        policy_dir=Path(__file__).parent / "policies",
        agent_did="did:mesh:adk-demo-agent",
    )

    agent = Agent(
        name="governed-research-agent",
        model="gemini-2.0-flash",
        instruction="You are a research assistant with governance controls.",
    )

    return Runner(agent=agent, plugins=[plugin], app_name="governed-demo")


if __name__ == "__main__":
    runner = create_governed_runner()
    print(f"Runner created with governance plugin enabled.")
