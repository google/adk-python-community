# Copyright 2026 Microsoft Corporation
#
# Licensed under the MIT License.
"""
Example: Google ADK agent with Agent Governance Toolkit policy enforcement.

Demonstrates:
1. Loading YAML governance policies
2. Evaluating policies before tool calls
3. Producing tamper-evident audit trails
"""

from pathlib import Path

from google.adk.agents import Agent
from governance_plugin import GovernancePlugin


def create_governed_agent() -> Agent:
    """Create an ADK agent with governance controls."""
    governance = GovernancePlugin(
        policy_dir=Path(__file__).parent / "policies",
        agent_did="did:mesh:adk-demo-agent",
    )

    # Example: check policy before a tool call
    result = governance.before_tool_call(
        tool_name="web_search",
        args={"query": "latest AI safety research"},
    )
    print(f"Policy decision: {result['decision']} — {result['reason']}")

    # Check audit trail
    summary = governance.get_audit_summary()
    print(f"Audit: {summary['total_entries']} entries, chain valid: {summary['chain_valid']}")

    return Agent(
        name="governed-research-agent",
        model="gemini-2.0-flash",
        instruction="You are a research assistant with governance controls.",
    )


if __name__ == "__main__":
    agent = create_governed_agent()
    print(f"Agent '{agent.name}' created with governance enabled.")