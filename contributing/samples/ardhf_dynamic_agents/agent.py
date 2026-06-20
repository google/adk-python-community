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

"""Orchestrator that discovers and dynamically uses A2A agents, MCP tools, and Skills.

This sample demonstrates the full discover-and-use pattern for all three
types of agentic resources available through ARD (Agentic Resource Discovery):

  **A2A Agents** — Remote agents that speak the Agent-to-Agent protocol.
      The orchestrator uses ``connect_agent`` to send a message and get
      a response via A2A JSON-RPC, enabling cross-framework delegation.

  **MCP Servers** — Remote tool servers that expose tools via the Model
      Context Protocol.  The orchestrator can connect to an MCP server
      and make its tools available for function calling.

  **Skills** — Markdown instructions (SKILL.md) that describe how to
      accomplish a task.  The orchestrator fetches the skill content
      and follows the instructions directly.

The flow:
  1. User asks the orchestrator something it can't do alone.
  2. Orchestrator searches ARD registries for capable resources.
  3. Based on the resource type found:
     - A2A agent -> delegate via ``connect_agent``
     - MCP server -> connect and use its tools
     - Skill -> fetch SKILL.md and follow its instructions
  4. Orchestrator returns the result to the user.

Usage::

    adk web contributing/samples/ardhf_dynamic_agents

Or with the challenge server::

    hf-discover challenge serve --port 8090 &
    ARDHF_REGISTRY_URL=http://127.0.0.1:8090 \
        adk web contributing/samples/ardhf_dynamic_agents

With a known A2A agent::

    python -m contributing.samples.ardhf_dynamic_agents.agent
"""

from __future__ import annotations

import asyncio
import os

from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

from google.adk_community.tools.ardhf import AgentFinderToolset

# -- Configuration ---------------------------------------------------------

_registry_url = os.environ.get(
    "ARDHF_REGISTRY_URL",
    "https://huggingface-hf-discover.hf.space",
)
_token = os.environ.get("HF_TOKEN")

# -- Agent -----------------------------------------------------------------

agent_finder_toolset = AgentFinderToolset(
    registry_url=_registry_url,
    token=_token,
)

root_agent = Agent(
    name="dynamic_orchestrator",
    description=(
        "An orchestrator agent that dynamically discovers and "
        "uses remote A2A agents, MCP tools, and Skills at runtime."
    ),
    instruction=(
        "You are a smart orchestrator.  You do not have built-in "
        "domain expertise -- instead, you dynamically find and "
        "use specialised remote resources.\n\n"
        "## Resource Types\n\n"
        "ARD registries contain three types of resources you can "
        "discover and use:\n\n"
        "### 1. A2A Agents (application/a2a-agent-card+json)\n"
        "Remote agents that accept messages via the A2A protocol.  "
        "Use `search_agents` to find them and `connect_agent` to "
        "send a message and get a response.  This is the best "
        "option when you need an agent with domain expertise to "
        "handle a task end-to-end.\n\n"
        "**Example flow:**\n"
        "1. `search_agents('messaging support')` -> finds agents\n"
        "2. `get_agent_card(url)` -> inspect capabilities\n"
        "3. `connect_agent(url, 'How do I register?')` -> get answer\n\n"
        "### 2. MCP Servers (application/mcp-server-card+json)\n"
        "Remote tool servers that expose callable tools.  Use "
        "`search_tools` to find them and `get_agent_card` to fetch "
        "the server descriptor with its endpoint URL.  Then tell "
        "the user the available MCP endpoint so they can connect "
        "their agent to it.  MCP servers provide specific tools "
        "(like code analysis, database queries) rather than full "
        "agent capabilities.\n\n"
        "**Example flow:**\n"
        "1. `search_tools('code complexity analysis')` -> finds servers\n"
        "2. `get_agent_card(url)` -> get server details and endpoint\n"
        "3. Report the MCP endpoint URL and available tools to the user\n\n"
        "### 3. Skills (application/ai-skill)\n"
        "Markdown instructions (SKILL.md) that describe how to "
        "accomplish a specific task.  Use `search_skills` to find "
        "them and `get_agent_card` to fetch the skill content.  "
        "Then follow the instructions in the skill markdown to "
        "help the user.  Skills are like recipes -- they tell you "
        "what to do step by step.\n\n"
        "**Example flow:**\n"
        "1. `search_skills('deploy to huggingface')` -> finds skills\n"
        "2. `get_agent_card(url)` -> fetch SKILL.md content\n"
        "3. Follow the instructions to help the user\n\n"
        "## Known Resources\n\n"
        "- **AgentMsg Support Agent** (A2A): An agent that answers "
        "questions about AgentMsg, a store-and-forward message relay "
        "for AI agents.  Agent card URL: "
        "https://agentmsg-support-462816930018.us-central1.run.app"
        "/.well-known/agent-card.json\n\n"
        "## Guidelines\n\n"
        "- Always search before delegating -- don't assume you know "
        "which resource to use.\n"
        "- Use `search_ards` for a broad search across all types, "
        "or use the type-specific search tools for targeted results.\n"
        "- If the user asks about a known resource (like AgentMsg), "
        "you can use `connect_agent` directly with its URL.\n"
        "- If search returns no suitable resources, tell the user "
        "what you searched for and that no matching resources were "
        "found.\n"
        "- If connect_agent fails, report the error and suggest "
        "alternatives from the search results.\n"
        "- Be transparent about delegation -- tell the user you are "
        "routing their request to a specialised resource."
    ),
    tools=[agent_finder_toolset],
)


# -- Main ------------------------------------------------------------------


async def main() -> None:
  """Run the orchestrator with a sample A2A query."""
  runner = InMemoryRunner(
      agent=root_agent,
      app_name="ardhf_dynamic_demo",
  )
  session = await runner.session_service.create_session(
      user_id="demo_user",
      app_name="ardhf_dynamic_demo",
  )

  # Demo: Ask the AgentMsg Support agent a question via A2A.
  prompt = (
      "Ask the AgentMsg Support agent how to register an agent "
      "with AgentMsg.  Its agent card is at: "
      "https://agentmsg-support-462816930018.us-central1.run.app"
      "/.well-known/agent-card.json"
  )
  print(f"User: {prompt}")

  async for event in runner.run_async(
      user_id="demo_user",
      session_id=session.id,
      new_message=types.Content(
          role="user",
          parts=[types.Part.from_text(text=prompt)],
      ),
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        text = getattr(part, "text", None)
        if text:
          print(f"{event.author}: {text}")


if __name__ == "__main__":
  asyncio.run(main())
