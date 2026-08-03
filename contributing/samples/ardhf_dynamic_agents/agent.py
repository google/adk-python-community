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

"""Orchestrator agent that dynamically discovers and delegates to A2A agents.

This sample demonstrates the dream UX for agentic resource discovery:
an agent that receives tasks it cannot handle itself, dynamically
searches for a capable remote A2A agent, and delegates the work via the
A2A protocol — all within a single conversation.

The flow:
  1. User asks the orchestrator something it can't do alone.
  2. Orchestrator uses ``search_agents`` to find capable remote agents.
  3. Orchestrator uses ``get_agent_card`` to inspect the best match.
  4. Orchestrator uses ``connect_agent`` to delegate the task via A2A.
  5. Orchestrator returns the result to the user.

Usage::

    adk web contributing/samples/ardhf_dynamic_agents

Or with the challenge server::

    hf-agentfinder challenge serve --port 8090 &
    ARDHF_REGISTRY_URL=http://127.0.0.1:8090 \
        adk web contributing/samples/ardhf_dynamic_agents
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
    "https://huggingface.co/api/agentfinder",
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
        "delegates to remote A2A agents at runtime."
    ),
    instruction=(
        "You are a smart orchestrator.  You do not have built-in "
        "domain expertise — instead, you dynamically find and "
        "delegate to specialised remote agents.\n\n"
        "## How you work\n\n"
        "When a user asks you to do something:\n\n"
        "1. **Search** — Use search_agents to find A2A-compatible "
        "remote agents capable of handling the request.  For broader "
        "searches across all artifact types, use search_ards.\n\n"
        "2. **Evaluate** — Review the search results.  Pick the "
        "best match based on the agent's description, capabilities, "
        "and relevance to the user's request.\n\n"
        "3. **Inspect** — Use get_agent_card with the chosen "
        "agent's URL to fetch its full card and verify it can "
        "handle the task.\n\n"
        "4. **Delegate** — Use connect_agent with the agent card "
        "URL and a clear, well-formed message describing what the "
        "user needs.  Translate the user's request into a specific "
        "task for the remote agent.\n\n"
        "5. **Report** — Present the remote agent's response to the "
        "user, noting which agent handled the task.\n\n"
        "## Guidelines\n\n"
        "- Always search before delegating — don't assume you know "
        "which agent to use.\n"
        "- If search returns no suitable A2A agents, tell the user "
        "what you searched for and that no matching agents were "
        "found.\n"
        "- If connect_agent fails, report the error and suggest "
        "alternatives from the search results.\n"
        "- Be transparent about delegation — tell the user you are "
        "routing their request to a specialised agent."
    ),
    tools=[agent_finder_toolset],
)


# -- Main ------------------------------------------------------------------


async def main() -> None:
  """Run the orchestrator with a sample query."""
  runner = InMemoryRunner(
      agent=root_agent,
      app_name="ardhf_dynamic_demo",
  )
  session = await runner.session_service.create_session(
      user_id="demo_user",
      app_name="ardhf_dynamic_demo",
  )

  prompt = "I need to remove the background from an image"
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
