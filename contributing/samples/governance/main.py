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

"""Example: multi-agent workflow with governance plugin.

This example shows how to register VeronicaGovernancePlugin on an ADK
Runner to enforce per-agent budgets, block tools, and degrade to a
cheaper model when budget runs low.

Usage:
  export GOOGLE_API_KEY="your-key"
  python main.py
"""

import asyncio
import logging

from google.adk import Runner
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.genai import types

from google.adk_community.governance import GovernanceConfig
from google.adk_community.governance import VeronicaGovernancePlugin

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
  # Configure governance limits
  config = GovernanceConfig(
      max_cost_usd=0.50,  # org-level: 50 cents
      agent_max_cost_usd=0.25,  # per-agent: 25 cents
      failure_threshold=3,  # circuit breaker after 3 failures
      recovery_timeout_s=30.0,
      degradation_threshold=0.7,  # degrade at 70% budget
      fallback_model="gemini-2.0-flash-lite",
      blocked_tools=["shell_exec"],
      disable_tools_on_degrade=["web_search"],
  )

  plugin = VeronicaGovernancePlugin(config=config)

  # Define agents
  researcher = Agent(
      model="gemini-2.5-flash",
      name="researcher",
      instruction=(
          "You are a research assistant. Answer questions using your"
          " knowledge. Be concise."
      ),
  )

  summarizer = Agent(
      model="gemini-2.5-flash",
      name="summarizer",
      instruction=(
          "You summarize text provided to you. Keep summaries to 2-3 sentences."
      ),
  )

  # Orchestrator delegates to sub-agents
  orchestrator = Agent(
      model="gemini-2.5-flash",
      name="orchestrator",
      instruction=(
          "You coordinate research tasks. Use the researcher agent to"
          " find information, then the summarizer to condense it."
      ),
      sub_agents=[researcher, summarizer],
  )

  # Create runner with governance plugin
  session_service = InMemorySessionService()
  runner = Runner(
      agent=orchestrator,
      app_name="governance_demo",
      session_service=session_service,
      plugins=[plugin],
  )

  async def run():
    session = await session_service.create_session(
        app_name="governance_demo",
        user_id="demo_user",
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part(text="What is agent governance?")],
    )

    async for event in runner.run_async(
        session_id=session.id,
        user_id="demo_user",
        new_message=user_message,
    ):
      if event.content and event.content.parts:
        for part in event.content.parts:
          if part.text:
            print(f"[{event.author}] {part.text[:200]}")

    # After run, the plugin logs a governance summary automatically.
    # You can also inspect programmatically:
    snap = plugin.budget.snapshot()
    print(f"\nTotal spent: ${snap.org_spent_usd:.4f}")
    for agent, spent in snap.agent_spent.items():
      print(f"  {agent}: ${spent:.4f}")

  asyncio.run(run())


if __name__ == "__main__":
  main()
