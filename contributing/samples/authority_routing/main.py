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

"""Example: Google ADK agent with authority-posture routing.

Demonstrates:
1. Wiring AuthorityRoutingPlugin into an ADK Runner.
2. How the deterministic guard (irreversibility tripwire + approval-state
   check) alone, with no model, already blocks unauthorized tool calls.
3. How an optional posture router composes with the guard under
   most-restrictive-wins.

Run it directly — it needs no API key and no network:

    python contributing/samples/authority_routing/main.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from google.adk_community.plugins import AuthorityRequest
from google.adk_community.plugins import AuthorityRoutingPlugin
from google.adk_community.plugins import Posture
from google.adk_community.plugins import PostureVerdict


def demo_router(request: AuthorityRequest) -> PostureVerdict:
  """A tiny illustrative router (stands in for a model call).

  Real deployments pass the request to an LLM and parse a structured verdict.
  Here we key off markers the caller left in context so the demo is fully
  deterministic:

  * ``policy: denied`` -> STOP (the work is disallowed)
  * ``wants: advice``  -> ADVISE (recommendation only, no side effects)
  * otherwise          -> EXECUTE (authorized to act)
  """
  context = request.context.lower()
  if "policy: denied" in context:
    return PostureVerdict(
        posture=Posture.STOP, rationale="blocked by standing policy"
    )
  if "wants: advice" in context:
    return PostureVerdict(
        posture=Posture.ADVISE,
        rationale="user asked for a recommendation, not an action",
    )
  return PostureVerdict(
      posture=Posture.EXECUTE, scope=f"call {request.action} as requested"
  )


def create_governed_runner() -> Runner:
  """Create an ADK runner whose tool calls pass through authority routing."""
  plugin = AuthorityRoutingPlugin(posture_router=demo_router)

  agent = Agent(
      name="authority_routed_agent",
      model="gemini-2.0-flash",
      instruction="You are an assistant with an authorization layer.",
  )

  return Runner(
      agent=agent,
      plugins=[plugin],
      app_name="authority-routing-demo",
      session_service=InMemorySessionService(),
  )


# ---------------------------------------------------------------------------
# A live walkthrough of the decision layer (no model, no network).
# ---------------------------------------------------------------------------


@dataclass
class _Tool:
  name: str


class _Ctx:

  def __init__(self, context: str = ""):
    self.state = {"authority_context": context}


@dataclass
class _Case:
  label: str
  tool: str
  args: dict
  context: str


CASES = [
    _Case(
        "authorized read — router EXECUTE, guard clear",
        "web_search",
        {"query": "weather in Paris tomorrow"},
        context="wants: action",
    ),
    _Case(
        "advice requested — agent must NOT execute",
        "update_ticket",
        {"id": 42, "status": "closed"},
        context="wants: advice on whether to close ticket 42",
    ),
    _Case(
        "irreversible call — guard overrides EXECUTE",
        "delete_records",
        {"table": "customers", "where": "all"},
        context="wants: action",
    ),
    _Case(
        "approval pending — guard floors to DEFER",
        "send_email",
        {"to": "vip@corp.com", "body": "the numbers"},
        context="wants: action; note: awaiting approval from finance",
    ),
    _Case(
        "disallowed — STOP survives strictest-wins",
        "read_vault",
        {"path": "prod credential store"},
        context="wants: action; policy: denied for credential stores",
    ),
]


async def walkthrough() -> None:
  plugin = AuthorityRoutingPlugin(posture_router=demo_router)
  print(f"{'CASE':<46}{'TOOL':<17}{'POSTURE':<9}OUTCOME")
  print("-" * 96)
  for case in CASES:
    decision = await plugin.before_tool_callback(
        tool=_Tool(case.tool),
        tool_args=case.args,
        tool_context=_Ctx(case.context),
    )
    if decision is None:
      posture, outcome = "EXECUTE", "tool runs"
    else:
      posture = decision["posture"]
      outcome = f"blocked — {decision['reason']}"
    print(f"{case.label:<46}{case.tool:<17}{posture:<9}{outcome}")


if __name__ == "__main__":
  runner = create_governed_runner()
  print("Runner created with authority-routing plugin enabled.\n")
  asyncio.run(walkthrough())
