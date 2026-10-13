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

"""Example: a coordinator ADK agent delegates to a worker holding an
attenuated VAID, driven through a real ADK Runner with VaidIdentityPlugin
registered.

Offline: no model/API key. The two agents are plain BaseAgent subclasses
that act without an LLM, so this proves the plugin's enforcement through
the genuine ADK extension points (before_tool_callback via the real
plugin manager) rather than through a mock of them.

Run: python contributing/samples/vaid_identity/main.py
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as gtypes
from vaid_adk import ToolAuthorization, TrustedIssuers, VaidPresentation
from vaid_mint import (
    InMemoryAudit,
    MintService,
    ReferenceIssuer,
    VaidSeed,
    prefix_entry,
)
from vaid_mint.mint_types import MintPop, build_mint_pop_payload
from vaid_mint.revocation import InMemoryRevocationList
from vaid_pop import canonical_request_signing_bytes

from google.adk_community.plugins import VaidIdentityPlugin

APP_NAME = "vaid-identity-sample"
USER_ID = "demo-user"

# RFC 2606 reserved domain, named once because it has to agree in two
# places: the issuer stamps it into every document it mints, and the
# plugin's trust configuration binds the kernel key to it.
TRUST_DOMAIN = "vaid.example"


class _FakeTool:
  """Carries only what VaidIdentityPlugin.before_tool_callback reads: .name.
  Avoids needing a real FunctionTool + LLM round trip for this offline
  example."""

  def __init__(self, name: str) -> None:
    self.name = name


class _FakeToolContext:
  """Carries only what VaidIdentityPlugin.before_tool_callback reads:
  .agent_name, the identity of the agent making the call."""

  def __init__(self, agent_name: str) -> None:
    self.agent_name = agent_name


class WorkerAgent(BaseAgent):
  """A worker agent that attempts two tool calls through the real plugin
  manager: lookup_order (within its attenuated scope) and issue_refund
  (outside it). No LLM involved; the calls are issued directly, where
  ADK's own LLM flow would issue them."""

  async def _run_async_impl(self, ctx: InvocationContext):
    for tool_name, resource in (
        ("lookup_order", "data.orders"),
        ("issue_refund", "data.payments"),
    ):
      denial = await ctx.plugin_manager.run_before_tool_callback(
          tool=_FakeTool(tool_name),
          tool_args={"resource": resource},
          tool_context=_FakeToolContext(self.name),
      )
      outcome = "ALLOWED" if denial is None else f"DENIED ({denial['code']})"
      yield Event(
          invocation_id=ctx.invocation_id,
          author=self.name,
          content=gtypes.Content(
              role="model",
              parts=[gtypes.Part(text=f"{tool_name}: {outcome}")],
          ),
      )


class CoordinatorAgent(BaseAgent):
  """Drives ADK's native in-process delegation (sub_agent.run_async)."""

  async def _run_async_impl(self, ctx: InvocationContext):
    yield Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        content=gtypes.Content(
            role="model", parts=[gtypes.Part(text="delegating to worker")]
        ),
    )
    for sub in self.sub_agents:
      async for event in sub.run_async(ctx):
        yield event


async def main() -> None:
  # 1. Mint a root VAID for the coordinator, then an attenuated child for
  #    the worker. Ordinary vaid_mint usage, no ADK involved yet.
  kernel_key = Ed25519PrivateKey.generate()
  issuer = ReferenceIssuer.from_seed(
      kernel_key.private_bytes_raw(), vaid_ttl_hours=1, trust_domain=TRUST_DOMAIN
  )
  revocation = InMemoryRevocationList.assume_nothing_revoked()
  mint = MintService(issuer, InMemoryAudit())

  root = mint.mint_root(
      VaidSeed(
          agent_class="coordinator",
          version="1.0.0",
          tenant_id="acme",
          scope_boundary=[prefix_entry("data.orders"), prefix_entry("data.payments")],
          capability_set=["read", "write"],
      )
  )

  worker_key = Ed25519PrivateKey.generate()
  worker_public = worker_key.public_key().public_bytes_raw()
  worker_seed = VaidSeed(
      agent_class="worker",
      version="1.0.0",
      tenant_id="acme",
      parent_vaid=root["vaid_id"],
      scope_boundary=[prefix_entry("data.orders")],  # attenuated: no data.payments
      capability_set=["read"],  # attenuated: no write
      public_key_der=worker_public,
  )
  pop_nonce = "sample-nonce-1"
  pop_issued_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
  pop_payload = build_mint_pop_payload(
      worker_seed, public_key_der=worker_public, nonce=pop_nonce, issued_at=pop_issued_at
  )
  pop = MintPop(
      nonce=pop_nonce,
      issued_at=pop_issued_at,
      signature=worker_key.sign(canonical_request_signing_bytes(pop_payload)),
  )
  worker_child = mint.mint_child(worker_seed, root, pop).vaid

  # 2. Wire the plugin: one trusted kernel key, a vouching revocation store
  #    (demo only), and the explicit tool -> capability map.
  plugin = VaidIdentityPlugin(
      keys=TrustedIssuers.from_raw_keys([(TRUST_DOMAIN, issuer.kernel_public_key())]),
      revocation=revocation,
      tool_authorizations={
          "lookup_order": ToolAuthorization(capability="read", resource="data.orders"),
          "issue_refund": ToolAuthorization(capability="write", resource="data.payments"),
      },
  )
  plugin.register("coordinator", VaidPresentation(leaf=root))
  plugin.register("worker", VaidPresentation(leaf=worker_child, chain=(root,)))

  # 3. Build the agents and run through a real ADK Runner.
  worker = WorkerAgent(name="worker")
  coordinator = CoordinatorAgent(name="coordinator", sub_agents=[worker])

  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID, session_id="s1"
  )
  runner = Runner(
      app_name=APP_NAME,
      agent=coordinator,
      session_service=session_service,
      plugins=[plugin],
  )

  async for event in runner.run_async(
      user_id=USER_ID,
      session_id="s1",
      new_message=gtypes.Content(role="user", parts=[gtypes.Part(text="go")]),
  ):
    if event.content and event.content.parts:
      text = event.content.parts[0].text
      if text:
        print(f"[{event.author}] {text}")


if __name__ == "__main__":
  asyncio.run(main())
