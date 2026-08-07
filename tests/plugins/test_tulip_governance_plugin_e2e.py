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

"""End-to-end tests for TulipGovernancePlugin — through the real ADK
dispatch path, not by calling the plugin's method directly.

test_tulip_governance_plugin.py calls ``plugin.before_tool_callback(...)``
by hand. These tests instead build a real ``LlmAgent`` with a real tool
function, run it through the real ``InMemoryRunner``/``PluginManager``, and
check whether the tool's Python body actually executed — proving the
choke point exists at the framework level, not just in this plugin's own
code. Read from ``google/adk-python``'s
``src/google/adk/flows/llm_flows/functions.py`` (``_run_with_trace``),
confirmed by line: the real tool call (``__call_tool_async``, Step 3) is
gated behind ``if function_response is None`` — Step 1 populates
``function_response`` from ``plugin_manager.run_before_tool_callback``, and
when a plugin returns a non-``None`` dict there, Step 3 never runs. This is
the actual mechanism, not an assumption about what "short-circuit" means.

``_MockModel`` narrowly reproduces the one relevant piece of
``google/adk-python``'s own ``tests/unittests/testing_utils.py::MockModel``
(same project, Apache-2.0) — a scripted, turn-by-turn fake ``BaseLlm`` — as
that module is test-only and not part of the installable ``google-adk``
package.
"""

from __future__ import annotations

import asyncio
from typing import Any
from typing import AsyncGenerator

import pytest

tulip = pytest.importorskip("tulip", reason="tulip-agents is not installed")

from google.adk.agents.llm_agent import LlmAgent  # noqa: E402
from google.adk.apps.app import App  # noqa: E402
from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402
from typing_extensions import override  # noqa: E402

from google.adk_community.plugins.tulip_governance_plugin import (  # noqa: E402
    TulipGovernancePlugin,
)
from tulip.control import Action  # noqa: E402
from tulip.security.audit import AuditTrail  # noqa: E402
from tulip.security.policy import ControlPolicy  # noqa: E402


class _MockModel(BaseLlm):
  """A scripted BaseLlm: yields one canned LlmResponse per call, in order."""

  model: str = "mock"
  responses: list[LlmResponse] = []
  index: int = -1

  @classmethod
  def supported_models(cls) -> list[str]:
    return ["mock"]

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    self.index += 1
    yield self.responses[self.index]


def _model_that_calls(tool_name: str, args: dict[str, Any]) -> _MockModel:
  """Turn 1: request `tool_name(**args)`. Turn 2: a plain-text reply, so the
  run completes normally whatever the tool call actually did."""
  return _MockModel(
      responses=[
          LlmResponse(
              content=types.Content(
                  role="model",
                  parts=[
                      types.Part.from_function_call(name=tool_name, args=args)
                  ],
              )
          ),
          LlmResponse(
              content=types.Content(
                  role="model",
                  parts=[types.Part.from_text(text="done")],
              )
          ),
      ]
  )


def _make_wipe_tool() -> tuple[Any, list[str]]:
  """A tool with a real, observable side effect — the only reliable way to
  tell "the plugin returned a dict" apart from "the tool actually ran"."""
  calls: list[str] = []

  def wipe_database() -> str:
    """Wipes the production database. Irreversible."""
    calls.append("wiped")
    return "database wiped"

  return wipe_database, calls


async def _run(agent: LlmAgent, plugin: TulipGovernancePlugin) -> list:
  runner = InMemoryRunner(
      app=App(name="test_app", root_agent=agent, plugins=[plugin])
  )
  session = await runner.session_service.create_session(
      app_name="test_app", user_id="u1"
  )
  events = []
  async for event in runner.run_async(
      user_id="u1",
      session_id=session.id,
      new_message=types.Content(
          role="user", parts=[types.Part.from_text(text="go")]
      ),
  ):
    events.append(event)
  return events


def _function_response_payload(events: list) -> dict[str, Any] | None:
  """Pulls the dict the tool call actually resolved to, from the function
  response event — i.e. what the *model* saw come back."""
  for event in events:
    if not event.content or not event.content.parts:
      continue
    for part in event.content.parts:
      if part.function_response:
        return dict(part.function_response.response or {})
  return None


class TestTulipGovernancePluginE2E:

  @pytest.mark.asyncio
  async def test_deny_prevents_the_tool_from_ever_running(self):
    wipe_database, calls = _make_wipe_tool()
    trail = AuditTrail()
    policy = ControlPolicy(
        require_verification_score=0.0, deny_for=frozenset({"irreversible"})
    )
    plugin = TulipGovernancePlugin(
        policy=policy,
        action=lambda tool, args: Action(
            name=tool.name, kind="irreversible", environment="production"
        ),
        trail=trail,
    )
    agent = LlmAgent(
        name="test_agent",
        model=_model_that_calls("wipe_database", {}),
        tools=[wipe_database],
    )

    events = await _run(agent, plugin)

    # The strongest possible assertion: the tool's Python body never ran.
    assert calls == []
    payload = _function_response_payload(events)
    assert payload is not None
    assert payload["error"] == "policy_denied"
    # The run still completed (turn 2's "done" text is present) — a denial
    # is a tool result the model sees, not a crash.
    final_text = "".join(
        part.text or ""
        for event in events
        if event.content
        for part in event.content.parts or []
        if part.text
    )
    assert "done" in final_text
    # And the denial is on the tamper-evident trail.
    assert len(trail) == 1
    assert trail.records()[0].payload["outcome"] == "deny"
    assert trail.verify() is True

  @pytest.mark.asyncio
  async def test_allow_lets_the_real_tool_run(self):
    """The mirror image of the deny test: prove the plugin does not block
    everything by default — an allowed call really executes."""
    wipe_database, calls = _make_wipe_tool()
    policy = ControlPolicy(require_verification_score=0.0)
    plugin = TulipGovernancePlugin(
        policy=policy,
        action=lambda tool, args: Action(tool.name, environment="staging"),
    )
    agent = LlmAgent(
        name="test_agent",
        model=_model_that_calls("wipe_database", {}),
        tools=[wipe_database],
    )

    events = await _run(agent, plugin)

    assert calls == ["wiped"]
    payload = _function_response_payload(events)
    assert payload is not None
    assert payload.get("result") == "database wiped"
    assert plugin.trail.records()[0].payload["outcome"] == "allow"

  @pytest.mark.asyncio
  async def test_require_human_also_prevents_the_tool_from_running(self):
    """require_human is a distinct outcome from deny, but it is just as
    effective at stopping the call — held, not merely logged."""
    wipe_database, calls = _make_wipe_tool()
    policy = ControlPolicy(
        require_verification_score=0.0,
        require_human_for=frozenset({"production"}),
    )
    plugin = TulipGovernancePlugin(
        policy=policy,
        action=lambda tool, args: Action(tool.name, environment="production"),
    )
    agent = LlmAgent(
        name="test_agent",
        model=_model_that_calls("wipe_database", {}),
        tools=[wipe_database],
    )

    events = await _run(agent, plugin)

    assert calls == []
    payload = _function_response_payload(events)
    assert payload is not None
    assert payload["error"] == "held_for_approval"
    assert plugin.trail.records()[0].payload["outcome"] == "require_human"

  @pytest.mark.asyncio
  async def test_two_tools_in_one_turn_are_each_gated_independently(self):
    """Adversarial: a single model turn requesting two calls back to back —
    one that should be denied and one that should be allowed. Neither
    decision may leak into the other."""

    calls: list[str] = []

    def safe_read() -> str:
      calls.append("read")
      return "ok"

    def dangerous_write() -> str:
      calls.append("wrote")
      return "ok"

    # require_human_for defaults to {"production"} (ControlPolicy's own
    # conservative default) — safe_read must be staging, not production, or
    # it too would be held for a human rather than allowed, which is a
    # ControlPolicy default worth knowing, not a bug in this fixture.
    policy = ControlPolicy(
        require_verification_score=0.0, deny_for=frozenset({"irreversible"})
    )

    def build_action(tool: Any, args: dict) -> Action:
      if tool.name == "dangerous_write":
        return Action(
            name=tool.name, kind="irreversible", environment="production"
        )
      return Action(name=tool.name, environment="staging")

    trail = AuditTrail()
    plugin = TulipGovernancePlugin(
        policy=policy, action=build_action, trail=trail
    )

    # Explicit, distinct ids: a real model response always carries one per
    # call so its function_response can be correlated back; two calls with
    # the same id=None silently collapsed to a single dispatched call in an
    # earlier version of this test (both bodies below never ran, catching
    # the fixture bug rather than a plugin bug) — the ids matter to the
    # test, not just to realism.
    model = _MockModel(
        responses=[
            LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id="call_read", name="safe_read", args={}
                            )
                        ),
                        types.Part(
                            function_call=types.FunctionCall(
                                id="call_write", name="dangerous_write", args={}
                            )
                        ),
                    ],
                )
            ),
            LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text="done")]
                )
            ),
        ]
    )
    agent = LlmAgent(
        name="test_agent", model=model, tools=[safe_read, dangerous_write]
    )

    await _run(agent, plugin)

    assert calls == ["read"]  # safe_read ran; dangerous_write did not
    outcomes = {
        r.payload["tool"]: r.payload["outcome"] for r in trail.records()
    }
    assert outcomes == {"safe_read": "allow", "dangerous_write": "deny"}

  @pytest.mark.asyncio
  async def test_a_raising_action_builder_fails_closed_not_open(self):
    """Adversarial: what happens when the caller's own action-builder
    throws (e.g. a KeyError reading an unexpected arg shape)? It must not
    silently degrade to "allow" — and it does not: the exception propagates
    out of before_tool_callback, so ADK never reaches Step 3
    (__call_tool_async) at all. ADK's own PluginManager wraps whatever a
    plugin raises in a RuntimeError (google/adk/plugins/plugin_manager.py,
    ``_run_callbacks``) rather than letting the original exception type
    escape — confirmed here rather than assumed, since it changes what a
    caller wrapping ``runner.run_async()`` needs to catch. Fail-closed but
    not graceful: an ordinary run crashes rather than degrading to a
    decision."""
    wipe_database, calls = _make_wipe_tool()

    def broken_action(tool: Any, args: dict) -> Action:
      raise KeyError("order_id")

    plugin = TulipGovernancePlugin(
        policy=ControlPolicy(require_verification_score=0.0),
        action=broken_action,
    )
    agent = LlmAgent(
        name="test_agent",
        model=_model_that_calls("wipe_database", {}),
        tools=[wipe_database],
    )
    runner = InMemoryRunner(
        app=App(name="test_app", root_agent=agent, plugins=[plugin])
    )
    session = await runner.session_service.create_session(
        app_name="test_app", user_id="u1"
    )

    with pytest.raises(RuntimeError, match="tulip_governance") as excinfo:
      async for _event in runner.run_async(
          user_id="u1",
          session_id=session.id,
          new_message=types.Content(
              role="user", parts=[types.Part.from_text(text="go")]
          ),
      ):
        pass
    assert isinstance(excinfo.value.__cause__, KeyError)

    assert calls == []  # the tool never ran, even though the run crashed

  @pytest.mark.asyncio
  async def test_concurrent_tool_calls_do_not_corrupt_the_audit_chain(self):
    """Adversarial: N before_tool_callback invocations racing under
    asyncio.gather (not a real ADK path — before_tool_callback calls are
    serialized per turn in functions.py — but the plugin's own contract
    should not silently rely on that). AuditTrail.record() has no internal
    await point, so it cannot be interleaved by the cooperative scheduler;
    this proves that invariant holds under real concurrent pressure rather
    than asserting it from reading the source."""
    trail = AuditTrail()
    policy = ControlPolicy(require_verification_score=0.0)
    plugin = TulipGovernancePlugin(
        policy=policy,
        action=lambda tool, args: Action(tool.name, environment="staging"),
        trail=trail,
    )

    class _Tool:

      def __init__(self, name: str):
        self.name = name

    class _Ctx:
      pass

    n = 50
    await asyncio.gather(*(
        plugin.before_tool_callback(
            tool=_Tool(f"tool_{i}"), tool_args={}, tool_context=_Ctx()
        )
        for i in range(n)
    ))

    assert len(trail) == n
    seqs = [r.seq for r in trail.records()]
    assert seqs == list(range(n))  # no duplicate/skipped seq under races
    assert trail.verify() is True
    assert {r.payload["tool"] for r in trail.records()} == {
        f"tool_{i}" for i in range(n)
    }
