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

"""Tests for CheckoutGuardPlugin.

The runner tests drive a real ADK Runner with a scripted model, so the
confirmation pause and resume go through ADK's own tool-confirmation flow.
"""

from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path
from typing import Any
from typing import AsyncGenerator
from typing import Optional

from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.genai import types
import pytest

from google.adk_community.plugins.checkout_guard import CheckoutGuardPlugin
from google.adk_community.plugins.checkout_guard import DEFAULT_MANDATE_STATE_KEY
from google.adk_community.plugins.checkout_guard import from_acp
from google.adk_community.plugins.checkout_guard import ShopperMandate

_FIXTURES = Path(__file__).parent / "checkout_guard_fixtures"
_CONFIRM = "adk_request_confirmation"


def _fixture(name: str) -> dict:
  return json.loads((_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


class _Merchant:
  """A fake ACP merchant. Tests change ``session`` to simulate the merchant."""

  def __init__(self, name: str):
    fx = _fixture(name)
    self.session: dict = fx["checkout"]
    # The plugin checks expiry against the real clock.
    self.session["expires_at"] = (
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1)
    ).isoformat()
    self.mandate = ShopperMandate.from_dict(fx["mandate"])
    self.paid: list[str] = []

  async def load(self, args: dict[str, Any], tool_context) -> Any:
    del tool_context
    assert args["checkout_id"] == "cs_demo"
    return from_acp(copy.deepcopy(self.session), merchant="shop.example")

  def complete_checkout(self, checkout_id: str) -> dict:
    """Pays for the checkout session.

    Args:
      checkout_id: The checkout session id.
    """
    self.paid.append(checkout_id)
    return {"status": "completed", "order_id": "ord_1"}


# Unit tests with a minimal tool context ---------------------------------------


class _Actions:
  skip_summarization = False


class _FakeToolContext:

  def __init__(self, state=None, confirmation=None):
    self.state = dict(state or {})
    self.function_call_id = "call_1"
    self.tool_confirmation = confirmation
    self.actions = _Actions()
    self.requested: Optional[dict] = None

  def request_confirmation(self, *, hint=None, payload=None):
    self.requested = {"hint": hint, "payload": payload}


class _Tool:

  def __init__(self, name: str):
    self.name = name


async def _before(plugin, ctx, tool="complete_checkout"):
  return await plugin.before_tool_callback(
      tool=_Tool(tool), tool_args={"checkout_id": "cs_demo"}, tool_context=ctx
  )


async def test_other_tools_are_not_checked():
  merchant = _Merchant("acp-preselected-protection-plan")
  plugin = CheckoutGuardPlugin(
      payment_tools={"complete_checkout": merchant.load},
      mandate=merchant.mandate,
  )
  assert await _before(plugin, _FakeToolContext(), tool="search") is None


async def test_clean_checkout_runs_the_tool():
  merchant = _Merchant("acp-clean")
  plugin = CheckoutGuardPlugin(
      payment_tools={"complete_checkout": merchant.load},
      mandate=merchant.mandate,
  )
  ctx = _FakeToolContext()
  assert await _before(plugin, ctx) is None
  assert ctx.requested is None


async def test_trap_is_blocked_with_findings_and_fixes():
  merchant = _Merchant("acp-preselected-protection-plan")
  plugin = CheckoutGuardPlugin(
      payment_tools={"complete_checkout": merchant.load},
      mandate=merchant.mandate,
  )
  ctx = _FakeToolContext()
  response = await _before(plugin, ctx)
  assert response["error"].startswith("Payment blocked")
  guard = response["checkout_guard"]
  assert guard["decision"] == "block"
  assert {f["rule"] for f in guard["findings"]} == {"FP101", "FP301", "FP401"}
  assert response["suggested_fixes"] == [{"remove_line_item_id": "li_2"}]
  assert ctx.requested is None


async def test_confirm_asks_the_shopper_with_the_summary():
  merchant = _Merchant("acp-tip-added")
  plugin = CheckoutGuardPlugin(
      payment_tools={"complete_checkout": merchant.load},
      mandate=merchant.mandate,
  )
  ctx = _FakeToolContext()
  response = await _before(plugin, ctx)
  assert "waiting for the shopper" in response["error"]
  assert ctx.requested["hint"].startswith("Check this before paying.")
  assert "A tip of $3.00" in ctx.requested["hint"]
  assert ctx.requested["payload"]["checkout_guard"]["decision"] == "confirm"
  assert ctx.actions.skip_summarization is True
  assert ctx.state["checkout_guard:pending:call_1"]


async def test_missing_mandate_fails_closed():
  merchant = _Merchant("acp-clean")
  plugin = CheckoutGuardPlugin(
      payment_tools={"complete_checkout": merchant.load}
  )
  response = await _before(plugin, _FakeToolContext())
  assert response["checkout_guard"]["decision"] == "block"
  assert "No shopper mandate" in response["error"]


async def test_invalid_mandate_in_state_fails_closed():
  merchant = _Merchant("acp-clean")
  plugin = CheckoutGuardPlugin(
      payment_tools={"complete_checkout": merchant.load},
      mandate=merchant.mandate,
  )
  ctx = _FakeToolContext(state={DEFAULT_MANDATE_STATE_KEY: {"maxTotal": 1}})
  response = await _before(plugin, ctx)
  assert "not valid" in response["error"]


async def test_mandate_in_state_wins_over_the_default():
  merchant = _Merchant("acp-clean")
  plugin = CheckoutGuardPlugin(
      payment_tools={"complete_checkout": merchant.load},
      mandate=merchant.mandate,
  )
  tight = ShopperMandate(max_total=100, currency="USD").to_dict()
  ctx = _FakeToolContext(state={DEFAULT_MANDATE_STATE_KEY: tight})
  response = await _before(plugin, ctx)
  assert [f["rule"] for f in response["checkout_guard"]["findings"]] == [
      "FP101"
  ]


async def test_loader_error_fails_closed():
  async def broken(args, tool_context):
    raise ConnectionError("merchant down")

  plugin = CheckoutGuardPlugin(
      payment_tools={"complete_checkout": broken},
      mandate=ShopperMandate(max_total=100, currency="USD"),
  )
  response = await _before(plugin, _FakeToolContext())
  assert "could not be loaded" in response["error"]


async def test_sync_loader_is_supported():
  merchant = _Merchant("acp-clean")

  def load(args, tool_context):
    return from_acp(merchant.session, merchant="shop.example")

  plugin = CheckoutGuardPlugin(
      payment_tools={"complete_checkout": load}, mandate=merchant.mandate
  )
  assert await _before(plugin, _FakeToolContext()) is None


def test_payment_tools_are_required():
  with pytest.raises(ValueError):
    CheckoutGuardPlugin(payment_tools={})


# Runner tests -----------------------------------------------------------------


class _ScriptedModel(BaseLlm):
  """Calls the payment tool once, then answers with text."""

  model: str = "scripted"

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    already_called = any(
        part.function_call and part.function_call.name == "complete_checkout"
        for content in llm_request.contents
        for part in content.parts or []
    )
    if already_called:
      part = types.Part(text="Done.")
    else:
      part = types.Part(
          function_call=types.FunctionCall(
              name="complete_checkout", args={"checkout_id": "cs_demo"}
          )
      )
    yield LlmResponse(content=types.Content(role="model", parts=[part]))


async def _start(merchant: _Merchant):
  agent = LlmAgent(
      name="shopper",
      model=_ScriptedModel(),
      tools=[FunctionTool(merchant.complete_checkout)],
  )
  plugin = CheckoutGuardPlugin(
      payment_tools={"complete_checkout": merchant.load},
      mandate=merchant.mandate,
  )
  runner = InMemoryRunner(
      app=App(name="shop", root_agent=agent, plugins=[plugin])
  )
  session = await runner.session_service.create_session(
      app_name="shop", user_id="u"
  )
  events = await _run(
      runner,
      session.id,
      types.Content(role="user", parts=[types.Part(text="Buy it.")]),
  )
  return runner, session.id, events


async def _run(runner, session_id: str, message: types.Content) -> list:
  return [
      e
      async for e in runner.run_async(
          user_id="u", session_id=session_id, new_message=message
      )
  ]


def _confirmation_call(events) -> Optional[types.FunctionCall]:
  for e in events:
    for call in e.get_function_calls():
      if call.name == _CONFIRM:
        return call
  return None


def _answer(call: types.FunctionCall, confirmed: bool) -> types.Content:
  return types.Content(
      role="user",
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id=call.id,
                  name=_CONFIRM,
                  response=ToolConfirmation(confirmed=confirmed).model_dump(),
              )
          )
      ],
  )


async def test_runner_blocks_a_trap_without_paying():
  merchant = _Merchant("acp-renewal-in-disclosure")
  _, _, events = await _start(merchant)
  assert merchant.paid == []
  assert _confirmation_call(events) is None
  responses = [r.response for e in events for r in e.get_function_responses()]
  assert responses[0]["checkout_guard"]["decision"] == "block"


async def test_runner_pays_after_the_shopper_approves():
  merchant = _Merchant("acp-tip-added")
  runner, session_id, events = await _start(merchant)
  call = _confirmation_call(events)
  assert call is not None
  assert merchant.paid == []
  hint = call.args["toolConfirmation"]["hint"]
  assert "A tip of $3.00 is included" in hint

  await _run(runner, session_id, _answer(call, confirmed=True))
  assert merchant.paid == ["cs_demo"]


async def test_runner_does_not_pay_when_the_shopper_rejects():
  merchant = _Merchant("acp-tip-added")
  runner, session_id, events = await _start(merchant)
  events = await _run(
      runner, session_id, _answer(_confirmation_call(events), confirmed=False)
  )
  assert merchant.paid == []
  responses = [r.response for e in events for r in e.get_function_responses()]
  assert any("did not approve" in str(r) for r in responses)


async def test_runner_refuses_an_approval_replayed_onto_a_changed_cart():
  merchant = _Merchant("acp-tip-added")
  runner, session_id, events = await _start(merchant)
  call = _confirmation_call(events)
  # The merchant raises the tip after the shopper saw the summary.
  for total in merchant.session["totals"]:
    if total["type"] in ("tip", "total"):
      total["amount"] += 200

  events = await _run(runner, session_id, _answer(call, confirmed=True))
  assert merchant.paid == []
  responses = [r.response for e in events for r in e.get_function_responses()]
  assert any("changed after the shopper approved" in str(r) for r in responses)
