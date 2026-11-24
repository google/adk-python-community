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

"""ADK plugin that checks a checkout before an agent's payment tool runs."""

from __future__ import annotations

import inspect
import logging
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import Union

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from ._check import check
from ._models import Checkout
from ._models import ShopperMandate
from ._models import Verdict

logger = logging.getLogger("google_adk." + __name__)

CheckoutLoader = Callable[
    [dict[str, Any], ToolContext], Union[Checkout, Awaitable[Checkout]]
]
"""Loads the checkout a payment tool call is about to pay for.

Receives the tool call's arguments and context. It should fetch the checkout
from the merchant as it stands now, not reuse a copy the model passed in, and
turn it into a ``Checkout`` with ``from_acp`` or ``from_ucp``.
"""

DEFAULT_MANDATE_STATE_KEY = "checkout_guard:mandate"
_PENDING_PREFIX = "checkout_guard:pending:"


class CheckoutGuardPlugin(BasePlugin):
  """Checks the merchant's checkout before a payment tool runs.

  The plugin watches the tools named in ``payment_tools``. Before one of them
  runs, it loads the checkout, compares it with what the shopper approved
  (the ``ShopperMandate``) and decides:

  * ``block``: the call is answered with the problems and the tool does not
    run. Examples are a total over budget, a subscription the shopper did not
    ask for, an add-on that was pre-selected, or parts that do not add up to
    the total.
  * ``confirm``: the plugin asks the shopper through ADK tool confirmation
    (``tool_context.request_confirmation``), with a plain-language summary of
    the charge as the hint and the verdict as the payload. When the shopper
    approves, the checkout is loaded and checked again, and the tool runs
    only if nothing changed since the shopper saw it.
  * ``allow``: the tool runs.

  Every check is deterministic code with no model call, so text a merchant
  puts in the checkout cannot talk its way past it. The plugin fails closed:
  a missing mandate or a checkout that cannot be loaded blocks the payment.

  Args:
    payment_tools: Maps the name of each tool that moves money (for example
      ``complete_checkout``) to a loader that returns the checkout that call
      would pay for.
    mandate: The shopper's mandate. Used when session state has none.
    mandate_state_key: Session state key holding the mandate as a dict (see
      ``ShopperMandate.to_dict``). It takes precedence over ``mandate``, so an
      app can set a mandate per shopper or per session.
    skip_rules: Rule ids to leave out, for example ``("FP103",)``.
    name: The plugin name.

  Example::

    async def load_checkout(args, tool_context):
      session = await merchant.get_checkout_session(args["checkout_id"])
      return from_acp(session, merchant="shop.example")

    plugin = CheckoutGuardPlugin(
        payment_tools={"complete_checkout": load_checkout},
        mandate=ShopperMandate(max_total=12000, currency="USD"),
    )
    runner = Runner(agent=agent, plugins=[plugin], ...)
  """

  def __init__(
      self,
      *,
      payment_tools: Mapping[str, CheckoutLoader],
      mandate: Optional[ShopperMandate] = None,
      mandate_state_key: str = DEFAULT_MANDATE_STATE_KEY,
      skip_rules: Iterable[str] = (),
      name: str = "checkout_guard",
  ) -> None:
    super().__init__(name=name)
    if not payment_tools:
      raise ValueError("payment_tools must name at least one tool.")
    self._payment_tools = dict(payment_tools)
    self._mandate = mandate
    self._mandate_state_key = mandate_state_key
    self._skip_rules = tuple(skip_rules)

  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[dict]:
    loader = self._payment_tools.get(tool.name)
    if loader is None:
      return None

    pending_key = f"{_PENDING_PREFIX}{tool_context.function_call_id}"
    confirmation = tool_context.tool_confirmation
    approved_digest = None
    if confirmation is not None:
      approved_digest = tool_context.state.get(pending_key)
      tool_context.state[pending_key] = None
      if not confirmation.confirmed:
        return {"error": "The shopper did not approve this payment."}

    try:
      mandate = self._load_mandate(tool_context)
    except (TypeError, ValueError) as e:
      return _fail_closed(f"The shopper's mandate is not valid: {e}")
    if mandate is None:
      return _fail_closed(
          "No shopper mandate is set, so the payment cannot be checked."
      )

    try:
      checkout = loader(tool_args, tool_context)
      if inspect.isawaitable(checkout):
        checkout = await checkout
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Could not load the checkout for %s: %s", tool.name, e)
      return _fail_closed("The checkout could not be loaded to check it.")

    verdict = check(mandate, checkout, skip_rules=self._skip_rules)
    logger.info(
        "Checkout guard %s for %s (%s)",
        verdict.decision,
        tool.name,
        ", ".join(f.rule for f in verdict.findings) or "no findings",
    )

    if verdict.decision == "block":
      return _blocked(verdict)

    if confirmation is not None:
      # The shopper approved a summary. Pay only if it still describes the
      # checkout, so an approval cannot be replayed onto a different cart.
      if approved_digest != verdict.digest:
        return {
            "error": (
                "The checkout changed after the shopper approved it. Check"
                " it again before paying."
            ),
            "checkout_guard": verdict.to_dict(),
        }
      return None

    if verdict.decision == "confirm":
      tool_context.request_confirmation(
          hint=verdict.summary,
          payload={"checkout_guard": verdict.to_dict()},
      )
      tool_context.state[pending_key] = verdict.digest
      # The pause is not a result for the model to summarize.
      tool_context.actions.skip_summarization = True
      return {"error": "This payment is waiting for the shopper's approval."}

    return None

  def _load_mandate(
      self, tool_context: ToolContext
  ) -> Optional[ShopperMandate]:
    stored = tool_context.state.get(self._mandate_state_key)
    if stored is None:
      return self._mandate
    if isinstance(stored, ShopperMandate):
      return stored
    if not isinstance(stored, Mapping):
      raise TypeError(f"expected a dict, got {type(stored).__name__}")
    return ShopperMandate.from_dict(stored)


def _blocked(verdict: Verdict) -> dict[str, Any]:
  fixes = [f.fix for f in verdict.findings if f.fix]
  response: dict[str, Any] = {
      "error": (
          "Payment blocked. Do not pay. Tell the shopper what is wrong, using"
          " the summary."
      ),
      "checkout_guard": verdict.to_dict(),
  }
  if fixes:
    response["suggested_fixes"] = fixes
  return response


def _fail_closed(reason: str) -> dict[str, Any]:
  return {
      "error": f"Payment blocked. {reason}",
      "checkout_guard": {"decision": "block", "reason": reason},
  }
