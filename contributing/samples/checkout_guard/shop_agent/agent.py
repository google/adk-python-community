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

"""A shopping agent whose payment tool is guarded by CheckoutGuardPlugin.

The merchant is an in-memory stand-in for an Agentic Commerce Protocol
merchant with three checkout sessions:

* ``cs_clean``: one burr grinder, nothing unusual. The payment goes through.
* ``cs_tip``: the merchant added a $3.00 tip. The shopper is asked first.
* ``cs_addon``: a protection plan was pre-selected and the total is over
  budget. The payment is blocked.
"""

from __future__ import annotations

import copy
import os
from typing import Any

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.tools.tool_context import ToolContext

from google.adk_community.plugins.checkout_guard import CheckoutGuardPlugin
from google.adk_community.plugins.checkout_guard import from_acp
from google.adk_community.plugins.checkout_guard import MandateItem
from google.adk_community.plugins.checkout_guard import ShopperMandate

MERCHANT = "shop.example"


def _session(checkout_id: str, extra_lines=(), extra_totals=()) -> dict:
  lines = [
      {
          "id": "li_1",
          "item": {"id": "GRINDER-1", "quantity": 1},
          "name": "Burr coffee grinder",
          "quantity": 1,
          "unit_amount": 8900,
          "totals": [
              {"type": "subtotal", "display_text": "Subtotal", "amount": 8900},
              {"type": "total", "display_text": "Total", "amount": 8900},
          ],
      },
      *extra_lines,
  ]
  subtotal = sum(li["unit_amount"] * li["quantity"] for li in lines)
  totals = [
      {"type": "subtotal", "display_text": "Subtotal", "amount": subtotal},
      {"type": "fulfillment", "display_text": "Shipping", "amount": 599},
      {"type": "tax", "display_text": "Tax", "amount": 790},
      *extra_totals,
  ]
  total = sum(t["amount"] for t in totals)
  return {
      "id": checkout_id,
      "status": "ready_for_payment",
      "currency": "usd",
      "line_items": lines,
      "fulfillment_details": {"address": {"postal_code": "94110"}},
      "totals": [
          *totals,
          {"type": "total", "display_text": "Total", "amount": total},
      ],
  }


SESSIONS: dict[str, dict] = {
    "cs_clean": _session("cs_clean"),
    "cs_tip": _session(
        "cs_tip",
        extra_totals=[{"type": "tip", "display_text": "Tip", "amount": 300}],
    ),
    "cs_addon": _session(
        "cs_addon",
        extra_lines=[{
            "id": "li_2",
            "item": {"id": "PLAN-3Y", "quantity": 1},
            "name": "3-year protection plan",
            "quantity": 1,
            "unit_amount": 2999,
            "totals": [
                {"type": "total", "display_text": "Total", "amount": 2999}
            ],
        }],
    ),
}

ORDERS: list[str] = []


def get_checkout(checkout_id: str) -> dict:
  """Returns the merchant's checkout session.

  Args:
    checkout_id: The checkout session id, for example cs_clean.
  """
  return SESSIONS.get(checkout_id) or {"error": f"No checkout {checkout_id}."}


def complete_checkout(checkout_id: str) -> dict:
  """Pays for a checkout session. This charges the shopper.

  Args:
    checkout_id: The checkout session id.
  """
  ORDERS.append(checkout_id)
  return {"status": "completed", "order_id": f"ord_{len(ORDERS)}"}


async def load_checkout(args: dict[str, Any], tool_context: ToolContext):
  """Fetches the checkout from the merchant, not from the model's arguments."""
  del tool_context
  session = SESSIONS.get(args.get("checkout_id", ""))
  if session is None:
    raise KeyError(args.get("checkout_id"))
  return from_acp(copy.deepcopy(session), merchant=MERCHANT)


checkout_guard = CheckoutGuardPlugin(
    payment_tools={"complete_checkout": load_checkout},
    mandate=ShopperMandate(
        max_total=12000,
        currency="USD",
        items=(MandateItem("grinder", max_quantity=1),),
        allowed_merchants=(MERCHANT,),
        ship_to_postal_code="94110",
    ),
)

root_agent = Agent(
    name="shop_agent",
    model=os.environ.get("ADK_MODEL", "gemini-flash-latest"),
    instruction=(
        "You buy things for the shopper. Look at the checkout with"
        " get_checkout, then pay with complete_checkout. If the payment is"
        " blocked, tell the shopper what is wrong in plain words and do not"
        " try to pay again."
    ),
    tools=[get_checkout, complete_checkout],
)

app = App(name="shop_agent", root_agent=root_agent, plugins=[checkout_guard])
