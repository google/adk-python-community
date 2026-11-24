# Checkout guard plugin for Google ADK

A pre-payment check for shopping agents. Register the tool that moves money,
and the plugin reads the merchant's checkout right before that tool runs. It
blocks the payment, or asks the shopper through ADK tool confirmation, when
the checkout holds something the shopper did not agree to:

* a total over the shopper's limit, a changed total, or parts that do not add
  up to the total,
* a subscription or a free trial that turns into a paid plan,
* an item, quantity, tip or gift wrap the shopper did not ask for,
* fees that appear only at checkout,
* a merchant, ship-to postal code or approval that does not match,
* checkout text that gives instructions to the AI, or hides text from the
  shopper.

Every check is plain Python: regular expressions and arithmetic, no model
call and no network call, so text a merchant writes in the checkout cannot
talk its way past it. It reads checkouts from the
[Agentic Commerce Protocol](https://github.com/agentic-commerce-protocol/agentic-commerce-protocol)
(spec 2026-04-17) and the
[Universal Commerce Protocol](https://github.com/Universal-Commerce-Protocol/ucp),
including UCP purchase options. The 21 rules are a Python port of
[Fine Print](https://github.com/frostyhand/fine-print) and keep its rule ids.

## Usage

```python
from google.adk.apps import App
from google.adk_community.plugins.checkout_guard import (
    CheckoutGuardPlugin, MandateItem, ShopperMandate, from_acp,
)

async def load_checkout(args, tool_context):
  # Fetch the checkout from the merchant. Do not trust a copy from the model.
  session = await merchant.get_checkout_session(args["checkout_id"])
  return from_acp(session, merchant="shop.example")

guard = CheckoutGuardPlugin(
    payment_tools={"complete_checkout": load_checkout},
    mandate=ShopperMandate(
        max_total=12000,              # minor units: $120.00, all in
        currency="USD",
        items=(MandateItem("grinder", max_quantity=1),),
        allowed_merchants=("shop.example",),
        ship_to_postal_code="94110",
    ),
)

app = App(name="shop", root_agent=agent, plugins=[guard])
```

To set the mandate per shopper or per session, put
`ShopperMandate(...).to_dict()` in session state under
`checkout_guard:mandate`. A mandate in state takes precedence over the one
passed to the plugin.

## What the agent sees

* **block**: the tool does not run. The call is answered with
  `checkout_guard.summary`, the findings, and `suggested_fixes` when there is
  a safe change (remove a pre-selected line, switch a line to its one-time
  purchase option).
* **confirm**: the plugin calls `tool_context.request_confirmation` with the
  summary as the hint and the verdict as the payload. When the shopper
  approves, the checkout is loaded and checked again, and the tool runs only
  if nothing changed since the shopper saw it. A rejected or stale approval
  does not pay.
* **allow**: the tool runs.

A missing mandate or a checkout that cannot be loaded blocks the payment.

## Run the sample

The sample merchant has three checkouts: `cs_clean` (pays), `cs_tip` (asks
the shopper about a $3.00 tip) and `cs_addon` (blocked: a pre-selected
protection plan puts the total over budget).

```bash
export GOOGLE_API_KEY=...
cd contributing/samples/checkout_guard
python main.py cs_tip
python main.py cs_addon
```

Or open it in the dev UI with `adk web contributing/samples/checkout_guard`.

The rules point at the kinds of practices consumer rules target. A finding
does not mean a merchant broke a law, and a clean result does not mean a
checkout is compliant.
