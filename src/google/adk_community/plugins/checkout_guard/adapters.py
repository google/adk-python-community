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

"""Turns ACP and UCP checkout objects into a protocol-neutral Checkout.

* Agentic Commerce Protocol, spec 2026-04-17. ACP has no structured field for
  recurring charges, so a subscription can only be disclosed in free text
  (``disclosures``, ``description``, ``messages``). The adapter scans that text.
* Universal Commerce Protocol, including the purchase options extension and
  the ``dev.ucp.shopping.policy.subscription`` policy. In UCP the merchant
  picks the default purchase option when the platform does not choose, so
  the adapter records the one-time option a line could switch to.
"""

from __future__ import annotations

from typing import Any
from typing import Literal
from typing import Mapping
from typing import Optional

from ._models import Checkout
from ._models import Fee
from ._models import Line
from ._models import Recurring
from ._models import TextBlock
from ._text import find_recurring

Json = Mapping[str, Any]

_ACP_READY = "ready_for_payment"
_UCP_READY = "ready_for_complete"
_UCP_SUBSCRIPTION_POLICY = "dev.ucp.shopping.policy.subscription"


def from_acp(session: Json, *, merchant: Optional[str] = None) -> Checkout:
  """Builds a Checkout from an ACP CheckoutSession.

  Args:
    session: The CheckoutSession JSON as the merchant returned it.
    merchant: ACP sessions do not name the merchant. Pass the domain the
      agent called.
  """
  texts: list[TextBlock] = []
  lines: list[Line] = []
  for i, li in enumerate(session.get("line_items") or []):
    where = f"$.line_items[{i}]"
    item = li.get("item") or {}
    line_texts: list[TextBlock] = []
    for key in ("name", "description"):
      if isinstance(li.get(key), str):
        line_texts.append(TextBlock(f"{where}.{key}", li[key]))
    for j, d in enumerate(li.get("disclosures") or []):
      if isinstance(d.get("content"), str):
        line_texts.append(
            TextBlock(
                f"{where}.disclosures[{j}]",
                d["content"],
                _format(d.get("content_type")),
            )
        )
    texts.extend(line_texts)
    lines.append(
        Line(
            id=str(li.get("id")),
            title=li.get("name")
            or item.get("name")
            or str(item.get("id") or li.get("id")),
            quantity=int(li.get("quantity") or item.get("quantity") or 1),
            unit_amount=_first(li.get("unit_amount"), item.get("unit_amount")),
            total=_amount_of(li.get("totals"), "total"),
            sku=li.get("sku") or li.get("product_id"),
            parent_id=li.get("parent_id"),
            recurring=_recurring_from_text(line_texts),
        )
    )

  texts.extend(_message_texts(session))
  totals = session.get("totals") or []
  return Checkout(
      protocol="acp",
      merchant=merchant,
      currency=str(session.get("currency") or "usd").upper(),
      status=session.get("status"),
      price_final=session.get("status") == _ACP_READY,
      lines=lines,
      fees=_fees(totals),
      subtotal=_amount_of(totals, "subtotal"),
      discounts=_magnitude(totals, "discount"),
      item_discounts=_magnitude(totals, "items_discount"),
      shipping=_amount_of(totals, "fulfillment"),
      tax=_amount_of(totals, "tax"),
      tip=_amount_of(totals, "tip"),
      gift_wrap=_amount_of(totals, "gift_wrap"),
      store_credit=_amount_of(totals, "store_credit"),
      total=_amount_of(totals, "total") or 0,
      ship_to_postal_code=(
          ((session.get("fulfillment_details") or {}).get("address") or {}).get(
              "postal_code"
          )
      ),
      expires_at=session.get("expires_at"),
      texts=texts,
  )


def from_ucp(checkout: Json, *, merchant: Optional[str] = None) -> Checkout:
  """Builds a Checkout from a UCP checkout object.

  Args:
    checkout: The checkout JSON as the business returned it.
    merchant: The business the agent called, for the merchant allowlist.
  """
  texts: list[TextBlock] = []
  policies = checkout.get("policies") or []
  for i, p in enumerate(policies):
    texts.extend(
        _desc_texts(p.get("description"), f"$.policies[{i}].description")
    )

  lines: list[Line] = []
  for i, li in enumerate(checkout.get("line_items") or []):
    where = f"$.line_items[{i}]"
    item = li.get("item") or {}
    options = li.get("purchase_options") or []
    selected_id = li.get("selected_purchase_option_id") or (
        options[0].get("id") if options else None
    )
    selected_index = next(
        (j for j, o in enumerate(options) if o.get("id") == selected_id), None
    )
    selected = options[selected_index] if selected_index is not None else None
    one_time = next((o for o in options if not o.get("schedules")), None)

    line_texts: list[TextBlock] = []
    if isinstance(item.get("title"), str):
      line_texts.append(TextBlock(f"{where}.item.title", item["title"]))
    for j, o in enumerate(options):
      opt_where = f"{where}.purchase_options[{j}]"
      line_texts.extend(
          _desc_texts(o.get("description"), f"{opt_where}.description")
      )
      for k, s in enumerate(o.get("schedules") or []):
        line_texts.extend(
            _desc_texts(
                s.get("description"), f"{opt_where}.schedules[{k}].description"
            )
        )
    texts.extend(line_texts)

    policy = next(
        (
            p
            for p in policies
            if p.get("type") == _UCP_SUBSCRIPTION_POLICY
            and _applies_to(p, i, li.get("id"))
        ),
        None,
    )
    recurring: Optional[Recurring] = None
    if selected and selected.get("schedules"):
      first = selected["schedules"][0]
      price = _first(
          (selected.get("price") or {}).get("amount"),
          (item.get("price") or {}).get("amount"),
      )
      recurring = Recurring(
          source="structured",
          interval=first.get("interval"),
          amount=_amount_of(first.get("totals"), "total"),
          starts_at=first.get("starts_at"),
          trial=price is not None and int(price) == 0,
          cancellation_described=policy is not None,
          cancellation_url=policy.get("url") if policy else None,
      )
    else:
      # Only the selected option's text describes what is being bought.
      relevant = [
          t
          for t in line_texts
          if ".purchase_options[" not in t.where
          or (
              selected_index is not None
              and t.where.startswith(
                  f"{where}.purchase_options[{selected_index}]"
              )
          )
      ]
      recurring = _recurring_from_text(relevant)
      if recurring and policy is not None:
        recurring.cancellation_described = True

    lines.append(
        Line(
            id=str(li.get("id", i)),
            title=item.get("title") or str(item.get("id") or li.get("id")),
            quantity=int(li.get("quantity") or 1),
            unit_amount=(item.get("price") or {}).get("amount"),
            total=_amount_of(li.get("totals"), "total"),
            sku=item.get("id"),
            parent_id=li.get("parent_id"),
            recurring=recurring,
            selected_option_id=selected_id,
            one_time_option_id=(
                one_time.get("id") if recurring and one_time else None
            ),
        )
    )

  texts.extend(_message_texts(checkout))
  totals = checkout.get("totals") or []
  return Checkout(
      protocol="ucp",
      merchant=merchant,
      currency=str(checkout.get("currency") or "USD").upper(),
      status=checkout.get("status"),
      price_final=checkout.get("status") == _UCP_READY,
      lines=lines,
      fees=_fees(totals),
      subtotal=_amount_of(totals, "subtotal"),
      discounts=_magnitude(totals, "discount"),
      item_discounts=_magnitude(totals, "items_discount"),
      shipping=_amount_of(totals, "fulfillment"),
      tax=_amount_of(totals, "tax"),
      total=_amount_of(totals, "total") or 0,
      ship_to_postal_code=_ucp_ship_to(checkout),
      expires_at=checkout.get("expires_at"),
      texts=texts,
  )


def _recurring_from_text(texts: list[TextBlock]) -> Optional[Recurring]:
  for t in texts:
    hit = find_recurring(t.text)
    if hit:
      return Recurring(
          source="text",
          evidence=hit.evidence,
          amount=hit.amount,
          interval=hit.interval,
          trial=hit.trial,
          cancellation_described=hit.mentions_cancel,
      )
  return None


def _message_texts(obj: Json) -> list[TextBlock]:
  return [
      TextBlock(
          f"$.messages[{i}]", msg["content"], _format(msg.get("content_type"))
      )
      for i, msg in enumerate(obj.get("messages") or [])
      if isinstance(msg.get("content"), str)
  ]


def _desc_texts(d: Any, where: str) -> list[TextBlock]:
  if isinstance(d, str):
    return [TextBlock(where, d)]
  if not isinstance(d, Mapping):
    return []
  formats: tuple[Literal["plain", "markdown", "html"], ...] = (
      "plain",
      "markdown",
      "html",
  )
  return [
      TextBlock(f"{where}.{fmt}", d[fmt], fmt)
      for fmt in formats
      if isinstance(d.get(fmt), str)
  ]


def _applies_to(policy: Json, index: int, line_id: Any) -> bool:
  targets = policy.get("applies_to") or []
  if not targets:
    return True  # No target means the whole response.
  return any(
      t in (f"$.line_items[{index}]", "$.line_items[*]", "$.line_items")
      or (line_id is not None and f"'{line_id}'" in t)
      for t in targets
  )


def _ucp_ship_to(checkout: Json) -> Optional[str]:
  for method in (checkout.get("fulfillment") or {}).get("methods") or []:
    dests = method.get("destinations") or []
    chosen = [
        d
        for d in dests
        if d.get("id") is not None
        and d.get("id") == method.get("selected_destination_id")
    ]
    for d in chosen or dests:
      zip_code = d.get("postal_code") or (d.get("address") or {}).get(
          "postal_code"
      )
      if zip_code:
        return str(zip_code)
  return None


def _fees(totals: list[Json]) -> list[Fee]:
  return [
      Fee(label=t.get("display_text") or "Fee", amount=int(t["amount"]))
      for t in totals
      if t.get("type") == "fee"
  ]


def _amount_of(totals: Any, kind: str) -> Optional[int]:
  hits = [t for t in totals or [] if t.get("type") == kind]
  return sum(int(t["amount"]) for t in hits) if hits else None


def _magnitude(totals: Any, kind: str) -> Optional[int]:
  # ACP states discounts as positive amounts and UCP as negative ones.
  hits = [t for t in totals or [] if t.get("type") == kind]
  return sum(abs(int(t["amount"])) for t in hits) if hits else None


def _format(content_type: Any) -> Literal["plain", "markdown"]:
  return "markdown" if content_type == "markdown" else "plain"


def _first(*values: Any) -> Any:
  return next((v for v in values if v is not None), None)
