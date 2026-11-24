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

"""Data types for the checkout guard.

All money is an integer in ISO 4217 minor units (cents for USD), matching
both the Agentic Commerce Protocol (``amount``) and the Universal Commerce
Protocol (``amount.json``).
"""

from __future__ import annotations

import dataclasses
from typing import Any
from typing import Literal
from typing import Mapping
from typing import Optional

Category = Literal["general", "live_event_ticket", "short_term_lodging"]
Severity = Literal["block", "confirm", "note"]
Decision = Literal["allow", "confirm", "block"]


@dataclasses.dataclass(frozen=True)
class MandateItem:
  """One thing the shopper asked for.

  A cart line matches when its title, SKU or id contains ``match``
  (case-insensitive).
  """

  match: str
  max_quantity: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class ShopperMandate:
  """What the shopper approved before the agent went shopping.

  Attributes:
    max_total: Hard ceiling for the all-in charge (items, fees, tax,
      shipping), in minor units.
    currency: ISO 4217 code, for example ``"USD"``.
    approved_total: The total the shopper saw and approved, if they approved
      a specific cart.
    tolerance: Allowed drift from ``approved_total`` and ``advertised_price``,
      in minor units.
    items: What the shopper asked for. Empty skips the item checks.
    allow_recurring: The shopper explicitly asked for a subscription or
      membership.
    allowed_merchants: Merchant identifiers (domain or id) the shopper
      allowed. Empty allows any merchant.
    ship_to_postal_code: Ship-to postal code the shopper approved.
    expires_at: RFC 3339 time after which the approval is no longer valid.
    advertised_price: Price the agent saw on the listing, before checkout.
    category: What is being bought. Fee rules are stricter for live-event
      tickets and short-term lodging.
  """

  max_total: int
  currency: str
  approved_total: Optional[int] = None
  tolerance: int = 0
  items: tuple[MandateItem, ...] = ()
  allow_recurring: bool = False
  allowed_merchants: tuple[str, ...] = ()
  ship_to_postal_code: Optional[str] = None
  expires_at: Optional[str] = None
  advertised_price: Optional[int] = None
  category: Category = "general"

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> ShopperMandate:
    """Builds a mandate from a JSON-style dict, such as one kept in state."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
      raise ValueError(f"Unknown mandate fields: {sorted(unknown)}")
    values = dict(data)
    values["items"] = tuple(
        i if isinstance(i, MandateItem) else MandateItem(**i)
        for i in values.get("items") or ()
    )
    values["allowed_merchants"] = tuple(values.get("allowed_merchants") or ())
    return cls(**values)

  def to_dict(self) -> dict[str, Any]:
    """Returns a JSON-safe dict, for example to keep in session state."""
    data = dataclasses.asdict(self)
    data["items"] = [dataclasses.asdict(i) for i in self.items]
    data["allowed_merchants"] = list(self.allowed_merchants)
    return data


@dataclasses.dataclass
class Recurring:
  """A charge that repeats after today.

  Attributes:
    source: ``"structured"`` when the protocol said so in a field, ``"text"``
      when it was found only in prose.
    interval: ISO 8601 duration (``P1M``) when known.
    amount: Amount of each renewal, when known.
    starts_at: RFC 3339 time of the first renewal, when known.
    trial: The first period is free and then converts to a paid plan.
    cancellation_described: The checkout told the buyer how to cancel.
    cancellation_url: Link to the cancellation terms, when given.
    evidence: The text that revealed the recurring charge.
  """

  source: Literal["structured", "text"]
  interval: Optional[str] = None
  amount: Optional[int] = None
  starts_at: Optional[str] = None
  trial: bool = False
  cancellation_described: bool = False
  cancellation_url: Optional[str] = None
  evidence: Optional[str] = None


@dataclasses.dataclass
class Line:
  """One line of the cart."""

  id: str
  title: str
  quantity: int
  unit_amount: Optional[int] = None
  total: Optional[int] = None
  sku: Optional[str] = None
  parent_id: Optional[str] = None
  recurring: Optional[Recurring] = None
  one_time_option_id: Optional[str] = None
  """UCP only: a one-time purchase option the line could switch to."""
  selected_option_id: Optional[str] = None
  """UCP only: the selected purchase option id."""


@dataclasses.dataclass
class Fee:
  label: str
  amount: int


@dataclasses.dataclass
class TextBlock:
  """Buyer-facing text and where in the checkout it came from."""

  where: str
  text: str
  format: Literal["plain", "markdown", "html"] = "plain"


@dataclasses.dataclass
class Checkout:
  """A protocol-neutral view of a checkout, built by an adapter."""

  protocol: Literal["acp", "ucp", "normalized"]
  currency: str
  price_final: bool
  total: int
  merchant: Optional[str] = None
  status: Optional[str] = None
  lines: list[Line] = dataclasses.field(default_factory=list)
  fees: list[Fee] = dataclasses.field(default_factory=list)
  subtotal: Optional[int] = None
  discounts: Optional[int] = None
  """Order-level discounts, as a positive number."""
  item_discounts: Optional[int] = None
  """Line-item discounts, as a positive number."""
  shipping: Optional[int] = None
  tax: Optional[int] = None
  tip: Optional[int] = None
  gift_wrap: Optional[int] = None
  store_credit: Optional[int] = None
  ship_to_postal_code: Optional[str] = None
  expires_at: Optional[str] = None
  texts: list[TextBlock] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Finding:
  """One problem found in the checkout."""

  rule: str
  severity: Severity
  title: str
  detail: str
  """Plain-language explanation the shopper can read before confirming."""
  line_id: Optional[str] = None
  where: Optional[str] = None
  fix: Optional[dict[str, Any]] = None
  """A concrete, safe change the agent can make instead, when one exists."""

  def to_dict(self) -> dict[str, Any]:
    return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


@dataclasses.dataclass
class Verdict:
  """The result of checking a checkout against a mandate.

  ``block`` means do not pay. ``confirm`` means show ``summary`` to the
  shopper and pay only on an explicit yes. ``allow`` means nothing was found.
  """

  decision: Decision
  findings: list[Finding]
  summary: str
  digest: str
  """SHA-256 of the canonical checkout, used to detect later changes."""

  def to_dict(self) -> dict[str, Any]:
    return {
        "decision": self.decision,
        "summary": self.summary,
        "digest": self.digest,
        "findings": [f.to_dict() for f in self.findings],
    }
