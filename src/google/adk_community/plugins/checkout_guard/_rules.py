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

"""The checkout rules.

Rule ids match the Fine Print project (https://github.com/frostyhand/fine-print),
which this module ports to Python.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import Callable
from typing import Optional

from ._models import Checkout
from ._models import Finding
from ._models import Line
from ._models import ShopperMandate
from ._text import find_agent_directed
from ._text import find_markup
from ._text import format_money
from ._text import has_hidden_characters
from ._text import is_vague_fee_label


@dataclasses.dataclass(frozen=True)
class RuleContext:
  mandate: ShopperMandate
  checkout: Checkout
  now: datetime.datetime

  def money(self, minor: int) -> str:
    return format_money(minor, self.checkout.currency)


@dataclasses.dataclass(frozen=True)
class Rule:
  id: str
  title: str
  why: str
  check: Callable[[RuleContext], list[Finding]]


_INTERVAL_WORDS = {
    "P1M": "every month",
    "P1Y": "every year",
    "P1W": "every week",
    "P3M": "every 3 months",
    "P30D": "every 30 days",
    "P2W": "every 2 weeks",
}


def interval_word(interval: str) -> str:
  return _INTERVAL_WORDS.get(interval, f"every {interval}")


def _line_matches(line: Line, match: str) -> bool:
  m = match.lower()
  return any(v and m in v.lower() for v in (line.title, line.sku, line.id))


def _describe_recurring(ctx: RuleContext, line: Line) -> str:
  r = line.recurring
  assert r is not None
  parts = []
  if r.amount is not None:
    parts.append(ctx.money(r.amount))
  if r.interval:
    parts.append(interval_word(r.interval))
  if r.starts_at:
    parts.append(f"starting {r.starts_at[:10]}")
  if not parts:
    return "an amount and schedule the checkout does not state"
  return " ".join(parts)


def _parse_time(value: str) -> Optional[datetime.datetime]:
  try:
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
  except ValueError:
    return None
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=datetime.timezone.utc)
  return parsed


def _has_passed(value: Optional[str], now: datetime.datetime) -> bool:
  if not value:
    return False
  parsed = _parse_time(value)
  # A time that cannot be read fails closed.
  return parsed is None or parsed <= now


# Money ----------------------------------------------------------------------


def _over_budget(ctx: RuleContext) -> list[Finding]:
  c, m = ctx.checkout, ctx.mandate
  if c.total <= m.max_total:
    return []
  return [
      Finding(
          rule="FP101",
          severity="block",
          title="Over budget",
          detail=(
              f"The total is {ctx.money(c.total)}, which is over the"
              f" {ctx.money(m.max_total)} limit you set."
          ),
      )
  ]


def _total_changed(ctx: RuleContext) -> list[Finding]:
  c, m = ctx.checkout, ctx.mandate
  if m.approved_total is None:
    return []
  drift = c.total - m.approved_total
  if abs(drift) <= m.tolerance:
    return []
  return [
      Finding(
          rule="FP102",
          severity="confirm",
          title="Total changed after approval",
          detail=(
              f"You approved {ctx.money(m.approved_total)}. The checkout now"
              f" says {ctx.money(c.total)} ({'up' if drift > 0 else 'down'}"
              f" {ctx.money(abs(drift))})."
          ),
      )
  ]


def _not_final(ctx: RuleContext) -> list[Finding]:
  c = ctx.checkout
  if c.price_final:
    return []
  return [
      Finding(
          rule="FP103",
          severity="confirm",
          title="Price not final yet",
          detail=(
              f'The merchant says this checkout is "{c.status or "not ready"}",'
              f" so the total of {ctx.money(c.total)} may still change. Check"
              " again once it is ready for payment."
          ),
      )
  ]


def _wrong_currency(ctx: RuleContext) -> list[Finding]:
  want = ctx.mandate.currency.upper()
  got = ctx.checkout.currency.upper()
  if want == got:
    return []
  return [
      Finding(
          rule="FP104",
          severity="block",
          title="Wrong currency",
          detail=(
              f"You approved a charge in {want}, but the checkout is in {got}."
          ),
      )
  ]


def _totals_do_not_add_up(ctx: RuleContext) -> list[Finding]:
  c = ctx.checkout
  if c.subtotal is None:
    return []
  expected = (
      c.subtotal
      + sum(f.amount for f in c.fees)
      + (c.shipping or 0)
      + (c.tax or 0)
      + (c.tip or 0)
      + (c.gift_wrap or 0)
      - abs(c.discounts or 0)
      - abs(c.store_credit or 0)
  )
  # The ACP examples disagree on whether the subtotal already has line-item
  # discounts taken off, so accept both readings.
  with_item_discounts = expected - abs(c.item_discounts or 0)
  if c.total in (expected, with_item_discounts):
    return []
  return [
      Finding(
          rule="FP105",
          severity="block",
          title="Totals do not add up",
          detail=(
              f"The listed parts add up to {ctx.money(expected)}, but the"
              f" checkout total is {ctx.money(c.total)}. The difference of"
              f" {ctx.money(abs(c.total - expected))} is not explained."
          ),
      )
  ]


# Recurring charges ----------------------------------------------------------


def _unrequested_subscription(ctx: RuleContext) -> list[Finding]:
  if ctx.mandate.allow_recurring:
    return []
  out = []
  for line in ctx.checkout.lines:
    if not line.recurring or line.recurring.source != "structured":
      continue
    fix = None
    if line.one_time_option_id:
      fix = {
          "line_item_id": line.id,
          "selected_purchase_option_id": line.one_time_option_id,
      }
    out.append(
        Finding(
            rule="FP201",
            severity="block",
            title="Subscription you did not ask for",
            line_id=line.id,
            detail=(
                f'"{line.title}" is set up as a recurring purchase:'
                f" {_describe_recurring(ctx, line)}. You asked for a one-time"
                " purchase."
            ),
            fix=fix,
        )
    )
  return out


def _renewal_in_text(ctx: RuleContext) -> list[Finding]:
  if ctx.mandate.allow_recurring:
    return []
  return [
      Finding(
          rule="FP202",
          severity="block",
          title="Recurring charge hidden in the fine print",
          line_id=line.id,
          detail=(
              f'The checkout data does not mark "{line.title}" as recurring,'
              f' but its terms say: "{line.recurring.evidence}". You asked for'
              " a one-time purchase."
          ),
      )
      for line in ctx.checkout.lines
      if line.recurring and line.recurring.source == "text"
  ]


def _subscription_by_default(ctx: RuleContext) -> list[Finding]:
  if ctx.mandate.allow_recurring:
    return []
  return [
      Finding(
          rule="FP203",
          severity="block",
          title="Merchant picked the subscription option for you",
          line_id=line.id,
          detail=(
              f'"{line.title}" can be bought once, but the checkout selected a'
              " subscription. Switch the line to the one-time option"
              f' "{line.one_time_option_id}".'
          ),
          fix={
              "line_item_id": line.id,
              "selected_purchase_option_id": line.one_time_option_id,
          },
      )
      for line in ctx.checkout.lines
      if line.recurring and line.one_time_option_id
  ]


def _trial_converts(ctx: RuleContext) -> list[Finding]:
  return [
      Finding(
          rule="FP204",
          severity="confirm",
          title="Free trial turns into a paid plan",
          line_id=line.id,
          detail=(
              f'"{line.title}" starts as a trial and then charges'
              f" {_describe_recurring(ctx, line)} unless you cancel."
          ),
      )
      for line in ctx.checkout.lines
      if line.recurring and line.recurring.trial
  ]


def _no_cancellation_terms(ctx: RuleContext) -> list[Finding]:
  return [
      Finding(
          rule="FP205",
          severity="confirm",
          title="No way to cancel is described",
          line_id=line.id,
          detail=(
              f'"{line.title}" renews, but the checkout gives no cancellation'
              " terms or link."
          ),
      )
      for line in ctx.checkout.lines
      if line.recurring and not line.recurring.cancellation_described
  ]


# What is in the cart --------------------------------------------------------


def _unrequested_item(ctx: RuleContext) -> list[Finding]:
  items = ctx.mandate.items
  if not items:
    return []
  out = []
  for line in ctx.checkout.lines:
    if any(_line_matches(line, i.match) for i in items):
      continue
    price = f" ({ctx.money(line.total)})" if line.total is not None else ""
    out.append(
        Finding(
            rule="FP301",
            severity="block",
            title="Item you did not ask for",
            line_id=line.id,
            detail=(
                f'The cart includes "{line.title}"{price}, which is not one of'
                " the items you asked for. Remove it before paying."
            ),
            fix={"remove_line_item_id": line.id},
        )
    )
  return out


def _too_many(ctx: RuleContext) -> list[Finding]:
  out = []
  for item in ctx.mandate.items:
    if item.max_quantity is None:
      continue
    qty = sum(
        line.quantity
        for line in ctx.checkout.lines
        if _line_matches(line, item.match)
    )
    if qty > item.max_quantity:
      out.append(
          Finding(
              rule="FP302",
              severity="block",
              title="More than you asked for",
              detail=(
                  f'The cart has {qty} of "{item.match}". You asked for at'
                  f" most {item.max_quantity}."
              ),
          )
      )
  return out


def _tip_or_gift_wrap(ctx: RuleContext) -> list[Finding]:
  c = ctx.checkout
  out = []
  if c.tip is not None and c.tip > 0:
    out.append(
        Finding(
            rule="FP303",
            severity="confirm",
            title="Tip or gift wrap added",
            detail=(
                f"A tip of {ctx.money(c.tip)} is included. Keep it only if you"
                " meant to tip."
            ),
        )
    )
  if c.gift_wrap is not None and c.gift_wrap > 0:
    out.append(
        Finding(
            rule="FP303",
            severity="confirm",
            title="Tip or gift wrap added",
            detail=f"Gift wrap for {ctx.money(c.gift_wrap)} is included.",
        )
    )
  return out


# Fees -----------------------------------------------------------------------


def _drip_pricing(ctx: RuleContext) -> list[Finding]:
  c, m = ctx.checkout, ctx.mandate
  if m.advertised_price is None:
    return []
  # Government taxes and shipping are not part of the advertised price.
  all_in = c.total - (c.tax or 0) - (c.shipping or 0)
  if all_in <= m.advertised_price + m.tolerance:
    return []
  fees = ", ".join(f"{f.label} {ctx.money(f.amount)}" for f in c.fees)
  strict = m.category in ("live_event_ticket", "short_term_lodging")
  detail = (
      f"The listing showed {ctx.money(m.advertised_price)}, but before tax and"
      f" shipping the checkout charges {ctx.money(all_in)}"
      f"{f' (fees: {fees})' if fees else ''}."
  )
  if strict:
    detail += (
        " For tickets and short-term lodging, the total price has to be shown"
        " up front."
    )
  return [
      Finding(
          rule="FP401",
          severity="block" if strict else "confirm",
          title="Price went up at checkout",
          detail=detail,
      )
  ]


def _vague_fee(ctx: RuleContext) -> list[Finding]:
  return [
      Finding(
          rule="FP402",
          severity="note",
          title="Fee with a vague name",
          detail=(
              f'"{f.label}" ({ctx.money(f.amount)}) does not say what it is'
              " for."
          ),
      )
      for f in ctx.checkout.fees
      if f.amount > 0 and is_vague_fee_label(f.label)
  ]


# Security -------------------------------------------------------------------


def _merchant_not_allowed(ctx: RuleContext) -> list[Finding]:
  allowed = ctx.mandate.allowed_merchants
  if not allowed:
    return []
  got = (ctx.checkout.merchant or "").lower()
  if got and any(a.lower() == got for a in allowed):
    return []
  who = f'"{got}"' if got else "an unidentified merchant"
  return [
      Finding(
          rule="FP501",
          severity="block",
          title="Merchant not on your list",
          detail=(
              f"This checkout is from {who}, which is not one you approved"
              f" ({', '.join(allowed)})."
          ),
      )
  ]


def _ship_to_changed(ctx: RuleContext) -> list[Finding]:
  want = (ctx.mandate.ship_to_postal_code or "").strip()
  got = (ctx.checkout.ship_to_postal_code or "").strip()
  if not want or not got or want[:5] == got[:5]:
    return []
  return [
      Finding(
          rule="FP502",
          severity="block",
          title="Shipping address changed",
          detail=(
              f"You approved shipping to ZIP {want}, but this checkout ships to"
              f" ZIP {got}."
          ),
      )
  ]


def _approval_expired(ctx: RuleContext) -> list[Finding]:
  exp = ctx.mandate.expires_at
  if not _has_passed(exp, ctx.now):
    return []
  return [
      Finding(
          rule="FP503",
          severity="block",
          title="Approval has expired",
          detail=(
              f"Your approval for this purchase ran out at {exp}. Ask again"
              " before paying."
          ),
      )
  ]


def _session_expired(ctx: RuleContext) -> list[Finding]:
  exp = ctx.checkout.expires_at
  if not _has_passed(exp, ctx.now):
    return []
  return [
      Finding(
          rule="FP504",
          severity="block",
          title="Checkout session has expired",
          detail=(
              f"The merchant's checkout session expired at {exp}. Start a new"
              " one."
          ),
      )
  ]


def _agent_directed_text(ctx: RuleContext) -> list[Finding]:
  out = []
  for t in ctx.checkout.texts:
    hit = find_agent_directed(t.text)
    if hit:
      out.append(
          Finding(
              rule="FP505",
              severity="confirm",
              title="Checkout text gives orders to the AI",
              where=t.where,
              detail=(
                  f"Text at {t.where} speaks to the AI rather than to you:"
                  f' "{hit}". Ignore it, and check the cart yourself before'
                  " paying."
              ),
          )
      )
  return out


def _hidden_text(ctx: RuleContext) -> list[Finding]:
  out = []
  for t in ctx.checkout.texts:
    markup = find_markup(t.text) if t.format != "html" else None
    if markup:
      out.append(
          Finding(
              rule="FP506",
              severity="confirm",
              title="Hidden characters or raw markup in terms",
              where=t.where,
              detail=(
                  f'Text at {t.where} contains raw markup ("{markup}"). ACP'
                  " and UCP both say to treat it as untrusted."
              ),
          )
      )
    if has_hidden_characters(t.text):
      out.append(
          Finding(
              rule="FP506",
              severity="confirm",
              title="Hidden characters or raw markup in terms",
              where=t.where,
              detail=(
                  f"Text at {t.where} contains invisible Unicode characters,"
                  " which can hide words from you while the AI still reads"
                  " them."
              ),
          )
      )
  return out


RULES: tuple[Rule, ...] = (
    Rule(
        "FP101",
        "Over budget",
        "The charge is higher than the ceiling the shopper set.",
        _over_budget,
    ),
    Rule(
        "FP102",
        "Total changed after approval",
        "The shopper approved one price and the checkout now shows another.",
        _total_changed,
    ),
    Rule(
        "FP103",
        "Price not final yet",
        "The merchant has not marked the session ready for payment.",
        _not_final,
    ),
    Rule(
        "FP104",
        "Wrong currency",
        "A charge in another currency can cost more after conversion.",
        _wrong_currency,
    ),
    Rule(
        "FP105",
        "Totals do not add up",
        "The shopper cannot tell what they are paying for.",
        _totals_do_not_add_up,
    ),
    Rule(
        "FP201",
        "Subscription you did not ask for",
        "Agents are asked to buy things, not to start recurring charges.",
        _unrequested_subscription,
    ),
    Rule(
        "FP202",
        "Recurring charge hidden in the fine print",
        "The fields say one-time, but the disclosure text describes a renewal.",
        _renewal_in_text,
    ),
    Rule(
        "FP203",
        "Merchant picked the subscription option for you",
        "In UCP the merchant chooses the default purchase option.",
        _subscription_by_default,
    ),
    Rule(
        "FP204",
        "Free trial turns into a paid plan",
        "Trials that convert are how people end up paying for things.",
        _trial_converts,
    ),
    Rule(
        "FP205",
        "No way to cancel is described",
        "The shopper may not be able to stop the charges.",
        _no_cancellation_terms,
    ),
    Rule(
        "FP301",
        "Item you did not ask for",
        "Pre-selected add-ons ride along with the item the shopper wanted.",
        _unrequested_item,
    ),
    Rule(
        "FP302",
        "More than you asked for",
        "A changed quantity multiplies the charge.",
        _too_many,
    ),
    Rule(
        "FP303",
        "Tip or gift wrap added",
        "Optional extras should be the shopper's choice.",
        _tip_or_gift_wrap,
    ),
    Rule(
        "FP401",
        "Price went up at checkout",
        "Mandatory fees appear only at the end (drip pricing).",
        _drip_pricing,
    ),
    Rule(
        "FP402",
        "Fee with a vague name",
        "The label does not say what the shopper is paying for.",
        _vague_fee,
    ),
    Rule(
        "FP501",
        "Merchant not on your list",
        "A checkout can be swapped for one at an unexpected seller.",
        _merchant_not_allowed,
    ),
    Rule(
        "FP502",
        "Shipping address changed",
        "A swapped address redirects the goods.",
        _ship_to_changed,
    ),
    Rule(
        "FP503",
        "Approval has expired",
        "An old approval should not be reused for a new purchase.",
        _approval_expired,
    ),
    Rule(
        "FP504",
        "Checkout session has expired",
        "Paying on an expired session can fail or be re-priced.",
        _session_expired,
    ),
    Rule(
        "FP505",
        "Checkout text gives orders to the AI",
        "Checkout copy has no reason to address the agent.",
        _agent_directed_text,
    ),
    Rule(
        "FP506",
        "Hidden characters or raw markup in terms",
        "The shopper and the agent can be shown different text.",
        _hidden_text,
    ),
)
