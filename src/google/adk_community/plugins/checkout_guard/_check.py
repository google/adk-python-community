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

"""Checks a checkout against what the shopper approved."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
from typing import Any
from typing import Iterable
from typing import Optional

from ._models import Checkout
from ._models import Decision
from ._models import Finding
from ._models import ShopperMandate
from ._models import Verdict
from ._rules import interval_word
from ._rules import Rule
from ._rules import RuleContext
from ._rules import RULES
from ._text import format_money


def check(
    mandate: ShopperMandate,
    checkout: Checkout,
    *,
    now: Optional[datetime.datetime] = None,
    skip_rules: Iterable[str] = (),
    extra_rules: Iterable[Rule] = (),
) -> Verdict:
  """Checks a checkout against the shopper's mandate.

  Args:
    mandate: What the shopper approved.
    checkout: The checkout as the merchant states it now.
    now: The time to check expiry against. Defaults to the current UTC time.
    skip_rules: Rule ids to leave out.
    extra_rules: Rules to run after the built-in ones.

  Returns:
    A verdict. ``block`` means do not pay, ``confirm`` means pay only after
    the shopper says yes to ``summary``, ``allow`` means nothing was found.
  """
  now = now or datetime.datetime.now(datetime.timezone.utc)
  skip = set(skip_rules)
  ctx = RuleContext(mandate=mandate, checkout=checkout, now=now)
  findings: list[Finding] = []
  for rule in (*RULES, *extra_rules):
    if rule.id not in skip:
      findings.extend(rule.check(ctx))

  decision: Decision = "allow"
  if any(f.severity == "block" for f in findings):
    decision = "block"
  elif any(f.severity == "confirm" for f in findings):
    decision = "confirm"

  return Verdict(
      decision=decision,
      findings=findings,
      summary=_summarize(checkout, findings, decision),
      digest=digest(checkout),
  )


def digest(checkout: Checkout) -> str:
  """SHA-256 over a canonical, sorted-key JSON encoding of the checkout."""
  canonical = json.dumps(
      _drop_none(dataclasses.asdict(checkout)),
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=False,
  )
  return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_unchanged(
    verdict_or_digest: Verdict | str, checkout: Checkout
) -> bool:
  """Returns False if the checkout changed in any way since it was checked."""
  want = (
      verdict_or_digest
      if isinstance(verdict_or_digest, str)
      else verdict_or_digest.digest
  )
  return digest(checkout) == want


def _drop_none(value: Any) -> Any:
  if isinstance(value, dict):
    return {k: _drop_none(v) for k, v in value.items() if v is not None}
  if isinstance(value, list):
    return [_drop_none(v) for v in value]
  return value


def _summarize(c: Checkout, findings: list[Finding], decision: Decision) -> str:
  """Writes what the shopper is agreeing to, in plain words."""

  def m(n: int) -> str:
    return format_money(n, c.currency)

  head = {
      "block": "Do not pay yet.",
      "confirm": "Check this before paying.",
      "allow": "Nothing unusual found.",
  }[decision]
  by = f" by {c.merchant}" if c.merchant else ""
  lines = [f"{head} You will be charged {m(c.total)}{by} today."]

  for line in c.lines:
    if line.total is not None:
      price = m(line.total)
    elif line.unit_amount is not None:
      price = m(line.unit_amount * line.quantity)
    else:
      price = "price not stated"
    lines.append(f"• {line.quantity} × {line.title}: {price}")
  for fee in c.fees:
    lines.append(f"• {fee.label}: {m(fee.amount)}")
  if c.shipping:
    lines.append(f"• Shipping: {m(c.shipping)}")
  if c.tax:
    lines.append(f"• Tax: {m(c.tax)}")
  if c.tip:
    lines.append(f"• Tip: {m(c.tip)}")
  if c.gift_wrap:
    lines.append(f"• Gift wrap: {m(c.gift_wrap)}")

  recurring = [line for line in c.lines if line.recurring]
  for line in recurring:
    r = line.recurring
    assert r is not None
    amount = m(r.amount) if r.amount is not None else "an unstated amount"
    when = f" {interval_word(r.interval)}" if r.interval else ""
    lines.append(
        f'After today, "{line.title}" keeps charging {amount}{when} until you'
        " cancel."
    )
  if not recurring:
    lines.append("This is a one-time charge. Nothing in the checkout renews.")

  problems = [f for f in findings if f.severity != "note"]
  if problems:
    lines.append("")
    lines.append(
        "One problem:" if len(problems) == 1 else f"{len(problems)} problems:"
    )
    lines.extend(f"• {f.detail}" for f in problems)
  return "\n".join(lines)
