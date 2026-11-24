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

"""Tests for the checkout guard rules and the ACP and UCP adapters.

Each fixture in ``checkout_guard_fixtures`` is a checkout a merchant could
really send, with the verdict the guard must return. The ACP fixtures
validate against the published ACP 2026-04-17 schema.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import json
from pathlib import Path
import re

import pytest

from google.adk_community.plugins.checkout_guard import check
from google.adk_community.plugins.checkout_guard import Checkout
from google.adk_community.plugins.checkout_guard import digest
from google.adk_community.plugins.checkout_guard import from_acp
from google.adk_community.plugins.checkout_guard import from_ucp
from google.adk_community.plugins.checkout_guard import Line
from google.adk_community.plugins.checkout_guard import MandateItem
from google.adk_community.plugins.checkout_guard import RULES
from google.adk_community.plugins.checkout_guard import ShopperMandate
from google.adk_community.plugins.checkout_guard import verify_unchanged

NOW = datetime.datetime(2026, 11, 22, 12, 0, tzinfo=datetime.timezone.utc)
FIXTURES = sorted(
    (Path(__file__).parent / "checkout_guard_fixtures").glob("*.json")
)


def _load(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def _adapt(fx: dict) -> Checkout:
  adapter = from_acp if fx["format"] == "acp" else from_ucp
  return adapter(fx["checkout"], merchant=fx["merchant"])


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_fixture_verdict(path: Path):
  fx = _load(path)
  verdict = check(ShopperMandate.from_dict(fx["mandate"]), _adapt(fx), now=NOW)

  assert sorted(f.rule for f in verdict.findings) == sorted(
      fx["expect"]["rules"]
  ), verdict.summary
  assert verdict.decision == fx["expect"]["decision"], verdict.summary
  if "expect_fix" in fx:
    fixes = [
        f.fix
        for f in verdict.findings
        if f.fix and "selected_purchase_option_id" in f.fix
    ]
    assert fixes and fixes[0] == fx["expect_fix"]


@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_summary_is_plain_text_for_the_shopper(path: Path):
  fx = _load(path)
  verdict = check(ShopperMandate.from_dict(fx["mandate"]), _adapt(fx), now=NOW)
  assert verdict.summary.splitlines()[0].endswith("today.")
  assert not re.search(r"—| - ", verdict.summary)


def test_rule_ids_are_unique():
  ids = [r.id for r in RULES]
  assert len(ids) == len(set(ids)) == 21


def _fixture(name: str) -> dict:
  return _load(
      Path(__file__).parent / "checkout_guard_fixtures" / f"{name}.json"
  )


def test_digest_changes_when_the_checkout_changes():
  fx = _fixture("acp-clean")
  first = _adapt(fx)
  changed = copy.deepcopy(fx)
  changed["checkout"]["line_items"][0]["quantity"] = 2
  assert digest(first) == digest(_adapt(fx))
  assert not verify_unchanged(digest(first), _adapt(changed))
  verdict = check(ShopperMandate.from_dict(fx["mandate"]), first, now=NOW)
  assert verify_unchanged(verdict, _adapt(fx))


def _clean_checkout(**overrides) -> Checkout:
  base = Checkout(
      protocol="normalized",
      currency="USD",
      price_final=True,
      total=5000,
      merchant="shop.example",
      lines=[Line(id="l1", title="Burr grinder", quantity=1, total=5000)],
      subtotal=5000,
  )
  return dataclasses.replace(base, **overrides)


def test_clean_checkout_is_allowed():
  verdict = check(
      ShopperMandate(max_total=6000, currency="USD"), _clean_checkout(), now=NOW
  )
  assert verdict.decision == "allow"
  assert verdict.findings == []
  assert "one-time charge" in verdict.summary


@pytest.mark.parametrize(
    "mandate, checkout, rule",
    [
        (
            ShopperMandate(max_total=6000, currency="EUR"),
            _clean_checkout(),
            "FP104",
        ),
        (
            ShopperMandate(
                max_total=6000,
                currency="USD",
                allowed_merchants=("grinders.example",),
            ),
            _clean_checkout(),
            "FP501",
        ),
        (
            ShopperMandate(
                max_total=6000,
                currency="USD",
                expires_at="2026-11-01T00:00:00Z",
            ),
            _clean_checkout(),
            "FP503",
        ),
        (
            ShopperMandate(max_total=6000, currency="USD"),
            _clean_checkout(expires_at="2026-11-22T11:00:00Z"),
            "FP504",
        ),
        (
            ShopperMandate(
                max_total=20000,
                currency="USD",
                items=(MandateItem("grinder", max_quantity=1),),
            ),
            _clean_checkout(
                total=10000,
                subtotal=10000,
                lines=[
                    Line(id="l1", title="Burr grinder", quantity=2, total=10000)
                ],
            ),
            "FP302",
        ),
    ],
    ids=[
        "currency",
        "merchant",
        "approval-expired",
        "session-expired",
        "quantity",
    ],
)
def test_blocking_rules(mandate, checkout, rule):
  verdict = check(mandate, checkout, now=NOW)
  assert verdict.decision == "block"
  assert [f.rule for f in verdict.findings] == [rule]


def test_unreadable_expiry_fails_closed():
  mandate = ShopperMandate(max_total=6000, currency="USD", expires_at="soon")
  verdict = check(mandate, _clean_checkout(), now=NOW)
  assert [f.rule for f in verdict.findings] == ["FP503"]


def test_skip_rules():
  mandate = ShopperMandate(max_total=6000, currency="EUR")
  verdict = check(mandate, _clean_checkout(), now=NOW, skip_rules=["FP104"])
  assert verdict.decision == "allow"


def test_mandate_round_trips_through_a_dict():
  mandate = ShopperMandate(
      max_total=12000,
      currency="USD",
      items=(MandateItem("grinder", max_quantity=1),),
      allowed_merchants=("shop.example",),
  )
  data = mandate.to_dict()
  assert json.loads(json.dumps(data)) == data
  assert ShopperMandate.from_dict(data) == mandate


def test_mandate_rejects_unknown_fields():
  with pytest.raises(ValueError, match="maxTotal"):
    ShopperMandate.from_dict({"maxTotal": 100, "currency": "USD"})
