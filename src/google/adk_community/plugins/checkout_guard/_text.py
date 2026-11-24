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

"""Deterministic text checks.

No model is involved, so nothing a merchant writes in the checkout can argue
its way past them.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Optional

_I = re.IGNORECASE

_STRONG_RECURRING = [
    re.compile(p, _I)
    for p in (
        r"\bauto(?:matic(?:ally)?)?[\s-]?renew(?:s|al|ed|ing)?\b",
        r"\brenews?\s+(?:automatically|monthly|annually|yearly|weekly|every)\b",
        r"\brecurring\s+(?:charge|payment|billing|fee|order|delivery)s?\b",
        r"\bbilled\s+(?:monthly|annually|yearly|weekly|quarterly|every)\b",
        r"\b(?:membership|subscription)\s+(?:fee|charge|plan|will)\b",
        r"\bfree\s+trial\b",
        r"\b\d+[\s-]day\s+trial\b",
        r"\buntil\s+(?:you\s+)?cancel(?:led|ed)?\b",
        r"\bcontinuous\s+service\b",
        r"\bsubscribe\s*(?:&|and)\s*save\b",
    )
]

_WEAK_RECURRING = [
    re.compile(p, _I)
    for p in (
        r"\bsubscri(?:be|ption)\b",
        r"\bmembership\b",
        r"\bper\s+(?:month|year|week)\b",
        r"/\s?(?:mo|month|yr|year|wk)\b",
        r"\bevery\s+(?:\d+\s+)?(?:days?|weeks?|months?|years?)\b",
    )
]

_MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)")
_TRIAL = re.compile(
    r"\b(?:free\s+trial|\d+[\s-]day\s+trial|trial\s+(?:period|ends)"
    r"|first\s+(?:month|week)\s+(?:free|for\s+\$))",
    _I,
)
_CANCEL = re.compile(r"\bcancel(?:l?ation|led|ed|ing)?\b", _I)
_INTERVAL = [
    (
        re.compile(
            r"\b(?:monthly|per\s+month|/\s?mo(?:nth)?\b|every\s+month)", _I
        ),
        "P1M",
    ),
    (
        re.compile(
            r"\b(?:annual(?:ly)?|yearly|per\s+year|/\s?yr\b|/\s?year\b"
            r"|every\s+year)",
            _I,
        ),
        "P1Y",
    ),
    (re.compile(r"\b(?:weekly|per\s+week|/\s?wk\b|every\s+week)", _I), "P1W"),
    (re.compile(r"\bquarterly\b", _I), "P3M"),
]


@dataclasses.dataclass
class RecurringHit:
  evidence: str
  amount: Optional[int]
  interval: Optional[str]
  trial: bool
  mentions_cancel: bool


def find_recurring(text: str) -> Optional[RecurringHit]:
  """Finds a recurring charge described in prose."""
  strong = next(
      (r.search(text) for r in _STRONG_RECURRING if r.search(text)), None
  )
  weak = next((r.search(text) for r in _WEAK_RECURRING if r.search(text)), None)
  money = _MONEY.search(text)
  # A weak cue ("subscribe") alone is too noisy: newsletters say it too.
  # Require an amount next to it.
  if not strong and not (weak and money):
    return None
  cue = strong or weak
  assert cue is not None
  return RecurringHit(
      evidence=_snippet(text, cue.start(), len(cue.group(0))),
      amount=to_minor(money.group(1)) if money else None,
      interval=next((i for r, i in _INTERVAL if r.search(text)), None),
      trial=bool(_TRIAL.search(text)),
      mentions_cancel=bool(_CANCEL.search(text)),
  )


# Legitimate checkout copy has no reason to address the agent, so any hit is
# treated as a prompt injection attempt.
_AGENT_DIRECTED = [
    re.compile(p, _I)
    for p in (
        r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier|your)"
        r"\s+(?:instructions|rules|guidelines|constraints)\b",
        r"\b(?:ai|llm)\s+(?:assistant|agent|shopper|model)s?\s*[,:]",
        r"\b(?:attention|note|message)\s+(?:to|for)\s+(?:the\s+)?"
        r"(?:ai|llm|agent|assistant|model)s?\b",
        r"\bif\s+you\s+are\s+an?\s+(?:ai|llm|agent|assistant|language\s+model)\b",
        r"\b(?:do\s+not|don't|never)\s+(?:tell|show|mention|inform|ask|notify)"
        r"\s+(?:the\s+)?(?:user|buyer|customer|shopper|human)\b",
        r"\bwithout\s+(?:asking|confirming\s+with|telling|notifying)\s+"
        r"(?:the\s+)?(?:user|buyer|customer|shopper|human)\b",
        r"\b(?:the\s+)?(?:user|buyer|customer|shopper)\s+has\s+(?:already\s+)?"
        r"(?:approved|agreed|consented|authorized)\b",
        r"\b(?:automatically|always)\s+(?:accept|approve|confirm|select|add"
        r"|agree)\b",
        r"\bauto[\s-]?(?:accept|approve|confirm|agree)\b",
        r"\bsystem\s*(?:prompt|message|override)\b",
    )
]


def find_agent_directed(text: str) -> Optional[str]:
  """Returns a snippet of text that gives orders to the AI, if any."""
  for r in _AGENT_DIRECTED:
    m = r.search(text)
    if m:
      return _snippet(text, m.start(), len(m.group(0)))
  return None


_RAW_HTML = re.compile(
    r"<\s*(?:script|iframe|img|a|style|form|input|object|embed|svg|link"
    r"|meta)\b|\bon(?:click|load|error|mouseover)\s*=|javascript:",
    _I,
)
_HIDDEN_TEXT = re.compile("[​-‏⁠-⁤﻿‪-‮\U000e0000-\U000e007f]")


def find_markup(text: str) -> Optional[str]:
  m = _RAW_HTML.search(text)
  return _snippet(text, m.start(), len(m.group(0))) if m else None


def has_hidden_characters(text: str) -> bool:
  return bool(_HIDDEN_TEXT.search(text))


_VAGUE_FEE = re.compile(
    r"^\s*(?:convenience|service|processing|handling|platform|booking|order"
    r"|admin(?:istrative)?|resort|facility|cleaning|misc(?:ellaneous)?)?\s*"
    r"(?:fee|charge|surcharge)s?\s*$",
    _I,
)
_NAMED_VAGUE_FEE = re.compile(
    r"\b(?:convenience|service|processing)\s+fee\b", _I
)


def is_vague_fee_label(label: str) -> bool:
  """Fee labels that do not say what the fee is for."""
  return bool(_VAGUE_FEE.search(label) or _NAMED_VAGUE_FEE.search(label))


def to_minor(s: str) -> int:
  return round(float(s.replace(",", "")) * 100)


_SYMBOLS = {"USD": "$", "CAD": "CA$", "AUD": "A$", "EUR": "€", "GBP": "£"}


def format_money(minor: int, currency: str = "USD") -> str:
  code = currency.upper()
  sign = "-" if minor < 0 else ""
  amount = f"{abs(minor) / 100:,.2f}"
  symbol = _SYMBOLS.get(code)
  return f"{sign}{symbol}{amount}" if symbol else f"{sign}{amount} {code}"


def _snippet(text: str, at: int, length: int) -> str:
  start = max(0, at - 40)
  end = min(len(text), at + length + 60)
  body = re.sub(r"\s+", " ", text[start:end]).strip()
  return f"{'…' if start > 0 else ''}{body}{'…' if end < len(text) else ''}"
