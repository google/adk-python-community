# Copyright 2025 Google LLC
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

"""Sardis payment tools for Google Agent Development Kit.

These tools enable ADK agents to make policy-controlled payments through
Sardis non-custodial MPC wallets.

Installation:
    pip install google-adk-community sardis

Usage:
    from google.adk.agents import Agent
    from google.adk_community.tools.sardis import (
        sardis_pay, sardis_check_balance, sardis_check_policy,
    )

    agent = Agent(
        model="gemini-2.0-flash",
        name="payment_agent",
        tools=[sardis_pay, sardis_check_balance, sardis_check_policy],
    )
"""

from __future__ import annotations

from decimal import Decimal
from decimal import InvalidOperation
import logging
import os
from typing import Optional

logger = logging.getLogger("google_adk." + __name__)


def _parse_amount(amount: str) -> Decimal:
  try:
    value = Decimal(amount)
  except (InvalidOperation, ValueError) as exc:
    raise ValueError(f"amount must be a decimal string, got {amount!r}") from exc
  if not value.is_finite() or value <= Decimal("0"):
    raise ValueError(f"amount must be a positive finite decimal, got {amount!r}")
  return value


def _sardis_exception_types() -> tuple[type[Exception], ...]:
  try:
    from sardis import AuthenticationError
    from sardis import InsufficientBalanceError
    from sardis import PolicyViolationError
  except (ImportError, AttributeError):
    return ()
  return (AuthenticationError, InsufficientBalanceError, PolicyViolationError)


def _sardis_error_response(exc: Exception) -> dict:
  return {"status": "error", "error": str(exc)}


def sardis_pay(
    recipient: str,
    amount: str,
    token: str = "USDC",
    chain: str = "base",
    memo: Optional[str] = None,
) -> dict:
  """Execute a policy-controlled payment via Sardis.

  Args:
    recipient: Destination wallet address or Sardis wallet ID.
    amount: Amount to send as a decimal string (e.g. "10.00").
    token: Token symbol (USDC, EURC, USDT, PYUSD). Default: USDC.
    chain: Chain to execute on (base, polygon, ethereum, arbitrum,
      optimism). Default: base.
    memo: Optional human-readable memo for the payment.

  Returns:
    A dict with keys: status, transaction_id, tx_hash, amount, token, chain.
  """
  try:
    from sardis import SardisClient
  except ImportError:
    return {
        "status": "error",
        "error": "sardis package required. Install with: pip install sardis",
    }

  api_key = os.environ.get("SARDIS_API_KEY")
  wallet_id = os.environ.get("SARDIS_WALLET_ID")
  if not api_key:
    return {
        "status": "error",
        "error": "SARDIS_API_KEY environment variable not set.",
    }
  if not wallet_id:
    return {
        "status": "error",
        "error": "SARDIS_WALLET_ID environment variable not set.",
    }

  try:
    amount_value = _parse_amount(amount)
  except ValueError as exc:
    return _sardis_error_response(exc)

  client = SardisClient(api_key=api_key)
  try:
    result = client.payments.send(
        wallet_id,
        to=recipient,
        amount=amount_value,
        token=token,
        chain=chain,
        purpose=memo or "Payment",
    )
  except _sardis_exception_types() as exc:
    return _sardis_error_response(exc)
  return {
      "status": "APPROVED" if result.success else "BLOCKED",
      "transaction_id": getattr(result, "tx_id", ""),
      "message": getattr(result, "message", ""),
      "amount": amount,
      "token": token,
      "chain": chain,
  }


def sardis_check_balance(
    token: str = "USDC",
) -> dict:
  """Check the balance of a Sardis wallet.

  Args:
    token: Token symbol to check (e.g. "USDC"). Default: USDC.

  Returns:
    A dict with keys: balance, remaining, token.
  """
  try:
    from sardis import SardisClient
  except ImportError:
    return {"status": "error", "error": "sardis package required."}

  api_key = os.environ.get("SARDIS_API_KEY")
  wallet_id = os.environ.get("SARDIS_WALLET_ID")
  if not api_key or not wallet_id:
    return {"status": "error", "error": "SARDIS_API_KEY and SARDIS_WALLET_ID required."}

  client = SardisClient(api_key=api_key)
  try:
    balance = client.wallets.get_balance(wallet_id, token=token)
  except _sardis_exception_types() as exc:
    return _sardis_error_response(exc)
  return {
      "balance": str(balance.balance),
      "remaining": str(balance.remaining),
      "token": token,
  }


def sardis_check_policy(
    amount: str,
    merchant: str,
) -> dict:
  """Check if a payment would be allowed by spending policy.

  Args:
    amount: Amount to check as a decimal string (e.g. "250.00").
    merchant: Merchant or recipient to check against policy rules.

  Returns:
    A dict with keys: allowed (bool), reason (str).
  """
  try:
    from sardis import SardisClient
  except ImportError:
    return {"status": "error", "error": "sardis package required."}

  api_key = os.environ.get("SARDIS_API_KEY")
  wallet_id = os.environ.get("SARDIS_WALLET_ID")
  if not api_key or not wallet_id:
    return {"status": "error", "error": "SARDIS_API_KEY and SARDIS_WALLET_ID required."}

  try:
    amount_value = _parse_amount(amount)
  except ValueError as exc:
    return _sardis_error_response(exc)

  client = SardisClient(api_key=api_key)
  try:
    balance = client.wallets.get_balance(wallet_id)
  except _sardis_exception_types() as exc:
    return _sardis_error_response(exc)
  if amount_value > Decimal(str(balance.remaining)):
    return {
        "allowed": False,
        "reason": (
            f"Amount ${amount_value} exceeds remaining limit "
            f"${balance.remaining}"
        ),
    }
  if amount_value > Decimal(str(balance.balance)):
    return {
        "allowed": False,
        "reason": f"Amount ${amount_value} exceeds balance ${balance.balance}",
    }
  return {
      "allowed": True,
      "reason": f"Payment of ${amount_value} to {merchant} would be allowed",
  }
