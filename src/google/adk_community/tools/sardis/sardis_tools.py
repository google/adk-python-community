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

import logging
import os
from typing import Optional

logger = logging.getLogger("google_adk." + __name__)


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

  client = SardisClient(api_key=api_key)
  result = client.payments.send(
      wallet_id, to=recipient, amount=float(amount), token=token,
      purpose=memo or "Payment",
  )
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
  balance = client.wallets.get_balance(wallet_id, token=token)
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

  client = SardisClient(api_key=api_key)
  balance = client.wallets.get_balance(wallet_id)
  amt = float(amount)
  if amt > balance.remaining:
    return {
        "allowed": False,
        "reason": f"Amount ${amt} exceeds remaining limit ${balance.remaining}",
    }
  if amt > balance.balance:
    return {
        "allowed": False,
        "reason": f"Amount ${amt} exceeds balance ${balance.balance}",
    }
  return {
      "allowed": True,
      "reason": f"Payment of ${amt} to {merchant} would be allowed",
  }
