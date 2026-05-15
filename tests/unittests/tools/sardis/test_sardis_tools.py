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

"""Tests for Sardis payment tools."""

from decimal import Decimal
import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import MagicMock

from google.adk_community.tools.sardis.sardis_tools import (
    sardis_check_balance,
    sardis_check_policy,
    sardis_pay,
)


def _install_fake_sardis(monkeypatch, client):
  module = ModuleType("sardis")

  class AuthenticationError(Exception):
    pass

  class InsufficientBalanceError(Exception):
    pass

  class PolicyViolationError(Exception):
    pass

  module.SardisClient = MagicMock(return_value=client)
  module.AuthenticationError = AuthenticationError
  module.InsufficientBalanceError = InsufficientBalanceError
  module.PolicyViolationError = PolicyViolationError
  monkeypatch.setitem(sys.modules, "sardis", module)
  return module


def test_sardis_pay_missing_api_key(monkeypatch):
  monkeypatch.delenv("SARDIS_API_KEY", raising=False)
  monkeypatch.delenv("SARDIS_WALLET_ID", raising=False)
  result = sardis_pay(recipient="0xABC", amount="10.00")
  assert result["status"] == "error"


def test_sardis_check_balance_missing_key(monkeypatch):
  monkeypatch.delenv("SARDIS_API_KEY", raising=False)
  result = sardis_check_balance()
  assert result["status"] == "error"


def test_sardis_check_policy_missing_key(monkeypatch):
  monkeypatch.delenv("SARDIS_API_KEY", raising=False)
  result = sardis_check_policy(amount="100.00", merchant="openai")
  assert result["status"] == "error"


def test_sardis_pay_uses_decimal_amount_and_chain(monkeypatch):
  monkeypatch.setenv("SARDIS_API_KEY", "sk_test")
  monkeypatch.setenv("SARDIS_WALLET_ID", "wal_test")
  client = MagicMock()
  client.payments.send.return_value = SimpleNamespace(
      success=True,
      tx_id="tx_123",
      message="approved",
  )
  _install_fake_sardis(monkeypatch, client)

  result = sardis_pay(
      recipient="merchant_123",
      amount="10.25",
      chain="polygon",
      memo="invoice",
  )

  assert result["status"] == "APPROVED"
  client.payments.send.assert_called_once_with(
      "wal_test",
      to="merchant_123",
      amount=Decimal("10.25"),
      token="USDC",
      chain="polygon",
      purpose="invoice",
  )


def test_sardis_pay_returns_typed_sdk_errors(monkeypatch):
  monkeypatch.setenv("SARDIS_API_KEY", "sk_test")
  monkeypatch.setenv("SARDIS_WALLET_ID", "wal_test")
  client = MagicMock()
  sardis = _install_fake_sardis(monkeypatch, client)
  client.payments.send.side_effect = sardis.PolicyViolationError("blocked")

  result = sardis_pay(recipient="merchant_123", amount="10.25")

  assert result == {"status": "error", "error": "blocked"}


def test_sardis_check_policy_uses_decimal_comparison(monkeypatch):
  monkeypatch.setenv("SARDIS_API_KEY", "sk_test")
  monkeypatch.setenv("SARDIS_WALLET_ID", "wal_test")
  client = MagicMock()
  client.wallets.get_balance.return_value = SimpleNamespace(
      balance=Decimal("20.00"),
      remaining=Decimal("15.00"),
  )
  _install_fake_sardis(monkeypatch, client)

  result = sardis_check_policy(amount="10.25", merchant="openai")

  assert result["allowed"] is True
  assert "10.25" in result["reason"]
