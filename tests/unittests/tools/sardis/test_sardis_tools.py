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

from unittest.mock import MagicMock, patch

from google.adk_community.tools.sardis.sardis_tools import (
    sardis_check_balance,
    sardis_check_policy,
    sardis_pay,
)


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
