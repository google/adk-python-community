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

"""Unit tests for AgentPay x402 payment tools.

These tests verify tool behavior across the full matrix of conditions:
missing env vars, subprocess failures, MCP protocol errors, and happy-path
success cases. All external calls (subprocess.run, shutil.which) are mocked
so tests run without Node.js or agentpay-mcp installed.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from google.adk_community.tools.agentpay.agentpay_tools import (
    check_spend_limit,
    fetch_paid_api,
    get_wallet_info,
    send_payment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mcp_response(id_: int, result_text: dict) -> str:
  """Build a newline-delimited MCP JSON-RPC response line."""
  return json.dumps(
      {
          "jsonrpc": "2.0",
          "id": id_,
          "result": {
              "content": [{"type": "text", "text": json.dumps(result_text)}]
          },
      }
  )


def _completed_process(stdout: str, returncode: int = 0) -> MagicMock:
  """Return a mock subprocess.CompletedProcess."""
  mock = MagicMock(spec=subprocess.CompletedProcess)
  mock.returncode = returncode
  mock.stdout = stdout
  mock.stderr = ""
  return mock


# ---------------------------------------------------------------------------
# Missing AGENTPAY_PRIVATE_KEY
# ---------------------------------------------------------------------------


class TestMissingPrivateKey:
  """All tools must return an informative error when the key is absent."""

  def test_fetch_paid_api_missing_key(self, monkeypatch):
    monkeypatch.delenv("AGENTPAY_PRIVATE_KEY", raising=False)
    result = fetch_paid_api(url="https://example.com/api")
    assert result["status"] == "error"
    assert "AGENTPAY_PRIVATE_KEY" in result["error"]

  def test_get_wallet_info_missing_key(self, monkeypatch):
    monkeypatch.delenv("AGENTPAY_PRIVATE_KEY", raising=False)
    result = get_wallet_info()
    assert result["status"] == "error"
    assert "AGENTPAY_PRIVATE_KEY" in result["error"]

  def test_check_spend_limit_missing_key(self, monkeypatch):
    monkeypatch.delenv("AGENTPAY_PRIVATE_KEY", raising=False)
    result = check_spend_limit(amount_usdc=1.0)
    assert result["status"] == "error"
    assert "AGENTPAY_PRIVATE_KEY" in result["error"]

  def test_send_payment_missing_key(self, monkeypatch):
    monkeypatch.delenv("AGENTPAY_PRIVATE_KEY", raising=False)
    result = send_payment(recipient="0xABC123", amount_usdc=1.0)
    assert result["status"] == "error"
    assert "AGENTPAY_PRIVATE_KEY" in result["error"]


# ---------------------------------------------------------------------------
# Input validation (independent of subprocess)
# ---------------------------------------------------------------------------


class TestInputValidation:
  """Tools should validate inputs before spawning a subprocess."""

  def test_check_spend_limit_zero_amount(self, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    result = check_spend_limit(amount_usdc=0)
    assert result["status"] == "error"
    assert "greater than 0" in result["error"]

  def test_check_spend_limit_negative_amount(self, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    result = check_spend_limit(amount_usdc=-5.0)
    assert result["status"] == "error"

  def test_send_payment_zero_amount(self, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    result = send_payment(recipient="0xABC123", amount_usdc=0)
    assert result["status"] == "error"
    assert "greater than 0" in result["error"]

  def test_send_payment_negative_amount(self, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    result = send_payment(recipient="0xABC123", amount_usdc=-1.0)
    assert result["status"] == "error"

  def test_send_payment_invalid_recipient(self, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    result = send_payment(recipient="not-an-address", amount_usdc=1.0)
    assert result["status"] == "error"
    assert "0x-prefixed" in result["error"]

  def test_send_payment_empty_recipient(self, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    result = send_payment(recipient="", amount_usdc=1.0)
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# MCP server not installed
# ---------------------------------------------------------------------------


class TestMcpServerNotFound:
  """When agentpay-mcp and npx are both absent, return a clear error."""

  @patch("google.adk_community.tools.agentpay.agentpay_tools.shutil.which", return_value=None)
  def test_fetch_paid_api_no_mcp(self, mock_which, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    result = fetch_paid_api(url="https://example.com/api")
    assert result["status"] == "error"
    assert "agentpay-mcp" in result["error"]

  @patch("google.adk_community.tools.agentpay.agentpay_tools.shutil.which", return_value=None)
  def test_get_wallet_info_no_mcp(self, mock_which, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    result = get_wallet_info()
    assert result["status"] == "error"
    assert "agentpay-mcp" in result["error"]

  @patch("google.adk_community.tools.agentpay.agentpay_tools.shutil.which", return_value=None)
  def test_check_spend_limit_no_mcp(self, mock_which, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    result = check_spend_limit(amount_usdc=5.0)
    assert result["status"] == "error"

  @patch("google.adk_community.tools.agentpay.agentpay_tools.shutil.which", return_value=None)
  def test_send_payment_no_mcp(self, mock_which, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    result = send_payment(recipient="0xABC123", amount_usdc=1.0)
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Subprocess failure
# ---------------------------------------------------------------------------


class TestSubprocessFailure:
  """When agentpay-mcp exits non-zero, propagate the error."""

  @patch("google.adk_community.tools.agentpay.agentpay_tools.subprocess.run")
  @patch(
      "google.adk_community.tools.agentpay.agentpay_tools.shutil.which",
      return_value="/usr/local/bin/agentpay-mcp",
  )
  def test_fetch_paid_api_subprocess_error(
      self, mock_which, mock_run, monkeypatch
  ):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    mock_run.return_value = _completed_process("", returncode=1)
    mock_run.return_value.stderr = "internal error"
    result = fetch_paid_api(url="https://example.com/api")
    assert result["status"] == "error"
    assert "1" in result["error"]

  @patch("google.adk_community.tools.agentpay.agentpay_tools.subprocess.run")
  @patch(
      "google.adk_community.tools.agentpay.agentpay_tools.shutil.which",
      return_value="/usr/local/bin/agentpay-mcp",
  )
  def test_timeout(self, mock_which, mock_run, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="agentpay-mcp", timeout=60)
    result = fetch_paid_api(url="https://example.com/api")
    assert result["status"] == "error"
    assert "timed out" in result["error"].lower()

  @patch("google.adk_community.tools.agentpay.agentpay_tools.subprocess.run")
  @patch(
      "google.adk_community.tools.agentpay.agentpay_tools.shutil.which",
      return_value="/usr/local/bin/agentpay-mcp",
  )
  def test_no_response_from_mcp(self, mock_which, mock_run, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    # Return only the init response (id=1), no tool response (id=2)
    init_resp = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}
    )
    mock_run.return_value = _completed_process(init_resp + "\n")
    result = fetch_paid_api(url="https://example.com/api")
    assert result["status"] == "error"
    assert "No response" in result["error"]


# ---------------------------------------------------------------------------
# MCP protocol errors
# ---------------------------------------------------------------------------


class TestMcpProtocolError:
  """JSON-RPC error responses from agentpay-mcp should surface cleanly."""

  @patch("google.adk_community.tools.agentpay.agentpay_tools.subprocess.run")
  @patch(
      "google.adk_community.tools.agentpay.agentpay_tools.shutil.which",
      return_value="/usr/local/bin/agentpay-mcp",
  )
  def test_rpc_error_propagated(self, mock_which, mock_run, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    error_resp = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "error": {"code": -32601, "message": "Method not found"},
        }
    )
    mock_run.return_value = _completed_process(error_resp + "\n")
    result = get_wallet_info()
    assert result["status"] == "error"
    assert "Method not found" in result["error"]


# ---------------------------------------------------------------------------
# Happy-path success cases
# ---------------------------------------------------------------------------


class TestHappyPath:
  """Verify correct parsing of well-formed MCP responses."""

  @patch("google.adk_community.tools.agentpay.agentpay_tools.subprocess.run")
  @patch(
      "google.adk_community.tools.agentpay.agentpay_tools.shutil.which",
      return_value="/usr/local/bin/agentpay-mcp",
  )
  def test_fetch_paid_api_success_no_payment(
      self, mock_which, mock_run, monkeypatch
  ):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    payload = {
        "status": "ok",
        "http_status": 200,
        "body": '{"price": 42000}',
        "payment_made": False,
        "amount_paid_usdc": 0.0,
        "tx_hash": None,
    }
    stdout = _mcp_response(2, payload) + "\n"
    mock_run.return_value = _completed_process(stdout)
    result = fetch_paid_api(url="https://api.example.com/price")
    assert result["status"] == "ok"
    assert result["http_status"] == 200
    assert result["payment_made"] is False
    assert result["amount_paid_usdc"] == 0.0

  @patch("google.adk_community.tools.agentpay.agentpay_tools.subprocess.run")
  @patch(
      "google.adk_community.tools.agentpay.agentpay_tools.shutil.which",
      return_value="/usr/local/bin/agentpay-mcp",
  )
  def test_fetch_paid_api_success_with_payment(
      self, mock_which, mock_run, monkeypatch
  ):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    payload = {
        "status": "ok",
        "http_status": 200,
        "body": '{"data": "premium"}',
        "payment_made": True,
        "amount_paid_usdc": 0.001,
        "tx_hash": "0xabcdef1234567890",
    }
    stdout = _mcp_response(2, payload) + "\n"
    mock_run.return_value = _completed_process(stdout)
    result = fetch_paid_api(
        url="https://paid-api.example.com/data",
        max_payment_usdc=0.01,
    )
    assert result["status"] == "ok"
    assert result["payment_made"] is True
    assert result["tx_hash"] == "0xabcdef1234567890"
    assert result["amount_paid_usdc"] == 0.001

  @patch("google.adk_community.tools.agentpay.agentpay_tools.subprocess.run")
  @patch(
      "google.adk_community.tools.agentpay.agentpay_tools.shutil.which",
      return_value="/usr/local/bin/agentpay-mcp",
  )
  def test_get_wallet_info_success(self, mock_which, mock_run, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    payload = {
        "status": "ok",
        "address": "0x1234567890abcdef1234567890abcdef12345678",
        "balance_usdc": 100.0,
        "spend_limit_per_tx_usdc": 10.0,
        "spend_limit_daily_usdc": 50.0,
        "spent_today_usdc": 5.0,
        "remaining_daily_usdc": 45.0,
        "chain": "base",
    }
    stdout = _mcp_response(2, payload) + "\n"
    mock_run.return_value = _completed_process(stdout)
    result = get_wallet_info()
    assert result["status"] == "ok"
    assert result["balance_usdc"] == 100.0
    assert result["chain"] == "base"
    assert result["remaining_daily_usdc"] == 45.0

  @patch("google.adk_community.tools.agentpay.agentpay_tools.subprocess.run")
  @patch(
      "google.adk_community.tools.agentpay.agentpay_tools.shutil.which",
      return_value="/usr/local/bin/agentpay-mcp",
  )
  def test_check_spend_limit_allowed(self, mock_which, mock_run, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    payload = {
        "status": "ok",
        "allowed": True,
        "amount_usdc": 2.5,
        "spend_limit_per_tx_usdc": 10.0,
        "spend_limit_daily_usdc": 50.0,
        "spent_today_usdc": 5.0,
        "remaining_daily_usdc": 45.0,
        "reason": None,
    }
    stdout = _mcp_response(2, payload) + "\n"
    mock_run.return_value = _completed_process(stdout)
    result = check_spend_limit(amount_usdc=2.5)
    assert result["status"] == "ok"
    assert result["allowed"] is True
    assert result["reason"] is None

  @patch("google.adk_community.tools.agentpay.agentpay_tools.subprocess.run")
  @patch(
      "google.adk_community.tools.agentpay.agentpay_tools.shutil.which",
      return_value="/usr/local/bin/agentpay-mcp",
  )
  def test_check_spend_limit_blocked(self, mock_which, mock_run, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    payload = {
        "status": "ok",
        "allowed": False,
        "amount_usdc": 100.0,
        "spend_limit_per_tx_usdc": 10.0,
        "spend_limit_daily_usdc": 50.0,
        "spent_today_usdc": 5.0,
        "remaining_daily_usdc": 45.0,
        "reason": "Exceeds per-transaction limit of $10.00",
    }
    stdout = _mcp_response(2, payload) + "\n"
    mock_run.return_value = _completed_process(stdout)
    result = check_spend_limit(amount_usdc=100.0)
    assert result["status"] == "ok"
    assert result["allowed"] is False
    assert "per-transaction" in result["reason"]

  @patch("google.adk_community.tools.agentpay.agentpay_tools.subprocess.run")
  @patch(
      "google.adk_community.tools.agentpay.agentpay_tools.shutil.which",
      return_value="/usr/local/bin/agentpay-mcp",
  )
  def test_send_payment_success(self, mock_which, mock_run, monkeypatch):
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    payload = {
        "status": "ok",
        "tx_hash": "0xdeadbeef1234567890abcdef",
        "recipient": "0xABCDEF1234567890abcdef1234567890ABCDEF12",
        "amount_usdc": 5.0,
        "chain": "base",
        "block_number": 12345678,
    }
    stdout = _mcp_response(2, payload) + "\n"
    mock_run.return_value = _completed_process(stdout)
    result = send_payment(
        recipient="0xABCDEF1234567890abcdef1234567890ABCDEF12",
        amount_usdc=5.0,
        memo="Test payment",
    )
    assert result["status"] == "ok"
    assert result["tx_hash"] == "0xdeadbeef1234567890abcdef"
    assert result["amount_usdc"] == 5.0
    assert result["chain"] == "base"

  @patch("google.adk_community.tools.agentpay.agentpay_tools.subprocess.run")
  @patch(
      "google.adk_community.tools.agentpay.agentpay_tools.shutil.which",
      return_value="/usr/local/bin/agentpay-mcp",
  )
  def test_fetch_paid_api_passes_method_and_headers(
      self, mock_which, mock_run, monkeypatch
  ):
    """Verify that HTTP method and headers are forwarded to the MCP call."""
    monkeypatch.setenv("AGENTPAY_PRIVATE_KEY", "0xdeadbeef")
    payload = {"status": "ok", "http_status": 200, "body": "ok",
               "payment_made": False, "amount_paid_usdc": 0.0, "tx_hash": None}
    stdout = _mcp_response(2, payload) + "\n"
    mock_run.return_value = _completed_process(stdout)
    result = fetch_paid_api(
        url="https://api.example.com/submit",
        method="POST",
        headers={"Content-Type": "application/json"},
        body='{"query": "hello"}',
    )
    assert result["status"] == "ok"
    # Verify subprocess was called with the correct stdin containing our params
    call_args = mock_run.call_args
    stdin_payload = call_args.kwargs.get("input", "")
    # The tool_request line should contain method=POST and our headers
    assert "POST" in stdin_payload
    assert "Content-Type" in stdin_payload


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


class TestModuleImports:
  """Ensure all public symbols are importable from the package."""

  def test_import_all_tools(self):
    from google.adk_community.tools.agentpay import (  # noqa: F401
        check_spend_limit,
        fetch_paid_api,
        get_wallet_info,
        send_payment,
    )

  def test_all_symbols_in_dunder_all(self):
    import google.adk_community.tools.agentpay as pkg

    assert "fetch_paid_api" in pkg.__all__
    assert "get_wallet_info" in pkg.__all__
    assert "check_spend_limit" in pkg.__all__
    assert "send_payment" in pkg.__all__
