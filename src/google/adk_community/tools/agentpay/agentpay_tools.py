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

"""AgentPay x402 payment tools for Google Agent Development Kit.

These tools enable ADK agents to make autonomous HTTP payments using the x402
protocol (HTTP 402 Payment Required). When an agent requests a paid API
endpoint, the server returns HTTP 402 with payment details. AgentPay handles
the on-chain payment automatically and retries the request.

Requires:
    Node.js >= 18
    npm install -g agentpay-mcp

Environment variables:
    AGENTPAY_PRIVATE_KEY: Wallet private key (0x-prefixed hex string).
    AGENTPAY_RPC_URL: (Optional) RPC endpoint. Defaults to Base mainnet.

Installation:
    pip install google-adk-community
    npm install -g agentpay-mcp

Usage:
    from google.adk.agents import Agent
    from google.adk_community.tools.agentpay import (
        fetch_paid_api,
        get_wallet_info,
        check_spend_limit,
        send_payment,
    )

    agent = Agent(
        model="gemini-2.0-flash",
        name="payment_agent",
        description="Agent that pays for API access autonomously via x402",
        tools=[fetch_paid_api, get_wallet_info, check_spend_limit, send_payment],
    )

Protocol:
    x402 is an open standard that revives HTTP's 402 Payment Required status
    code for machine-to-machine micropayments. See https://www.x402.org/

Package:
    agentpay-mcp: https://www.npmjs.com/package/agentpay-mcp (patent pending)
    agentwallet-sdk: https://www.npmjs.com/package/agentwallet-sdk
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Optional

logger = logging.getLogger("google_adk." + __name__)

# MCP server executable name (installed globally via npm)
_MCP_SERVER = "agentpay-mcp"

# MCP protocol framing: each message is a JSON-RPC 2.0 object on one line
_JSONRPC_VERSION = "2.0"


def _check_mcp_server() -> Optional[str]:
  """Return path to agentpay-mcp binary, or None if not installed."""
  path = shutil.which(_MCP_SERVER)
  if path:
    return path
  # Also check common npx/node_modules/.bin locations
  npx = shutil.which("npx")
  if npx:
    return None  # caller will use npx -y agentpay-mcp
  return None


def _build_env() -> dict[str, str]:
  """Build subprocess environment with required AgentPay variables."""
  env = os.environ.copy()
  return env


def _mcp_call(method: str, params: dict[str, Any]) -> dict[str, Any]:
  """Send a single JSON-RPC 2.0 request to the agentpay-mcp stdio server.

  Spawns the MCP server as a subprocess, sends one request, reads the
  response, and terminates the process. This is the standard MCP stdio
  transport pattern.

  Args:
    method: MCP tool name to invoke (e.g. "fetch_paid_api").
    params: Parameters dict for the tool call.

  Returns:
    Parsed JSON response dict, or an error dict on failure.
  """
  private_key = os.environ.get("AGENTPAY_PRIVATE_KEY")
  if not private_key:
    return {
        "status": "error",
        "error": (
            "AGENTPAY_PRIVATE_KEY environment variable not set. "
            "Set it to your AgentPay wallet private key (0x-prefixed)."
        ),
    }

  # Build the JSON-RPC initialize + call sequence for MCP stdio transport
  init_request = {
      "jsonrpc": _JSONRPC_VERSION,
      "id": 1,
      "method": "initialize",
      "params": {
          "protocolVersion": "2024-11-05",
          "capabilities": {},
          "clientInfo": {"name": "google-adk-community", "version": "1.0.0"},
      },
  }
  tool_request = {
      "jsonrpc": _JSONRPC_VERSION,
      "id": 2,
      "method": "tools/call",
      "params": {"name": method, "arguments": params},
  }
  # MCP stdio: newline-delimited JSON
  stdin_payload = (
      json.dumps(init_request) + "\n" + json.dumps(tool_request) + "\n"
  )

  server_path = _check_mcp_server()
  if server_path:
    cmd = [server_path]
  elif shutil.which("npx"):
    cmd = ["npx", "--yes", _MCP_SERVER]
  else:
    return {
        "status": "error",
        "error": (
            f"'{_MCP_SERVER}' not found. Install with: npm install -g"
            f" {_MCP_SERVER}"
        ),
    }

  env = _build_env()
  try:
    result = subprocess.run(
        cmd,
        input=stdin_payload,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
  except subprocess.TimeoutExpired:
    return {"status": "error", "error": "agentpay-mcp timed out after 60s."}
  except FileNotFoundError:
    return {
        "status": "error",
        "error": (
            f"'{_MCP_SERVER}' not found. Install with: npm install -g"
            f" {_MCP_SERVER}"
        ),
    }
  except OSError as exc:
    return {"status": "error", "error": f"Failed to launch agentpay-mcp: {exc}"}

  if result.returncode != 0:
    stderr_snippet = result.stderr[:500] if result.stderr else ""
    logger.debug("agentpay-mcp stderr: %s", result.stderr)
    return {
        "status": "error",
        "error": f"agentpay-mcp exited with code {result.returncode}.",
        "detail": stderr_snippet,
    }

  # Parse last JSON-RPC response (id=2) from stdout
  tool_response: Optional[dict[str, Any]] = None
  for line in result.stdout.splitlines():
    line = line.strip()
    if not line:
      continue
    try:
      obj = json.loads(line)
      if obj.get("id") == 2:
        tool_response = obj
    except json.JSONDecodeError:
      logger.debug("agentpay-mcp non-JSON line: %s", line)

  if tool_response is None:
    return {
        "status": "error",
        "error": "No response received from agentpay-mcp.",
        "stdout": result.stdout[:500],
    }

  if "error" in tool_response:
    rpc_err = tool_response["error"]
    return {
        "status": "error",
        "error": rpc_err.get("message", str(rpc_err)),
        "code": rpc_err.get("code"),
    }

  # MCP tools/call result: {"result": {"content": [{"type": "text", "text": "..."}]}}
  mcp_result = tool_response.get("result", {})
  content = mcp_result.get("content", [])
  if content and content[0].get("type") == "text":
    text = content[0]["text"]
    try:
      return json.loads(text)
    except json.JSONDecodeError:
      return {"status": "ok", "result": text}

  return {"status": "ok", "result": mcp_result}


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------


def fetch_paid_api(
    url: str,
    method: str = "GET",
    headers: Optional[dict[str, str]] = None,
    body: Optional[str] = None,
    max_payment_usdc: Optional[float] = None,
) -> dict:
  """Fetch a URL, automatically paying any x402 HTTP 402 challenge.

  If the server responds with HTTP 402 Payment Required, AgentPay handles
  the on-chain USDC payment on Base and retries the request automatically.
  Spend limits are enforced on-chain before any payment is signed.

  Args:
    url: Full URL of the API endpoint to fetch.
    method: HTTP method (GET, POST, PUT, DELETE). Default: GET.
    headers: Optional additional HTTP headers as a dict.
    body: Optional request body string (for POST/PUT).
    max_payment_usdc: Optional ceiling on what the agent will pay for this
      request (USDC). If the 402 challenge exceeds this, the request is
      aborted without payment.

  Returns:
    A dict with keys:
      - status: "ok" or "error"
      - http_status: HTTP status code of the final response (int)
      - body: Response body string
      - payment_made: True if an x402 payment was executed
      - amount_paid_usdc: Amount paid in USDC (float), 0.0 if no payment
      - tx_hash: On-chain transaction hash if payment was made, else None
      - error: Error message string (only present on failure)
  """
  params: dict[str, Any] = {"url": url, "method": method.upper()}
  if headers:
    params["headers"] = headers
  if body:
    params["body"] = body
  if max_payment_usdc is not None:
    params["max_payment_usdc"] = max_payment_usdc

  logger.info("fetch_paid_api: %s %s", method.upper(), url)
  return _mcp_call("fetch_paid_api", params)


def get_wallet_info() -> dict:
  """Return AgentPay wallet address, USDC balance, and spend limits.

  Queries the AgentPay smart contract on Base for the current wallet state,
  including on-chain spend limits set by the wallet owner.

  Returns:
    A dict with keys:
      - status: "ok" or "error"
      - address: Wallet address (0x-prefixed string)
      - balance_usdc: Current USDC balance (float)
      - spend_limit_per_tx_usdc: Maximum single-transaction spend limit (float)
      - spend_limit_daily_usdc: Maximum daily spend limit (float)
      - spent_today_usdc: Amount spent today so far (float)
      - remaining_daily_usdc: Remaining daily allowance (float)
      - chain: Chain name (e.g. "base")
      - error: Error message string (only present on failure)
  """
  logger.info("get_wallet_info: querying AgentPay wallet")
  return _mcp_call("get_wallet_info", {})


def check_spend_limit(
    amount_usdc: float,
) -> dict:
  """Pre-check whether a payment is within on-chain spend limits.

  Validates against both per-transaction and daily spend limits enforced
  by the AgentPay smart contract. Use this before committing to a payment.

  Args:
    amount_usdc: Proposed payment amount in USDC.

  Returns:
    A dict with keys:
      - status: "ok" or "error"
      - allowed: True if the payment is within all limits (bool)
      - amount_usdc: The amount checked (float)
      - spend_limit_per_tx_usdc: Per-transaction limit (float)
      - spend_limit_daily_usdc: Daily limit (float)
      - spent_today_usdc: Amount spent so far today (float)
      - remaining_daily_usdc: Remaining daily allowance (float)
      - reason: Human-readable explanation if not allowed (str or None)
      - error: Error message string (only present on failure)
  """
  if amount_usdc <= 0:
    return {
        "status": "error",
        "error": "amount_usdc must be greater than 0.",
    }

  logger.info("check_spend_limit: checking %.6f USDC", amount_usdc)
  return _mcp_call("check_spend_limit", {"amount_usdc": amount_usdc})


def send_payment(
    recipient: str,
    amount_usdc: float,
    memo: Optional[str] = None,
) -> dict:
  """Send USDC directly to a wallet address within spend policy limits.

  Executes an on-chain USDC transfer on Base. The payment is validated
  against on-chain spend limits before signing. The transaction is
  non-custodial — the private key never leaves the agent's environment.

  Args:
    recipient: Destination wallet address (0x-prefixed).
    amount_usdc: Amount of USDC to send (e.g. 1.50 for $1.50).
    memo: Optional memo string stored in transaction calldata.

  Returns:
    A dict with keys:
      - status: "ok" or "error"
      - tx_hash: On-chain transaction hash (str)
      - recipient: Destination address (str)
      - amount_usdc: Amount sent (float)
      - chain: Chain name (e.g. "base")
      - block_number: Block number of the transaction (int)
      - error: Error message string (only present on failure)
  """
  if amount_usdc <= 0:
    return {
        "status": "error",
        "error": "amount_usdc must be greater than 0.",
    }
  if not recipient or not recipient.startswith("0x"):
    return {
        "status": "error",
        "error": "recipient must be a 0x-prefixed Ethereum address.",
    }

  params: dict[str, Any] = {
      "recipient": recipient,
      "amount_usdc": amount_usdc,
  }
  if memo:
    params["memo"] = memo

  logger.info(
      "send_payment: sending %.6f USDC to %s", amount_usdc, recipient
  )
  return _mcp_call("send_payment", params)
