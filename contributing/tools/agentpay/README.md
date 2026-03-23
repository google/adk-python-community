# AgentPay x402 Payment Tools for Google ADK

Enable Google ADK agents to make autonomous HTTP payments using the
[x402 payment protocol](https://www.x402.org/) — HTTP 402 Payment Required,
handled automatically.

Powered by [agentpay-mcp](https://www.npmjs.com/package/agentpay-mcp) and
[agentwallet-sdk](https://www.npmjs.com/package/agentwallet-sdk) (patent pending).

## What is x402?

x402 is an open protocol that revives HTTP's original `402 Payment Required`
status code. When an agent hits a paid API endpoint, the server returns a 402
with payment details. The agent's wallet pays automatically and retries the
request — no manual intervention, no API keys for every service.

## What is AgentPay?

AgentPay is a non-custodial smart contract wallet system for AI agents. Wallet
ownership is represented by an NFT — your agent controls funds without a
custodian. On-chain **spend limits** cap per-transaction and daily totals,
protecting against runaway agent behavior.

- **Chain:** Base (live), multi-chain coming
- **Protocol:** x402 / HTTP 402
- **NPM:** [`agentpay-mcp`](https://www.npmjs.com/package/agentpay-mcp) v4.0.0
- **SDK:** [`agentwallet-sdk`](https://www.npmjs.com/package/agentwallet-sdk) v6.0.4
- **Status:** Patent pending

## Installation

```bash
# Python ADK package
pip install google-adk-community

# AgentPay MCP server (requires Node.js ≥ 18)
npm install -g agentpay-mcp
```

## Setup

```bash
# Your AgentPay wallet private key (non-custodial)
export AGENTPAY_PRIVATE_KEY="0x..."

# Optional: custom RPC endpoint (defaults to Base mainnet)
export AGENTPAY_RPC_URL="https://mainnet.base.org"
```

Get a wallet at [agentpay.xyz](https://agentpay.xyz) or via the
`agentwallet-sdk` CLI.

## Usage

```python
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
    description="An agent that can pay for API access autonomously using x402",
    tools=[fetch_paid_api, get_wallet_info, check_spend_limit, send_payment],
)

# The agent can now call paid APIs, check its wallet, and send payments
response = agent.run(
    "Fetch the latest market data from https://api.example.com/market"
    " and check my remaining daily spend limit."
)
```

## Available Tools

| Tool | Description |
|------|-------------|
| `fetch_paid_api` | Make an HTTP request; auto-pay x402 challenge if needed and retry |
| `get_wallet_info` | Return wallet address, USDC balance, spend limits, and remaining allowance |
| `check_spend_limit` | Pre-check whether a payment amount is within on-chain spend limits |
| `send_payment` | Send USDC directly to an address within autonomous spend policy |

## How `fetch_paid_api` Works

```
Agent calls fetch_paid_api(url="https://paid-api.example.com/data")
  │
  ├─ agentpay-mcp sends GET /data
  │    Server returns HTTP 402 + payment details
  │
  ├─ agentpay-mcp pays on-chain (Base, USDC)
  │    On-chain spend limit checked before signing
  │
  └─ agentpay-mcp retries GET /data with payment proof
       Server returns 200 + data → returned to agent
```

## Security Model

- **Non-custodial** — private key stays in your environment; no third party holds funds
- **On-chain spend limits** — per-transaction and daily caps enforced by smart contract
- **NFT ownership** — wallet controlled by NFT; rotate ownership without moving funds
- **Audit trail** — every payment recorded on-chain

## Links

- [AgentPay NPM](https://www.npmjs.com/package/agentpay-mcp)
- [agentwallet-sdk NPM](https://www.npmjs.com/package/agentwallet-sdk)
- [x402 Protocol](https://www.x402.org/)
- [ADK Integrations](https://google.github.io/adk-docs/integrations/)
