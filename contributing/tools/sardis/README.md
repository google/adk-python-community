# Sardis Payment Tools for Google ADK

Policy-controlled payments for AI agents built with [Google Agent Development Kit](https://google.github.io/adk-docs/).

## Installation

```bash
pip install sardis-adk
```

## Setup

Set your environment variables:

```bash
export SARDIS_API_KEY="sk_..."
export SARDIS_WALLET_ID="wal_..."
```

## Usage

```python
from google.adk.agents import Agent
from tools.sardis import sardis_pay, sardis_check_balance, sardis_check_policy

agent = Agent(
    model="gemini-2.0-flash",
    name="payment_agent",
    description="An agent that can make policy-controlled payments",
    tools=[sardis_pay, sardis_check_balance, sardis_check_policy],
)
```

## Available Tools

| Tool | Description |
|------|-------------|
| `sardis_pay` | Execute a payment with policy guardrails |
| `sardis_check_balance` | Check wallet balance and spending limits |
| `sardis_check_policy` | Pre-check if a payment would be allowed |

## Features

- **Policy guardrails** — spending limits, merchant restrictions, category rules
- **Non-custodial** — MPC wallets, no private keys stored
- **Multi-chain** — Base, Polygon, Ethereum, Arbitrum, Optimism
- **Stablecoin native** — USDC, USDT, EURC, PYUSD
- **Full audit trail** — append-only ledger for compliance

## Links

- [Sardis Documentation](https://sardis.sh/docs)
- [sardis-adk on PyPI](https://pypi.org/project/sardis-adk/)
- [GitHub](https://github.com/EfeDurmaz16/sardis)
