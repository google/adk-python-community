# Spraay Batch Payment Tools for Google ADK

[Spraay](https://spraay.app) enables AI agents to batch-send ETH or ERC-20 tokens to up to 200 recipients in a single transaction on [Base](https://base.org), with ~80% gas savings compared to individual transfers.

## Overview

These tools allow any Google ADK agent to execute batch cryptocurrency payments on Base. Common use cases include:

- **Payroll**: Pay team members in ETH or stablecoins in one transaction
- **Airdrops**: Distribute tokens to community members efficiently
- **Bounties**: Send rewards to multiple contributors at once
- **Revenue sharing**: Split payments across stakeholders

## Installation

```bash
pip install google-adk-community web3
```

## Quick Start

```python
from google.adk.agents import Agent
from google.adk_community.tools.spraay import (
    spraay_batch_eth,
    spraay_batch_token,
    spraay_batch_eth_variable,
    spraay_batch_token_variable,
)

agent = Agent(
    name="payment_agent",
    model="gemini-2.5-flash",
    instruction="""You are a payment assistant that helps users send
    batch cryptocurrency payments on Base using Spraay. Always confirm
    recipient addresses and amounts before executing transactions.""",
    tools=[
        spraay_batch_eth,
        spraay_batch_token,
        spraay_batch_eth_variable,
        spraay_batch_token_variable,
    ],
)
```

## Configuration

Set the following environment variables:

| Variable | Required | Description |
|---|---|---|
| `SPRAAY_PRIVATE_KEY` | Yes | Private key of the sending wallet |
| `SPRAAY_RPC_URL` | No | Base RPC endpoint (default: `https://mainnet.base.org`) |
| `SPRAAY_CONTRACT_ADDRESS` | No | Override Spraay contract address |

```bash
export SPRAAY_PRIVATE_KEY="0x..."
```

## Tools

### `spraay_batch_eth`

Send equal amounts of ETH to multiple recipients.

```python
# Example: Send 0.01 ETH to 3 recipients
result = spraay_batch_eth(
    recipients=["0xAddr1...", "0xAddr2...", "0xAddr3..."],
    amount_per_recipient_eth="0.01",
)
```

### `spraay_batch_token`

Send equal amounts of an ERC-20 token to multiple recipients. Handles token approval automatically.

```python
# Example: Send 100 USDC to 3 recipients
USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
result = spraay_batch_token(
    token_address=USDC_BASE,
    recipients=["0xAddr1...", "0xAddr2...", "0xAddr3..."],
    amount_per_recipient="100",
    token_decimals=6,  # USDC uses 6 decimals
)
```

### `spraay_batch_eth_variable`

Send different ETH amounts to each recipient.

```python
# Example: Send variable amounts to 3 recipients
result = spraay_batch_eth_variable(
    recipients=["0xAddr1...", "0xAddr2...", "0xAddr3..."],
    amounts_eth=["0.1", "0.25", "0.05"],
)
```

### `spraay_batch_token_variable`

Send different token amounts to each recipient.

```python
# Example: Send variable USDC amounts to 3 recipients
result = spraay_batch_token_variable(
    token_address=USDC_BASE,
    recipients=["0xAddr1...", "0xAddr2...", "0xAddr3..."],
    amounts=["100", "250.5", "75"],
    token_decimals=6,
)
```

## Protocol Details

- **Contract**: `0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC` on Base Mainnet
- **Max recipients**: 200 per transaction
- **Fee**: 0.3% protocol fee
- **Gas savings**: ~80% compared to individual transfers
- **Token support**: Any ERC-20 token on Base
- **Website**: [spraay.app](https://spraay.app)
- **Source**: [github.com/plagtech](https://github.com/plagtech)

## Running Tests

```bash
pytest tests/unittests/tools/spraay/ -v
```

## License

Apache 2.0 - See [LICENSE](../../LICENSE) for details.
