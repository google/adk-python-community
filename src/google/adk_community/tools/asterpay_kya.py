"""AsterPay KYA (Know Your Agent) tools for Google ADK.

Provides trust scoring, identity verification, and EUR settlement
estimates for AI agent commerce. Free API — no key required.

Usage with Google ADK:
    from google.adk import Agent
    from google.adk_community.tools import asterpay_kya_trust_score

    agent = Agent(
        name="payment_verifier",
        model="gemini-2.0-flash",
        tools=[asterpay_kya_trust_score],
    )

Docs: https://asterpay.io
API:  https://x402.asterpay.io
"""

import json

import requests


def asterpay_kya_trust_score(address: str) -> str:
    """Check the trust score (0-100) of an AI agent or wallet.

    Returns trust tier (Open/Verified/Trusted/Enterprise), component
    scores (ERC-8004 identity, sanctions screening, on-chain activity,
    behavioral signals), and risk assessment.

    Use this before paying or transacting with an unknown agent.
    Free — no API key required.

    Args:
        address: Ethereum address (0x...) to score.

    Returns:
        JSON string with trust score, tier, and component breakdown.
    """
    try:
        resp = requests.get(
            f"https://x402.asterpay.io/v1/kya/trust-score/{address}",
            timeout=30,
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2, default=str)
    except Exception as e:
        return f"Error checking trust score: {e}"


def asterpay_kya_verify(address: str) -> str:
    """Verify if an Ethereum address is a registered AI agent (ERC-8004).

    Checks on-chain registration on Base and returns agent ID, owner,
    and metadata URI.

    Free — no API key required.

    Args:
        address: Ethereum address (0x...) to verify.

    Returns:
        JSON string with verified status, agent ID, owner, metadata.
    """
    try:
        resp = requests.get(
            f"https://x402.asterpay.io/v1/kya/verify/{address}",
            timeout=30,
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2, default=str)
    except Exception as e:
        return f"Error verifying agent: {e}"


def asterpay_kya_tier(address: str) -> str:
    """Get trust tier and spending limit for an AI agent.

    Returns tier (Open/Verified/Trusted/Enterprise) and maximum
    per-transaction limit.

    Free — no API key required.

    Args:
        address: Ethereum address (0x...) to check.

    Returns:
        JSON string with tier classification and limits.
    """
    try:
        resp = requests.get(
            f"https://x402.asterpay.io/v1/kya/tier/{address}",
            timeout=30,
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2, default=str)
    except Exception as e:
        return f"Error getting tier: {e}"


def asterpay_settlement_estimate(amount_usdc: float) -> str:
    """Estimate USDC to EUR settlement via SEPA Instant.

    Returns estimated EUR amount, fees, exchange rate, and
    settlement time (typically under 10 seconds).

    Free — no API key required.

    Args:
        amount_usdc: Amount in USDC to settle to EUR.

    Returns:
        JSON string with EUR estimate, fees, and rate.
    """
    try:
        resp = requests.get(
            "https://x402.asterpay.io/v1/settlement/estimate",
            params={"amount": amount_usdc},
            timeout=30,
        )
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2, default=str)
    except Exception as e:
        return f"Error estimating settlement: {e}"
