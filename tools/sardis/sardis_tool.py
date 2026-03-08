"""Sardis payment tool for Google Agent Development Kit.

This tool enables ADK agents to make policy-controlled payments through
Sardis non-custodial MPC wallets.

Installation:
    pip install sardis-adk

Usage:
    from tools.sardis import sardis_pay, sardis_check_balance

    # Use in an ADK agent
    agent = Agent(
        model="gemini-2.0-flash",
        name="payment_agent",
        tools=[sardis_pay, sardis_check_balance],
    )
"""

from sardis_adk.tools import (
    sardis_pay,
    sardis_check_balance,
    sardis_check_policy,
    SARDIS_TOOLS,
)

__all__ = [
    "sardis_pay",
    "sardis_check_balance",
    "sardis_check_policy",
    "SARDIS_TOOLS",
]
