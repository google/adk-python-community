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

"""Spraay batch payment tools for Google ADK agents.

Spraay (https://spraay.app) enables AI agents to batch-send ETH or ERC-20
tokens to up to 200 recipients in a single transaction on Base, with ~80%
gas savings compared to individual transfers.

Tools:
    spraay_batch_eth: Send equal ETH to multiple recipients.
    spraay_batch_token: Send equal ERC-20 tokens to multiple recipients.
    spraay_batch_eth_variable: Send variable ETH amounts per recipient.
    spraay_batch_token_variable: Send variable token amounts per recipient.

Usage:
    from google.adk_community.tools.spraay import (
        spraay_batch_eth,
        spraay_batch_token,
        spraay_batch_eth_variable,
        spraay_batch_token_variable,
    )
    from google.adk.agents import Agent

    agent = Agent(
        name="payment_agent",
        model="gemini-2.5-flash",
        tools=[
            spraay_batch_eth,
            spraay_batch_token,
            spraay_batch_eth_variable,
            spraay_batch_token_variable,
        ],
    )
"""

from google.adk_community.tools.spraay.spraay_tools import (
    spraay_batch_eth,
    spraay_batch_eth_variable,
    spraay_batch_token,
    spraay_batch_token_variable,
)

__all__ = [
    "spraay_batch_eth",
    "spraay_batch_token",
    "spraay_batch_eth_variable",
    "spraay_batch_token_variable",
]
