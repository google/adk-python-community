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

"""Constants for Spraay batch payment tools."""

# Spraay contract on Base Mainnet
SPRAAY_CONTRACT_ADDRESS = "0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC"

# Sentinel address: sprayEqual uses address(0) as the token to send native ETH.
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

# Base Mainnet chain configuration
BASE_CHAIN_ID = 8453
BASE_RPC_URL = "https://mainnet.base.org"

# Protocol fee fallback in basis points (30 = 0.3%). The contract's fee is
# mutable by its owner (capped on-chain at 5%), so the live value is read
# from feeBps() at call time; this constant is only used if that read fails.
SPRAAY_FEE_BPS = 30

# Maximum recipients per transaction (mirrors the contract's MAX_RECIPIENTS)
MAX_RECIPIENTS = 200

# On-chain fee cap in basis points (mirrors the contract's MAX_FEE_BPS = 5%)
MAX_FEE_BPS = 500

# ERC-20 max approval
MAX_UINT256 = 2**256 - 1

# SprayContract ABI (relevant functions only).
# Copied verbatim from the verified source of
# 0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC on Base (chain 8453), as
# published on Sourcify. Do not hand-edit; regenerate from the verified ABI.
SPRAAY_ABI = [
    {
        "name": "sprayETH",
        "type": "function",
        "inputs": [
            {
                "name": "recipients",
                "type": "tuple[]",
                "components": [
                    {
                        "name": "recipient",
                        "type": "address",
                        "internalType": "address payable"
                    },
                    {
                        "name": "amount",
                        "type": "uint256",
                        "internalType": "uint256"
                    }
                ],
                "internalType": "struct SprayContract.Recipient[]"
            }
        ],
        "outputs": [],
        "stateMutability": "payable"
    },
    {
        "name": "sprayToken",
        "type": "function",
        "inputs": [
            {
                "name": "token",
                "type": "address",
                "internalType": "address"
            },
            {
                "name": "recipients",
                "type": "tuple[]",
                "components": [
                    {
                        "name": "recipient",
                        "type": "address",
                        "internalType": "address payable"
                    },
                    {
                        "name": "amount",
                        "type": "uint256",
                        "internalType": "uint256"
                    }
                ],
                "internalType": "struct SprayContract.Recipient[]"
            }
        ],
        "outputs": [],
        "stateMutability": "nonpayable"
    },
    {
        "name": "sprayEqual",
        "type": "function",
        "inputs": [
            {
                "name": "token",
                "type": "address",
                "internalType": "address"
            },
            {
                "name": "recipients",
                "type": "address[]",
                "internalType": "address payable[]"
            },
            {
                "name": "amountPerRecipient",
                "type": "uint256",
                "internalType": "uint256"
            }
        ],
        "outputs": [],
        "stateMutability": "payable"
    },
    {
        "name": "feeBps",
        "type": "function",
        "inputs": [],
        "outputs": [
            {
                "name": "",
                "type": "uint256",
                "internalType": "uint256"
            }
        ],
        "stateMutability": "view"
    },
    {
        "name": "calculateTotalCost",
        "type": "function",
        "inputs": [
            {
                "name": "totalAmount",
                "type": "uint256",
                "internalType": "uint256"
            }
        ],
        "outputs": [
            {
                "name": "",
                "type": "uint256",
                "internalType": "uint256"
            }
        ],
        "stateMutability": "view"
    }
]

# ERC-20 approve ABI
ERC20_APPROVE_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "spender", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "owner", "type": "address"},
            {"internalType": "address", "name": "spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]
