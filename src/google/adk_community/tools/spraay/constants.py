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

# Base Mainnet chain configuration
BASE_CHAIN_ID = 8453
BASE_RPC_URL = "https://mainnet.base.org"

# Protocol fee: 0.3%
SPRAAY_FEE_BPS = 30  # basis points

# Maximum recipients per transaction
MAX_RECIPIENTS = 200

# ERC-20 max approval
MAX_UINT256 = 2**256 - 1

# Spraay contract ABI (relevant functions only)
SPRAAY_ABI = [
    {
        "inputs": [
            {"internalType": "address[]", "name": "_recipients", "type": "address[]"},
            {"internalType": "uint256", "name": "_amount", "type": "uint256"},
        ],
        "name": "spraayETH",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "_token", "type": "address"},
            {"internalType": "address[]", "name": "_recipients", "type": "address[]"},
            {"internalType": "uint256", "name": "_amount", "type": "uint256"},
        ],
        "name": "spraayToken",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address[]", "name": "_recipients", "type": "address[]"},
            {"internalType": "uint256[]", "name": "_amounts", "type": "uint256[]"},
        ],
        "name": "spraayETHVariable",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "address", "name": "_token", "type": "address"},
            {"internalType": "address[]", "name": "_recipients", "type": "address[]"},
            {"internalType": "uint256[]", "name": "_amounts", "type": "uint256[]"},
        ],
        "name": "spraayTokenVariable",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
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
