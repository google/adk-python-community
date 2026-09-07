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

"""Spraay batch payment tool functions for Google ADK.

These functions are designed to be used as ADK FunctionTools. When assigned
to an agent's tools list, the ADK framework automatically wraps them and
generates schemas from the function signatures and docstrings.

Environment Variables:
    SPRAAY_RPC_URL: Base RPC endpoint (default: https://mainnet.base.org)
    SPRAAY_PRIVATE_KEY: Private key for signing transactions (required)
    SPRAAY_CONTRACT_ADDRESS: Override default Spraay contract address

Dependencies:
    pip install web3
"""

import logging
import os
from decimal import Decimal
from typing import Optional

from google.adk_community.tools.spraay.constants import (
    BASE_CHAIN_ID,
    BASE_RPC_URL,
    ERC20_APPROVE_ABI,
    MAX_FEE_BPS,
    MAX_RECIPIENTS,
    MAX_UINT256,
    SPRAAY_ABI,
    SPRAAY_CONTRACT_ADDRESS,
    SPRAAY_FEE_BPS,
    ZERO_ADDRESS,
)

logger = logging.getLogger(__name__)


def _get_web3():
    """Initialize Web3 connection to Base."""
    try:
        from web3 import Web3
    except ImportError:
        raise ImportError(
            "web3 is required for Spraay tools. Install with: pip install web3"
        )

    rpc_url = os.environ.get("SPRAAY_RPC_URL", BASE_RPC_URL)
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise ConnectionError(f"Failed to connect to Base RPC at {rpc_url}")
    return w3


def _get_account():
    """Get the signing account from environment."""
    try:
        from web3 import Account
    except ImportError:
        raise ImportError(
            "web3 is required for Spraay tools. Install with: pip install web3"
        )

    private_key = os.environ.get("SPRAAY_PRIVATE_KEY")
    if not private_key:
        raise ValueError(
            "SPRAAY_PRIVATE_KEY environment variable is required. "
            "Set it to the private key of the wallet that will send payments."
        )
    return Account.from_key(private_key)


def _get_contract_address() -> str:
    """Get the Spraay contract address (allows override via env)."""
    return os.environ.get("SPRAAY_CONTRACT_ADDRESS", SPRAAY_CONTRACT_ADDRESS)


def _validate_recipients(recipients: list[str]) -> list[str]:
    """Validate and checksum recipient addresses."""
    if not recipients:
        raise ValueError("Recipients list cannot be empty.")
    if len(recipients) > MAX_RECIPIENTS:
        raise ValueError(
            f"Maximum {MAX_RECIPIENTS} recipients per transaction. "
            f"Got {len(recipients)}."
        )

    try:
        from web3 import Web3
    except ImportError:
        raise ImportError(
            "web3 is required for Spraay tools. Install with: pip install web3"
        )

    checksummed = []
    for addr in recipients:
        if not Web3.is_address(addr):
            raise ValueError(f"Invalid Ethereum address: {addr}")
        checksummed.append(Web3.to_checksum_address(addr))
    return checksummed


def _get_fee_bps(spraay_contract) -> int:
    """Read the current protocol fee (basis points) from the contract.

    The fee is owner-adjustable on-chain (capped at MAX_FEE_BPS), so it is
    read live rather than hardcoded. Falls back to SPRAAY_FEE_BPS if the
    read fails or returns an implausible value.
    """
    try:
        fee_bps = spraay_contract.functions.feeBps().call()
        if isinstance(fee_bps, int) and 0 <= fee_bps <= MAX_FEE_BPS:
            return fee_bps
    except Exception:  # pragma: no cover - network dependent
        logger.warning(
            "Could not read feeBps() from contract; falling back to %s bps.",
            SPRAAY_FEE_BPS,
        )
    return SPRAAY_FEE_BPS


def _calculate_fee(total_wei: int, fee_bps: int = SPRAAY_FEE_BPS) -> int:
    """Calculate the Spraay protocol fee for a total amount."""
    return (total_wei * fee_bps) // 10000


def _verify_chain_id(w3) -> None:
    """Verify the connected chain ID matches Base.

    Raises:
        ValueError: If chain ID doesn't match BASE_CHAIN_ID.
    """
    connected_chain_id = w3.eth.chain_id
    if connected_chain_id != BASE_CHAIN_ID:
        raise ValueError(
            f"Chain ID mismatch: connected to {connected_chain_id}, "
            f"expected {BASE_CHAIN_ID} (Base). "
            "Check your RPC configuration."
        )


def spraay_batch_eth(
    recipients: list[str],
    amount_per_recipient_eth: str,
) -> dict:
    """Send equal amounts of ETH to multiple recipients in a single transaction on Base.

    Uses the Spraay protocol to batch-send ETH, saving ~80% on gas compared
    to individual transfers. Supports up to 200 recipients per transaction.
    A 0.3% protocol fee is applied.

    Args:
        recipients: List of Ethereum addresses to receive ETH.
            Maximum 200 addresses per transaction.
        amount_per_recipient_eth: Amount of ETH each recipient will receive,
            as a decimal string (e.g. "0.01" for 0.01 ETH per recipient).

    Returns:
        dict with keys:
            - status: "success" or "error"
            - tx_hash: Transaction hash (on success)
            - recipients_count: Number of recipients
            - amount_per_recipient: ETH amount per recipient
            - total_eth: Total ETH sent (including fee)
            - error: Error message (on failure)
    """
    try:
        w3 = _get_web3()
        account = _get_account()
        contract_address = _get_contract_address()

        _verify_chain_id(w3)

        checksummed = _validate_recipients(recipients)
        amount_wei = w3.to_wei(Decimal(amount_per_recipient_eth), "ether")

        if amount_wei <= 0:
            return {"status": "error", "error": "Amount must be greater than 0."}

        contract = w3.eth.contract(
            address=w3.to_checksum_address(contract_address),
            abi=SPRAAY_ABI,
        )

        total_wei = amount_wei * len(checksummed)
        fee_wei = _calculate_fee(total_wei, _get_fee_bps(contract))
        total_with_fee = total_wei + fee_wei

        # sprayEqual with token=address(0) is the contract's native-ETH
        # equal-amount path. msg.value must cover total + fee.
        tx = contract.functions.sprayEqual(
            ZERO_ADDRESS, checksummed, amount_wei
        ).build_transaction(
            {
                "from": account.address,
                "value": total_with_fee,
                "nonce": w3.eth.get_transaction_count(account.address, "pending"),
                "chainId": BASE_CHAIN_ID,
            }
        )

        # build_transaction estimates gas (no "gas" placeholder: some RPCs,
        # including the default Base RPC, treat an explicit 0 as a hard cap
        # and reject estimation). Add a 10% buffer to the estimate.
        tx["gas"] = int(tx["gas"] * 1.1)
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

        return {
            "status": "success",
            "tx_hash": tx_hash.hex(),
            "recipients_count": len(checksummed),
            "amount_per_recipient": amount_per_recipient_eth,
            "total_eth": str(w3.from_wei(total_with_fee, "ether")),
        }

    except Exception as e:
        logger.error("spraay_batch_eth failed: %s", str(e))
        return {"status": "error", "error": str(e)}


def spraay_batch_token(
    token_address: str,
    recipients: list[str],
    amount_per_recipient: str,
    token_decimals: int = 18,
) -> dict:
    """Send equal amounts of an ERC-20 token to multiple recipients on Base.

    Uses the Spraay protocol to batch-send tokens, saving ~80% on gas.
    Automatically handles token approval if needed. Supports up to 200
    recipients per transaction. A 0.3% protocol fee is applied.

    Args:
        token_address: The ERC-20 token contract address (e.g. USDC on Base).
        recipients: List of Ethereum addresses to receive tokens.
            Maximum 200 addresses per transaction.
        amount_per_recipient: Amount of tokens each recipient receives,
            as a decimal string (e.g. "10.5" for 10.5 tokens each).
        token_decimals: Number of decimals for the token (default 18).
            USDC uses 6 decimals. Check the token contract if unsure.

    Returns:
        dict with keys:
            - status: "success" or "error"
            - tx_hash: Transaction hash (on success)
            - approval_tx_hash: Approval tx hash (if approval was needed)
            - recipients_count: Number of recipients
            - amount_per_recipient: Token amount per recipient
            - token_address: Token contract address
            - error: Error message (on failure)
    """
    try:
        w3 = _get_web3()
        account = _get_account()
        contract_address = _get_contract_address()

        _verify_chain_id(w3)

        checksummed = _validate_recipients(recipients)
        token_addr = w3.to_checksum_address(token_address)
        spraay_addr = w3.to_checksum_address(contract_address)

        # Use Decimal for precise token amount conversion
        amount_units = int(
            Decimal(amount_per_recipient) * Decimal(10**token_decimals)
        )
        if amount_units <= 0:
            return {"status": "error", "error": "Amount must be greater than 0."}

        spraay_contract = w3.eth.contract(address=spraay_addr, abi=SPRAAY_ABI)

        total_units = amount_units * len(checksummed)
        fee_units = _calculate_fee(total_units, _get_fee_bps(spraay_contract))
        total_with_fee = total_units + fee_units

        result = {"approval_tx_hash": None}

        # Check and handle token approval
        token_contract = w3.eth.contract(address=token_addr, abi=ERC20_APPROVE_ABI)
        allowance = token_contract.functions.allowance(
            account.address, spraay_addr
        ).call()

        if allowance < total_with_fee:
            approve_tx = token_contract.functions.approve(
                spraay_addr, MAX_UINT256
            ).build_transaction(
                {
                    "from": account.address,
                    "nonce": w3.eth.get_transaction_count(account.address, "pending"),
                    "chainId": BASE_CHAIN_ID,
                }
            )
            # build_transaction estimated gas; add a 10% buffer.
            approve_tx["gas"] = int(approve_tx["gas"] * 1.1)
            signed_approve = account.sign_transaction(approve_tx)
            approve_hash = w3.eth.send_raw_transaction(
                signed_approve.raw_transaction
            )
            w3.eth.wait_for_transaction_receipt(approve_hash, timeout=120)
            result["approval_tx_hash"] = approve_hash.hex()

        # Execute batch transfer. sprayEqual with a token address is the
        # contract's ERC-20 equal-amount path; the contract pulls
        # total + fee via transferFrom, so the allowance above covers it.
        nonce = w3.eth.get_transaction_count(account.address, "pending")

        tx = spraay_contract.functions.sprayEqual(
            token_addr, checksummed, amount_units
        ).build_transaction(
            {
                "from": account.address,
                "nonce": nonce,
                "chainId": BASE_CHAIN_ID,
            }
        )
        # build_transaction estimated gas; add a 10% buffer.
        tx["gas"] = int(tx["gas"] * 1.1)
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

        result.update(
            {
                "status": "success",
                "tx_hash": tx_hash.hex(),
                "recipients_count": len(checksummed),
                "amount_per_recipient": amount_per_recipient,
                "token_address": token_address,
            }
        )
        return result

    except Exception as e:
        logger.error("spraay_batch_token failed: %s", str(e))
        return {"status": "error", "error": str(e)}


def spraay_batch_eth_variable(
    recipients: list[str],
    amounts_eth: list[str],
) -> dict:
    """Send variable amounts of ETH to multiple recipients on Base.

    Each recipient receives a different amount of ETH, useful for payroll,
    bounties, or reward distributions where amounts differ per person.
    Supports up to 200 recipients. A 0.3% protocol fee is applied.

    Args:
        recipients: List of Ethereum addresses to receive ETH.
            Maximum 200 addresses per transaction.
        amounts_eth: List of ETH amounts as decimal strings, one per recipient.
            Must be the same length as recipients.
            Example: ["0.1", "0.25", "0.05"] for three recipients.

    Returns:
        dict with keys:
            - status: "success" or "error"
            - tx_hash: Transaction hash (on success)
            - recipients_count: Number of recipients
            - total_eth: Total ETH sent (including fee)
            - error: Error message (on failure)
    """
    try:
        w3 = _get_web3()
        account = _get_account()
        contract_address = _get_contract_address()

        _verify_chain_id(w3)

        checksummed = _validate_recipients(recipients)

        if len(amounts_eth) != len(checksummed):
            return {
                "status": "error",
                "error": (
                    f"Recipients count ({len(checksummed)}) must match "
                    f"amounts count ({len(amounts_eth)})."
                ),
            }

        amounts_wei = [w3.to_wei(Decimal(a), "ether") for a in amounts_eth]
        if any(a <= 0 for a in amounts_wei):
            return {"status": "error", "error": "All amounts must be greater than 0."}

        contract = w3.eth.contract(
            address=w3.to_checksum_address(contract_address),
            abi=SPRAAY_ABI,
        )

        total_wei = sum(amounts_wei)
        fee_wei = _calculate_fee(total_wei, _get_fee_bps(contract))
        total_with_fee = total_wei + fee_wei

        # sprayETH takes an array of Recipient structs: (address, amount).
        recipient_structs = list(zip(checksummed, amounts_wei))

        tx = contract.functions.sprayETH(
            recipient_structs
        ).build_transaction(
            {
                "from": account.address,
                "value": total_with_fee,
                "nonce": w3.eth.get_transaction_count(account.address, "pending"),
                "chainId": BASE_CHAIN_ID,
            }
        )
        # build_transaction estimated gas; add a 10% buffer.
        tx["gas"] = int(tx["gas"] * 1.1)
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

        return {
            "status": "success",
            "tx_hash": tx_hash.hex(),
            "recipients_count": len(checksummed),
            "total_eth": str(w3.from_wei(total_with_fee, "ether")),
        }

    except Exception as e:
        logger.error("spraay_batch_eth_variable failed: %s", str(e))
        return {"status": "error", "error": str(e)}


def spraay_batch_token_variable(
    token_address: str,
    recipients: list[str],
    amounts: list[str],
    token_decimals: int = 18,
) -> dict:
    """Send variable amounts of an ERC-20 token to multiple recipients on Base.

    Each recipient receives a different token amount. Automatically handles
    token approval. Supports up to 200 recipients. A 0.3% protocol fee
    is applied.

    Args:
        token_address: The ERC-20 token contract address.
        recipients: List of Ethereum addresses to receive tokens.
            Maximum 200 addresses per transaction.
        amounts: List of token amounts as decimal strings, one per recipient.
            Must be the same length as recipients.
            Example: ["100", "250.5", "75"] for three recipients.
        token_decimals: Number of decimals for the token (default 18).

    Returns:
        dict with keys:
            - status: "success" or "error"
            - tx_hash: Transaction hash (on success)
            - approval_tx_hash: Approval tx hash (if approval was needed)
            - recipients_count: Number of recipients
            - token_address: Token contract address
            - error: Error message (on failure)
    """
    try:
        w3 = _get_web3()
        account = _get_account()
        contract_address = _get_contract_address()

        _verify_chain_id(w3)

        checksummed = _validate_recipients(recipients)
        token_addr = w3.to_checksum_address(token_address)
        spraay_addr = w3.to_checksum_address(contract_address)

        if len(amounts) != len(checksummed):
            return {
                "status": "error",
                "error": (
                    f"Recipients count ({len(checksummed)}) must match "
                    f"amounts count ({len(amounts)})."
                ),
            }

        # Use Decimal for precise token amount conversion
        amounts_units = [
            int(Decimal(a) * Decimal(10**token_decimals)) for a in amounts
        ]
        if any(a <= 0 for a in amounts_units):
            return {"status": "error", "error": "All amounts must be greater than 0."}

        spraay_contract = w3.eth.contract(address=spraay_addr, abi=SPRAAY_ABI)

        total_units = sum(amounts_units)
        fee_units = _calculate_fee(total_units, _get_fee_bps(spraay_contract))
        total_with_fee = total_units + fee_units

        result = {"approval_tx_hash": None}

        # Check and handle token approval
        token_contract = w3.eth.contract(address=token_addr, abi=ERC20_APPROVE_ABI)
        allowance = token_contract.functions.allowance(
            account.address, spraay_addr
        ).call()

        if allowance < total_with_fee:
            approve_tx = token_contract.functions.approve(
                spraay_addr, MAX_UINT256
            ).build_transaction(
                {
                    "from": account.address,
                    "nonce": w3.eth.get_transaction_count(account.address, "pending"),
                    "chainId": BASE_CHAIN_ID,
                }
            )
            # build_transaction estimated gas; add a 10% buffer.
            approve_tx["gas"] = int(approve_tx["gas"] * 1.1)
            signed_approve = account.sign_transaction(approve_tx)
            approve_hash = w3.eth.send_raw_transaction(
                signed_approve.raw_transaction
            )
            w3.eth.wait_for_transaction_receipt(approve_hash, timeout=120)
            result["approval_tx_hash"] = approve_hash.hex()

        # Execute batch transfer. sprayToken takes an array of Recipient
        # structs: (address, amount). The contract pulls total + fee via
        # transferFrom, so the allowance above covers it.
        nonce = w3.eth.get_transaction_count(account.address, "pending")

        recipient_structs = list(zip(checksummed, amounts_units))

        tx = spraay_contract.functions.sprayToken(
            token_addr, recipient_structs
        ).build_transaction(
            {
                "from": account.address,
                "nonce": nonce,
                "chainId": BASE_CHAIN_ID,
            }
        )
        # build_transaction estimated gas; add a 10% buffer.
        tx["gas"] = int(tx["gas"] * 1.1)
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

        result.update(
            {
                "status": "success",
                "tx_hash": tx_hash.hex(),
                "recipients_count": len(checksummed),
                "token_address": token_address,
            }
        )
        return result

    except Exception as e:
        logger.error("spraay_batch_token_variable failed: %s", str(e))
        return {"status": "error", "error": str(e)}
