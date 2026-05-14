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

"""Unit tests for Spraay batch payment tools."""

import os
import unittest
from unittest.mock import MagicMock, patch

from google.adk_community.tools.spraay import spraay_tools as spraay_module
from google.adk_community.tools.spraay.constants import (
    BASE_CHAIN_ID,
    MAX_RECIPIENTS,
    SPRAAY_CONTRACT_ADDRESS,
    SPRAAY_FEE_BPS,
)
from google.adk_community.tools.spraay.spraay_tools import (
    _calculate_fee,
    _validate_recipients,
    spraay_batch_eth,
    spraay_batch_eth_variable,
    spraay_batch_token,
    spraay_batch_token_variable,
)

# Valid test addresses (checksummed)
ADDR_1 = "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD1e"
ADDR_2 = "0xAb5801a7D398351b8bE11C439e05C5b3259aeC9B"
ADDR_3 = "0xCA35b7d915458EF540aDe6068dFe2F44E8fa733c"
TOKEN_ADDR = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # USDC on Base


def _make_mock_w3():
    """Create a mock Web3 instance with chain_id set to Base."""
    mock_w3 = MagicMock()
    mock_w3.eth.chain_id = BASE_CHAIN_ID
    return mock_w3


def _make_mock_web3_module():
    """Create a mock web3 module with Web3 class."""
    mock_web3_mod = MagicMock()
    mock_web3_mod.Web3.is_address.return_value = True
    mock_web3_mod.Web3.to_checksum_address.side_effect = lambda x: x
    return mock_web3_mod


class TestValidateRecipients(unittest.TestCase):
    """Tests for recipient address validation."""

    def test_empty_list(self):
        """Empty recipient list should raise ValueError."""
        with self.assertRaises(ValueError):
            _validate_recipients([])

    def test_too_many_recipients(self):
        """More than MAX_RECIPIENTS should raise ValueError."""
        addresses = [f"0x{'0' * 39}{i:01x}" for i in range(MAX_RECIPIENTS + 1)]
        with self.assertRaises(ValueError):
            _validate_recipients(addresses)

    def test_valid_addresses(self):
        """Valid addresses should be checksummed and returned."""
        import sys

        mock_web3_mod = _make_mock_web3_module()
        with patch.dict(sys.modules, {"web3": mock_web3_mod}):
            result = _validate_recipients([ADDR_1, ADDR_2])
            self.assertEqual(len(result), 2)

    def test_invalid_address(self):
        """Invalid address should raise ValueError."""
        import sys

        mock_web3_mod = MagicMock()
        mock_web3_mod.Web3.is_address.return_value = False
        with patch.dict(sys.modules, {"web3": mock_web3_mod}):
            with self.assertRaises(ValueError):
                _validate_recipients(["not_an_address"])


class TestCalculateFee(unittest.TestCase):
    """Tests for fee calculation."""

    def test_fee_calculation(self):
        """Fee should be 0.3% (30 basis points)."""
        total = 10000
        fee = _calculate_fee(total)
        self.assertEqual(fee, (total * SPRAAY_FEE_BPS) // 10000)

    def test_zero_amount(self):
        """Zero amount should produce zero fee."""
        self.assertEqual(_calculate_fee(0), 0)

    def test_small_amount(self):
        """Small amounts should still produce valid fee."""
        fee = _calculate_fee(100)
        self.assertIsInstance(fee, int)
        self.assertGreaterEqual(fee, 0)


class TestSpraayBatchEth(unittest.TestCase):
    """Tests for spraay_batch_eth function."""

    @patch.object(spraay_module, "_get_account")
    @patch.object(spraay_module, "_get_web3")
    def test_missing_private_key(self, mock_web3, mock_account):
        """Should return error if SPRAAY_PRIVATE_KEY is not set."""
        mock_w3 = _make_mock_w3()
        mock_web3.return_value = mock_w3
        mock_account.side_effect = ValueError(
            "SPRAAY_PRIVATE_KEY environment variable is required."
        )

        result = spraay_batch_eth([ADDR_1], "0.01")
        self.assertEqual(result["status"], "error")
        self.assertIn("SPRAAY_PRIVATE_KEY", result["error"])

    @patch.object(spraay_module, "_validate_recipients")
    @patch.object(spraay_module, "_get_account")
    @patch.object(spraay_module, "_get_web3")
    def test_zero_amount_returns_error(self, mock_web3, mock_account, mock_validate):
        """Zero ETH amount should return error."""
        mock_w3 = _make_mock_w3()
        mock_w3.to_wei.return_value = 0
        mock_web3.return_value = mock_w3
        mock_account.return_value = MagicMock()
        mock_validate.return_value = [ADDR_1]

        result = spraay_batch_eth([ADDR_1], "0")
        self.assertEqual(result["status"], "error")
        self.assertIn("greater than 0", result["error"])


class TestSpraayBatchEthVariable(unittest.TestCase):
    """Tests for spraay_batch_eth_variable function."""

    @patch.object(spraay_module, "_validate_recipients")
    @patch.object(spraay_module, "_get_account")
    @patch.object(spraay_module, "_get_web3")
    def test_mismatched_lengths(self, mock_web3, mock_account, mock_validate):
        """Recipients and amounts must have same length."""
        mock_w3 = _make_mock_w3()
        mock_w3.to_wei.side_effect = lambda x, _: int(float(str(x)) * 10**18)
        mock_web3.return_value = mock_w3
        mock_account.return_value = MagicMock()
        mock_validate.return_value = [ADDR_1, ADDR_2]

        result = spraay_batch_eth_variable(
            [ADDR_1, ADDR_2], ["0.1"]
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("must match", result["error"])


class TestSpraayBatchToken(unittest.TestCase):
    """Tests for spraay_batch_token function."""

    @patch.object(spraay_module, "_get_account")
    @patch.object(spraay_module, "_get_web3")
    def test_missing_private_key(self, mock_web3, mock_account):
        """Should return error if SPRAAY_PRIVATE_KEY is not set."""
        mock_w3 = _make_mock_w3()
        mock_web3.return_value = mock_w3
        mock_account.side_effect = ValueError(
            "SPRAAY_PRIVATE_KEY environment variable is required."
        )

        result = spraay_batch_token(TOKEN_ADDR, [ADDR_1], "10")
        self.assertEqual(result["status"], "error")
        self.assertIn("SPRAAY_PRIVATE_KEY", result["error"])


class TestSpraayBatchTokenVariable(unittest.TestCase):
    """Tests for spraay_batch_token_variable function."""

    @patch.object(spraay_module, "_get_account")
    @patch.object(spraay_module, "_get_web3")
    def test_missing_private_key(self, mock_web3, mock_account):
        """Should return error if SPRAAY_PRIVATE_KEY is not set."""
        mock_w3 = _make_mock_w3()
        mock_web3.return_value = mock_w3
        mock_account.side_effect = ValueError(
            "SPRAAY_PRIVATE_KEY environment variable is required."
        )

        result = spraay_batch_token_variable(
            TOKEN_ADDR, [ADDR_1], ["10"]
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("SPRAAY_PRIVATE_KEY", result["error"])


class TestConstants(unittest.TestCase):
    """Tests for Spraay constants."""

    def test_contract_address_format(self):
        """Contract address should be valid checksum format."""
        self.assertTrue(SPRAAY_CONTRACT_ADDRESS.startswith("0x"))
        self.assertEqual(len(SPRAAY_CONTRACT_ADDRESS), 42)

    def test_max_recipients(self):
        """Max recipients should be 200."""
        self.assertEqual(MAX_RECIPIENTS, 200)

    def test_fee_bps(self):
        """Fee should be 30 basis points (0.3%)."""
        self.assertEqual(SPRAAY_FEE_BPS, 30)


if __name__ == "__main__":
    unittest.main()
