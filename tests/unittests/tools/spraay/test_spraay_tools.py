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

from google.adk_community.tools.spraay.constants import (
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


class TestValidateRecipients(unittest.TestCase):
    """Tests for recipient address validation."""

    @patch("google.adk_community.tools.spraay.spraay_tools.Web3")
    def test_valid_addresses(self, mock_web3_class):
        """Valid addresses should be checksummed and returned."""
        mock_web3_class.is_address.return_value = True
        mock_web3_class.to_checksum_address.side_effect = lambda x: x
        result = _validate_recipients([ADDR_1, ADDR_2])
        self.assertEqual(len(result), 2)

    def test_empty_list(self):
        """Empty recipient list should raise ValueError."""
        with self.assertRaises(ValueError):
            _validate_recipients([])

    @patch("google.adk_community.tools.spraay.spraay_tools.Web3")
    def test_too_many_recipients(self, mock_web3_class):
        """More than MAX_RECIPIENTS should raise ValueError."""
        mock_web3_class.is_address.return_value = True
        addresses = [f"0x{'0' * 39}{i:01x}" for i in range(MAX_RECIPIENTS + 1)]
        with self.assertRaises(ValueError):
            _validate_recipients(addresses)

    @patch("google.adk_community.tools.spraay.spraay_tools.Web3")
    def test_invalid_address(self, mock_web3_class):
        """Invalid address should raise ValueError."""
        mock_web3_class.is_address.return_value = False
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

    def test_missing_private_key(self):
        """Should return error if SPRAAY_PRIVATE_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = spraay_batch_eth([ADDR_1], "0.01")
            self.assertEqual(result["status"], "error")
            self.assertIn("SPRAAY_PRIVATE_KEY", result["error"])

    @patch("google.adk_community.tools.spraay.spraay_tools._get_web3")
    @patch("google.adk_community.tools.spraay.spraay_tools._get_account")
    def test_zero_amount_returns_error(self, mock_account, mock_web3):
        """Zero ETH amount should return error."""
        mock_w3 = MagicMock()
        mock_w3.to_wei.return_value = 0
        mock_web3.return_value = mock_w3
        mock_account.return_value = MagicMock()

        result = spraay_batch_eth([ADDR_1], "0")
        self.assertEqual(result["status"], "error")
        self.assertIn("greater than 0", result["error"])


class TestSpraayBatchEthVariable(unittest.TestCase):
    """Tests for spraay_batch_eth_variable function."""

    def test_mismatched_lengths(self):
        """Recipients and amounts must have same length."""
        with patch.dict(os.environ, {"SPRAAY_PRIVATE_KEY": "0x" + "a" * 64}):
            with patch(
                "google.adk_community.tools.spraay.spraay_tools._get_web3"
            ) as mock_web3:
                mock_w3 = MagicMock()
                mock_w3.to_wei.side_effect = lambda x, _: int(float(x) * 10**18)
                mock_web3.return_value = mock_w3

                with patch(
                    "google.adk_community.tools.spraay.spraay_tools._validate_recipients"
                ) as mock_validate:
                    mock_validate.return_value = [ADDR_1, ADDR_2]

                    result = spraay_batch_eth_variable(
                        [ADDR_1, ADDR_2], ["0.1"]
                    )
                    self.assertEqual(result["status"], "error")
                    self.assertIn("must match", result["error"])


class TestSpraayBatchToken(unittest.TestCase):
    """Tests for spraay_batch_token function."""

    def test_missing_private_key(self):
        """Should return error if SPRAAY_PRIVATE_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = spraay_batch_token(TOKEN_ADDR, [ADDR_1], "10")
            self.assertEqual(result["status"], "error")
            self.assertIn("SPRAAY_PRIVATE_KEY", result["error"])


class TestSpraayBatchTokenVariable(unittest.TestCase):
    """Tests for spraay_batch_token_variable function."""

    def test_missing_private_key(self):
        """Should return error if SPRAAY_PRIVATE_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
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
