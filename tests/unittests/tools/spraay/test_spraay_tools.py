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
    MAX_FEE_BPS,
    MAX_RECIPIENTS,
    SPRAAY_ABI,
    SPRAAY_CONTRACT_ADDRESS,
    SPRAAY_FEE_BPS,
    ZERO_ADDRESS,
)
from google.adk_community.tools.spraay.spraay_tools import (
    _calculate_fee,
    _get_fee_bps,
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


class TestAbiMatchesDeployedContract(unittest.TestCase):
    """Regression tests pinning the ABI to the verified deployed contract.

    The deployed SprayContract (0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC
    on Base) exposes sprayETH/sprayToken (Recipient[] structs) and
    sprayEqual. These tests fail if the ABI drifts from the verified
    on-chain interface again.
    """

    def _fn(self, name):
        matches = [e for e in SPRAAY_ABI if e.get("name") == name]
        self.assertEqual(len(matches), 1, f"expected exactly one {name} in ABI")
        return matches[0]

    def test_abi_function_names(self):
        """ABI must contain exactly the deployed payment + fee functions."""
        names = {e["name"] for e in SPRAAY_ABI if e.get("type") == "function"}
        self.assertEqual(
            names,
            {"sprayETH", "sprayToken", "sprayEqual", "feeBps",
             "calculateTotalCost"},
        )

    def test_spray_eth_takes_recipient_structs(self):
        """sprayETH takes a single Recipient[] (address, uint256) argument."""
        fn = self._fn("sprayETH")
        self.assertEqual(len(fn["inputs"]), 1)
        arg = fn["inputs"][0]
        self.assertEqual(arg["type"], "tuple[]")
        self.assertEqual(
            [(c["name"], c["type"]) for c in arg["components"]],
            [("recipient", "address"), ("amount", "uint256")],
        )
        self.assertEqual(fn["stateMutability"], "payable")

    def test_spray_token_takes_recipient_structs(self):
        """sprayToken takes (address token, Recipient[] recipients)."""
        fn = self._fn("sprayToken")
        self.assertEqual(
            [i["type"] for i in fn["inputs"]], ["address", "tuple[]"]
        )

    def test_spray_equal_signature(self):
        """sprayEqual takes (address, address[], uint256) and is payable."""
        fn = self._fn("sprayEqual")
        self.assertEqual(
            [i["type"] for i in fn["inputs"]],
            ["address", "address[]", "uint256"],
        )
        self.assertEqual(fn["stateMutability"], "payable")


class TestGetFeeBps(unittest.TestCase):
    """Tests for the live fee read with fallback."""

    def test_uses_onchain_value(self):
        """A plausible on-chain feeBps value should be used."""
        contract = MagicMock()
        contract.functions.feeBps.return_value.call.return_value = 25
        self.assertEqual(_get_fee_bps(contract), 25)

    def test_fallback_on_error(self):
        """RPC failure should fall back to SPRAAY_FEE_BPS."""
        contract = MagicMock()
        contract.functions.feeBps.return_value.call.side_effect = Exception(
            "rpc down"
        )
        self.assertEqual(_get_fee_bps(contract), SPRAAY_FEE_BPS)

    def test_fallback_on_implausible_value(self):
        """Values above the on-chain MAX_FEE_BPS cap should be rejected."""
        contract = MagicMock()
        contract.functions.feeBps.return_value.call.return_value = (
            MAX_FEE_BPS + 1
        )
        self.assertEqual(_get_fee_bps(contract), SPRAAY_FEE_BPS)


class TestCallConstruction(unittest.TestCase):
    """Tests that tools build calls against the deployed function names."""

    def _mock_w3(self):
        mock_w3 = _make_mock_w3()
        mock_w3.to_wei.side_effect = lambda x, _: int(float(str(x)) * 10**18)
        mock_w3.from_wei.side_effect = lambda x, _: x / 10**18
        mock_w3.to_checksum_address.side_effect = lambda x: x
        mock_w3.eth.get_transaction_count.return_value = 1
        mock_w3.eth.estimate_gas.return_value = 100_000
        tx_hash = MagicMock()
        tx_hash.hex.return_value = "0xabc"
        mock_w3.eth.send_raw_transaction.return_value = tx_hash
        return mock_w3

    def _contract(self, mock_w3):
        contract = MagicMock()
        contract.functions.feeBps.return_value.call.return_value = 30
        for fn in ("sprayEqual", "sprayETH", "sprayToken"):
            getattr(
                contract.functions, fn
            ).return_value.build_transaction.return_value = {"gas": 100_000}
        mock_w3.eth.contract.return_value = contract
        return contract

    @patch.object(spraay_module, "_validate_recipients")
    @patch.object(spraay_module, "_get_account")
    @patch.object(spraay_module, "_get_web3")
    def test_equal_eth_uses_spray_equal_with_zero_address(
        self, mock_web3, mock_account, mock_validate
    ):
        """Equal ETH sends must call sprayEqual(address(0), ...)."""
        mock_w3 = self._mock_w3()
        contract = self._contract(mock_w3)
        mock_web3.return_value = mock_w3
        mock_account.return_value = MagicMock()
        mock_validate.return_value = [ADDR_1, ADDR_2]

        result = spraay_batch_eth([ADDR_1, ADDR_2], "0.01")

        self.assertEqual(result["status"], "success")
        args = contract.functions.sprayEqual.call_args[0]
        self.assertEqual(args[0], ZERO_ADDRESS)
        self.assertEqual(args[1], [ADDR_1, ADDR_2])
        self.assertEqual(args[2], 10**16)

    @patch.object(spraay_module, "_validate_recipients")
    @patch.object(spraay_module, "_get_account")
    @patch.object(spraay_module, "_get_web3")
    def test_variable_eth_uses_spray_eth_structs(
        self, mock_web3, mock_account, mock_validate
    ):
        """Variable ETH sends must call sprayETH with (address, amount) structs."""
        mock_w3 = self._mock_w3()
        contract = self._contract(mock_w3)
        mock_web3.return_value = mock_w3
        mock_account.return_value = MagicMock()
        mock_validate.return_value = [ADDR_1, ADDR_2]

        result = spraay_batch_eth_variable([ADDR_1, ADDR_2], ["0.1", "0.25"])

        self.assertEqual(result["status"], "success")
        (structs,) = contract.functions.sprayETH.call_args[0]
        self.assertEqual(
            structs,
            [(ADDR_1, 10**17), (ADDR_2, 25 * 10**16)],
        )

    @patch.object(spraay_module, "_validate_recipients")
    @patch.object(spraay_module, "_get_account")
    @patch.object(spraay_module, "_get_web3")
    def test_variable_token_uses_spray_token_structs(
        self, mock_web3, mock_account, mock_validate
    ):
        """Variable token sends must call sprayToken with structs."""
        mock_w3 = self._mock_w3()
        contract = self._contract(mock_w3)
        # allowance already sufficient -> no approval tx
        contract.functions.allowance.return_value.call.return_value = 2**255
        mock_web3.return_value = mock_w3
        mock_account.return_value = MagicMock()
        mock_validate.return_value = [ADDR_1]

        result = spraay_batch_token_variable(
            TOKEN_ADDR, [ADDR_1], ["10"], token_decimals=6
        )

        self.assertEqual(result["status"], "success")
        token_arg, structs = contract.functions.sprayToken.call_args[0]
        self.assertEqual(token_arg, TOKEN_ADDR)
        self.assertEqual(structs, [(ADDR_1, 10_000_000)])


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
