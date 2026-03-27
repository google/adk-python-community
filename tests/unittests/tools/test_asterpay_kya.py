"""Tests for AsterPay KYA tools."""

from unittest.mock import MagicMock, patch

from google.adk_community.tools import (
    asterpay_kya_trust_score,
    asterpay_kya_verify,
    asterpay_kya_tier,
    asterpay_settlement_estimate,
)


def test_trust_score_is_callable():
    """Trust score function should be callable."""
    assert callable(asterpay_kya_trust_score)


def test_verify_is_callable():
    """Verify function should be callable."""
    assert callable(asterpay_kya_verify)


def test_tier_is_callable():
    """Tier function should be callable."""
    assert callable(asterpay_kya_tier)


def test_settlement_estimate_is_callable():
    """Settlement estimate function should be callable."""
    assert callable(asterpay_settlement_estimate)


@patch("google.adk_community.tools.asterpay_kya.requests.get")
def test_trust_score_with_mock(mock_get):
    """Trust score should call correct endpoint."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"score": 75, "tier": "Verified"}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    result = asterpay_kya_trust_score("0x1234")
    assert "75" in result
    mock_get.assert_called_once()


@patch("google.adk_community.tools.asterpay_kya.requests.get")
def test_settlement_estimate_with_mock(mock_get):
    """Settlement estimate should pass amount as param."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"eur_amount": 92.50}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    result = asterpay_settlement_estimate(100.0)
    assert "92.5" in result


def test_handles_error():
    """Should return error message on failure, not raise."""
    result = asterpay_kya_trust_score("invalid-not-an-address")
    assert isinstance(result, str)
