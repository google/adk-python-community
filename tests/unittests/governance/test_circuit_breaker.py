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

"""Tests for governance circuit breaker."""

import time
from unittest import mock

import pytest

from google.adk_community.governance._circuit_breaker import CircuitBreaker
from google.adk_community.governance._circuit_breaker import CircuitState


@pytest.fixture()
def cb():
  return CircuitBreaker(failure_threshold=3, recovery_timeout_s=10.0)


class TestCircuitBreaker:

  def test_initial_state_is_closed(self, cb):
    assert cb.get_state("agent_a") == CircuitState.CLOSED
    assert cb.is_open("agent_a") is False

  def test_failures_below_threshold_stay_closed(self, cb):
    cb.record_failure("agent_a")
    cb.record_failure("agent_a")
    assert cb.get_state("agent_a") == CircuitState.CLOSED

  def test_failures_at_threshold_open(self, cb):
    for _ in range(3):
      cb.record_failure("agent_a")
    assert cb.get_state("agent_a") == CircuitState.OPEN
    assert cb.is_open("agent_a") is True

  def test_success_resets_failures(self, cb):
    cb.record_failure("agent_a")
    cb.record_failure("agent_a")
    cb.record_success("agent_a")
    assert cb.get_state("agent_a") == CircuitState.CLOSED

  def test_half_open_after_timeout(self, cb):
    for _ in range(3):
      cb.record_failure("agent_a")
    assert cb.is_open("agent_a") is True

    with mock.patch("time.monotonic", return_value=time.monotonic() + 11):
      assert cb.is_half_open("agent_a") is True
      assert cb.is_open("agent_a") is False

  def test_agents_are_independent(self, cb):
    for _ in range(3):
      cb.record_failure("agent_a")
    assert cb.is_open("agent_a") is True
    assert cb.is_open("agent_b") is False

  def test_summary(self, cb):
    for _ in range(3):
      cb.record_failure("agent_a")
    cb.record_failure("agent_b")
    summary = cb.summary()
    assert summary["agent_a"] == "open"
    assert summary["agent_b"] == "closed"
