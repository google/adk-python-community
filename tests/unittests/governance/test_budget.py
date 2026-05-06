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

"""Tests for governance budget tracker."""

import pytest

from google.adk_community.governance._budget import BudgetTracker


@pytest.fixture()
def tracker():
  return BudgetTracker(
      org_limit_usd=1.0,
      agent_limit_usd=0.5,
      cost_per_1k_input_tokens=0.001,
      cost_per_1k_output_tokens=0.002,
  )


class TestBudgetTracker:

  def test_initial_state_allows(self, tracker):
    allowed, _ = tracker.check("agent_a")
    assert allowed is True

  def test_estimate_cost(self, tracker):
    cost = tracker.estimate_cost(1000, 1000)
    assert cost == pytest.approx(0.003)

  def test_record_and_check_org_limit(self, tracker):
    tracker.record("agent_a", 0.6)
    tracker.record("agent_b", 0.5)
    allowed, reason = tracker.check("agent_a")
    assert allowed is False
    assert "Org budget" in reason

  def test_record_and_check_agent_limit(self, tracker):
    tracker.record("agent_a", 0.5)
    allowed, reason = tracker.check("agent_a")
    assert allowed is False
    assert "agent_a" in reason

  def test_utilization(self, tracker):
    tracker.record("agent_a", 0.3)
    assert tracker.utilization() == pytest.approx(0.3)

  def test_snapshot(self, tracker):
    tracker.record("agent_a", 0.2)
    tracker.record("agent_b", 0.1)
    snap = tracker.snapshot()
    assert snap.org_spent_usd == pytest.approx(0.3)
    assert snap.org_limit_usd == 1.0
    assert snap.agent_spent["agent_a"] == pytest.approx(0.2)
    assert snap.agent_spent["agent_b"] == pytest.approx(0.1)
    assert snap.org_utilization == pytest.approx(0.3)

  def test_zero_limit_always_blocked(self):
    t = BudgetTracker(
        org_limit_usd=0.0,
        agent_limit_usd=0.0,
        cost_per_1k_input_tokens=0.001,
        cost_per_1k_output_tokens=0.002,
    )
    allowed, _ = t.check("agent_a")
    assert allowed is False
