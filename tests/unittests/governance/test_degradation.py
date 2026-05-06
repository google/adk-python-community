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

"""Tests for governance degradation manager."""

import pytest

from google.adk_community.governance._degradation import DegradationManager


class TestDegradationManager:

  def test_not_configured_without_fallback(self):
    dm = DegradationManager(threshold=0.8, fallback_model=None)
    assert dm.is_configured is False
    assert dm.should_degrade(0.9) is False

  def test_configured_with_fallback(self):
    dm = DegradationManager(
        threshold=0.8, fallback_model="gemini-2.0-flash-lite"
    )
    assert dm.is_configured is True

  def test_should_degrade_above_threshold(self):
    dm = DegradationManager(
        threshold=0.8, fallback_model="gemini-2.0-flash-lite"
    )
    assert dm.should_degrade(0.8) is True
    assert dm.should_degrade(0.9) is True

  def test_should_not_degrade_below_threshold(self):
    dm = DegradationManager(
        threshold=0.8, fallback_model="gemini-2.0-flash-lite"
    )
    assert dm.should_degrade(0.79) is False

  def test_should_disable_tool(self):
    dm = DegradationManager(
        threshold=0.5,
        fallback_model="lite",
        disable_tools_on_degrade=["expensive_search"],
    )
    assert dm.should_disable_tool("expensive_search", 0.6) is True
    assert dm.should_disable_tool("cheap_tool", 0.6) is False
    assert dm.should_disable_tool("expensive_search", 0.4) is False

  def test_record_event(self):
    dm = DegradationManager(
        threshold=0.8, fallback_model="gemini-2.0-flash-lite"
    )
    event = dm.record_event(
        agent_name="agent_a",
        utilization=0.85,
        original_model="gemini-2.5-pro",
    )
    assert event.agent_name == "agent_a"
    assert event.utilization_pct == 85.0
    assert event.original_model == "gemini-2.5-pro"
    assert event.fallback_model == "gemini-2.0-flash-lite"
    assert len(dm.events) == 1

  def test_events_are_independent_copies(self):
    dm = DegradationManager(threshold=0.8, fallback_model="lite")
    dm.record_event(agent_name="a", utilization=0.9, original_model="pro")
    events1 = dm.events
    events2 = dm.events
    assert events1 is not events2
    assert events1 == events2
