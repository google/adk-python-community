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

"""Tests for governance tool policy."""

import pytest

from google.adk_community.governance._policy import ToolPolicy


class TestToolPolicy:

  def test_no_restrictions_allows_all(self):
    policy = ToolPolicy()
    allowed, _ = policy.check("any_tool")
    assert allowed is True

  def test_blocked_tool(self):
    policy = ToolPolicy(blocked_tools=["dangerous_tool"])
    allowed, reason = policy.check("dangerous_tool")
    assert allowed is False
    assert "blocked" in reason

  def test_blocked_does_not_affect_others(self):
    policy = ToolPolicy(blocked_tools=["dangerous_tool"])
    allowed, _ = policy.check("safe_tool")
    assert allowed is True

  def test_allowlist_permits_listed(self):
    policy = ToolPolicy(allowed_tools=["search", "calculator"])
    allowed, _ = policy.check("search")
    assert allowed is True

  def test_allowlist_blocks_unlisted(self):
    policy = ToolPolicy(allowed_tools=["search", "calculator"])
    allowed, reason = policy.check("delete_database")
    assert allowed is False
    assert "not in the allowed list" in reason

  def test_blocklist_takes_precedence_over_allowlist(self):
    policy = ToolPolicy(
        blocked_tools=["search"],
        allowed_tools=["search", "calculator"],
    )
    allowed, reason = policy.check("search")
    assert allowed is False
    assert "blocked" in reason
