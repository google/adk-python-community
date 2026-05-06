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

"""Tool policy enforcement for governance plugin."""

from __future__ import annotations


class ToolPolicy:
  """Decides whether a tool call is allowed."""

  def __init__(
      self,
      *,
      blocked_tools: list[str] | None = None,
      allowed_tools: list[str] | None = None,
  ) -> None:
    self._blocked: frozenset[str] = frozenset(blocked_tools or [])
    self._allowed: frozenset[str] | None = (
        frozenset(allowed_tools) if allowed_tools is not None else None
    )

  def check(self, tool_name: str) -> tuple[bool, str]:
    """Check if tool is allowed. Returns (allowed, reason)."""
    if tool_name in self._blocked:
      return False, f"Tool '{tool_name}' is blocked by governance policy"
    if self._allowed is not None and tool_name not in self._allowed:
      return False, f"Tool '{tool_name}' is not in the allowed list"
    return True, ""
