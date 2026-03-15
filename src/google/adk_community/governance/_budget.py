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

"""Budget tracking for governance plugin."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import threading


@dataclass
class BudgetSnapshot:
  """Read-only snapshot of current budget state."""

  org_spent_usd: float
  org_limit_usd: float
  agent_spent: dict[str, float]
  agent_limit_usd: float

  @property
  def org_utilization(self) -> float:
    if self.org_limit_usd <= 0:
      return 1.0
    return self.org_spent_usd / self.org_limit_usd


class BudgetTracker:
  """Thread-safe budget tracker with per-agent and org-level limits."""

  def __init__(
      self,
      *,
      org_limit_usd: float,
      agent_limit_usd: float,
      cost_per_1k_input_tokens: float,
      cost_per_1k_output_tokens: float,
  ) -> None:
    self._org_limit_usd = org_limit_usd
    self._agent_limit_usd = agent_limit_usd
    self._cost_per_1k_input = cost_per_1k_input_tokens
    self._cost_per_1k_output = cost_per_1k_output_tokens
    self._org_spent_usd: float = 0.0
    self._agent_spent: dict[str, float] = {}
    self._lock = threading.Lock()

  def estimate_cost(
      self,
      input_tokens: int,
      output_tokens: int,
  ) -> float:
    """Estimate cost from token counts (clamped to non-negative)."""
    raw = (
        max(input_tokens, 0) / 1000.0 * self._cost_per_1k_input
        + max(output_tokens, 0) / 1000.0 * self._cost_per_1k_output
    )
    return max(raw, 0.0)

  def check(self, agent_name: str) -> tuple[bool, str]:
    """Check if agent is within budget. Returns (allowed, reason)."""
    with self._lock:
      if self._org_spent_usd >= self._org_limit_usd:
        return False, (
            f"Org budget exhausted: ${self._org_spent_usd:.4f}"
            f" / ${self._org_limit_usd:.4f}"
        )
      agent_spent = self._agent_spent.get(agent_name, 0.0)
      if agent_spent >= self._agent_limit_usd:
        return False, (
            f"Agent '{agent_name}' budget exhausted:"
            f" ${agent_spent:.4f} / ${self._agent_limit_usd:.4f}"
        )
      return True, ""

  def record(self, agent_name: str, cost_usd: float) -> None:
    """Record cost for an agent."""
    with self._lock:
      self._org_spent_usd += cost_usd
      self._agent_spent[agent_name] = (
          self._agent_spent.get(agent_name, 0.0) + cost_usd
      )

  def utilization(self) -> float:
    """Current org-level budget utilization (0.0 to 1.0+)."""
    with self._lock:
      if self._org_limit_usd <= 0:
        return 1.0
      return self._org_spent_usd / self._org_limit_usd

  def snapshot(self) -> BudgetSnapshot:
    """Return a read-only snapshot of current budget state."""
    with self._lock:
      return BudgetSnapshot(
          org_spent_usd=self._org_spent_usd,
          org_limit_usd=self._org_limit_usd,
          agent_spent=dict(self._agent_spent),
          agent_limit_usd=self._agent_limit_usd,
      )
