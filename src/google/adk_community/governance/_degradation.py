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

"""Degradation logic for governance plugin."""

from __future__ import annotations

from dataclasses import dataclass
import threading


@dataclass
class DegradationEvent:
  """Record of a single degradation action."""

  agent_name: str
  utilization_pct: float
  original_model: str
  fallback_model: str


class DegradationManager:
  """Manages model degradation and tool disabling near budget limits."""

  def __init__(
      self,
      *,
      threshold: float,
      fallback_model: str | None,
      disable_tools_on_degrade: list[str] | None = None,
  ) -> None:
    self._threshold = threshold
    self._fallback_model = fallback_model
    self._disable_tools: frozenset[str] = frozenset(
        disable_tools_on_degrade or []
    )
    self._events: list[DegradationEvent] = []
    self._lock = threading.Lock()

  @property
  def is_configured(self) -> bool:
    return self._fallback_model is not None

  def should_degrade(self, utilization: float) -> bool:
    """Check if degradation should be triggered."""
    return self.is_configured and utilization >= self._threshold

  def should_disable_tool(self, tool_name: str, utilization: float) -> bool:
    """Check if a tool should be disabled due to degradation."""
    if utilization < self._threshold:
      return False
    return tool_name in self._disable_tools

  def record_event(
      self,
      *,
      agent_name: str,
      utilization: float,
      original_model: str,
  ) -> DegradationEvent:
    """Record a degradation event. Returns the event for logging."""
    event = DegradationEvent(
        agent_name=agent_name,
        utilization_pct=round(utilization * 100, 1),
        original_model=original_model,
        fallback_model=self._fallback_model or "",
    )
    with self._lock:
      self._events.append(event)
    return event

  @property
  def fallback_model(self) -> str | None:
    return self._fallback_model

  @property
  def events(self) -> list[DegradationEvent]:
    with self._lock:
      return list(self._events)
