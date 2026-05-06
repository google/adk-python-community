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

"""Circuit breaker for governance plugin."""

from __future__ import annotations

import enum
import threading
import time


class CircuitState(enum.Enum):
  CLOSED = "closed"
  OPEN = "open"
  HALF_OPEN = "half_open"


class CircuitBreaker:
  """Per-agent circuit breaker tracking consecutive failures."""

  def __init__(
      self,
      *,
      failure_threshold: int,
      recovery_timeout_s: float,
  ) -> None:
    self._failure_threshold = failure_threshold
    self._recovery_timeout_s = recovery_timeout_s
    self._agents: dict[str, _AgentCircuit] = {}
    self._lock = threading.Lock()

  def is_open(self, agent_name: str) -> bool:
    """Check if agent circuit is open (should be isolated)."""
    with self._lock:
      circuit = self._agents.get(agent_name)
      if circuit is None:
        return False
      return circuit.state() == CircuitState.OPEN

  def is_half_open(self, agent_name: str) -> bool:
    """Check if agent circuit is half-open (allow one probe)."""
    with self._lock:
      circuit = self._agents.get(agent_name)
      if circuit is None:
        return False
      return circuit.state() == CircuitState.HALF_OPEN

  def record_failure(self, agent_name: str) -> CircuitState:
    """Record a failure for the agent. Returns new state."""
    with self._lock:
      circuit = self._ensure(agent_name)
      circuit.consecutive_failures += 1
      if circuit.consecutive_failures >= self._failure_threshold:
        if circuit.opened_at is None:
          circuit.opened_at = time.monotonic()
      circuit.probe_in_flight = False
      return circuit.state()

  def record_success(self, agent_name: str) -> None:
    """Record a success, resetting failure count."""
    with self._lock:
      circuit = self._ensure(agent_name)
      circuit.consecutive_failures = 0
      circuit.opened_at = None
      circuit.probe_in_flight = False

  def claim_probe(self, agent_name: str) -> bool:
    """Atomically claim the HALF_OPEN probe slot. Returns True if claimed."""
    with self._lock:
      circuit = self._agents.get(agent_name)
      if circuit is None:
        return False
      if circuit.state() != CircuitState.HALF_OPEN:
        return False
      if circuit.probe_in_flight:
        return False
      circuit.probe_in_flight = True
      return True

  def get_state(self, agent_name: str) -> CircuitState:
    """Get current circuit state for an agent."""
    with self._lock:
      circuit = self._agents.get(agent_name)
      if circuit is None:
        return CircuitState.CLOSED
      return circuit.state()

  def summary(self) -> dict[str, str]:
    """Return {agent_name: state_name} for all tracked agents."""
    with self._lock:
      return {
          name: circuit.state().value for name, circuit in self._agents.items()
      }

  def _ensure(self, agent_name: str) -> _AgentCircuit:
    if agent_name not in self._agents:
      self._agents[agent_name] = _AgentCircuit(
          failure_threshold=self._failure_threshold,
          recovery_timeout_s=self._recovery_timeout_s,
      )
    return self._agents[agent_name]


class _AgentCircuit:
  """State for a single agent's circuit."""

  def __init__(
      self, *, failure_threshold: int, recovery_timeout_s: float
  ) -> None:
    self._failure_threshold = failure_threshold
    self._recovery_timeout_s = recovery_timeout_s
    self.consecutive_failures: int = 0
    self.opened_at: float | None = None
    self.probe_in_flight: bool = False

  def state(self) -> CircuitState:
    if self.consecutive_failures < self._failure_threshold:
      return CircuitState.CLOSED
    if self.opened_at is None:
      return CircuitState.CLOSED
    elapsed = time.monotonic() - self.opened_at
    if elapsed >= self._recovery_timeout_s:
      return CircuitState.HALF_OPEN
    return CircuitState.OPEN
