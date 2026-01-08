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

"""Test helpers for simulating timeout and failure scenarios.

This module provides agents and tools that simulate various failure modes
for testing resilience patterns.
"""

import asyncio
from typing import Any

from google.adk import Agent
from google.adk.tools import AgentTool, BaseTool
from google.adk.tools.tool_context import ToolContext


# ============================================================================
# Slow Agent (Simulates Timeout)
# ============================================================================

slow_agent = Agent(
    name='slow_agent',
    model='gemini-2.5-flash-lite',
    description='An agent that intentionally takes a long time to respond',
    instruction="""
    You are a very thorough agent that takes your time. When given any task:
    1. Acknowledge the task
    2. Think carefully about it for a while
    3. Break it down into many detailed steps
    4. Consider each step carefully
    5. Provide a comprehensive response
    
    Be extremely detailed and thorough. Take your time to ensure quality.
    Always provide extensive context and multiple examples.
    """,
)


# ============================================================================
# Failing Agent (Simulates Errors)
# ============================================================================

failing_agent = Agent(
    name='failing_agent',
    model='gemini-2.5-flash-lite',
    description='An agent that intentionally fails',
    instruction="""
    You are an agent that simulates failures. When given any task:
    1. Acknowledge the task
    2. Attempt to process it
    3. Always return an error message indicating failure
    
    Always respond with: "ERROR: This agent is configured to simulate failures. 
    This is a test scenario to demonstrate error handling."
    """,
)


# ============================================================================
# Intermittent Agent (Sometimes Fails)
# ============================================================================

intermittent_agent = Agent(
    name='intermittent_agent',
    model='gemini-2.5-flash-lite',
    description='An agent that sometimes fails, sometimes succeeds',
    instruction="""
    You are an agent with intermittent reliability. When given a task:
    - If the task contains the word "fail" or "error", return an error
    - Otherwise, process normally
    
    This simulates real-world scenarios where agents sometimes fail
    due to external factors.
    """,
)


# ============================================================================
# Custom Tool that Simulates Timeout
# ============================================================================

class TimeoutSimulatorTool(BaseTool):
  """A tool that simulates timeout by sleeping."""

  def __init__(self, sleep_duration: float = 35.0):
    """Initialize timeout simulator.

    Args:
      sleep_duration: How long to sleep (should exceed timeout to trigger).
    """
    super().__init__(
        name='timeout_simulator',
        description='A tool that simulates timeout scenarios',
    )
    self.sleep_duration = sleep_duration

  async def run_async(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    """Sleep for the specified duration to simulate timeout."""
    await asyncio.sleep(self.sleep_duration)
    return {'status': 'completed', 'message': 'This should not be reached'}


# ============================================================================
# Custom Tool that Simulates Failure
# ============================================================================

class FailureSimulatorTool(BaseTool):
  """A tool that always fails."""

  def __init__(self, error_message: str = 'Simulated failure'):
    """Initialize failure simulator.

    Args:
      error_message: The error message to raise.
    """
    super().__init__(
        name='failure_simulator',
        description='A tool that simulates failures',
    )
    self.error_message = error_message

  async def run_async(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    """Always raise an exception."""
    raise RuntimeError(self.error_message)


# ============================================================================
# Test Agent Configurations
# ============================================================================

# Agent that uses timeout simulator tool
timeout_test_agent = Agent(
    name='timeout_test_agent',
    model='gemini-2.5-flash-lite',
    description='Agent for testing timeout scenarios',
    instruction="""
    You are a test agent. When asked to do anything, use the timeout_simulator
    tool. This will help test timeout handling.
    """,
    tools=[TimeoutSimulatorTool(sleep_duration=35.0)],
)

# Agent that uses failure simulator tool
failure_test_agent = Agent(
    name='failure_test_agent',
    model='gemini-2.5-flash-lite',
    description='Agent for testing failure scenarios',
    instruction="""
    You are a test agent. When asked to do anything, use the failure_simulator
    tool. This will help test error handling.
    """,
    tools=[FailureSimulatorTool(error_message='Test failure simulation')],
)

