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

"""Example script demonstrating how to test the resilience sample.

This script shows how to use the test_helpers.py utilities to test
various resilience patterns including timeouts and failures.
"""

import asyncio
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional - user can set GOOGLE_API_KEY directly
    pass

from google.adk.runners import Runner
from google.adk.tools import AgentTool
from agent import coordinator_agent, app, TimeoutAgentTool
from test_helpers import (
    timeout_test_agent,
    failure_test_agent,
)


async def test_normal_operation():
  """Test normal operation with a simple query."""
  print("=" * 60)
  print("Test 1: Normal Operation")
  print("=" * 60)
  
  runner = Runner(
      app_name="resilience_test",
      agent=coordinator_agent,
  )
  
  query = "What is quantum computing?"
  print(f"\nQuery: {query}\n")
  
  response = await runner.run_async(query)
  print(f"\nResponse: {response.text}\n")


async def test_timeout_scenario():
  """Test timeout scenario using test helpers."""
  print("=" * 60)
  print("Test 2: Timeout Scenario")
  print("=" * 60)
  
  # Create a coordinator with timeout_test_agent wrapped in TimeoutAgentTool
  from agent import Agent
  
  timeout_coordinator = Agent(
      name='timeout_coordinator',
      model='gemini-2.5-flash-lite',
      description='Coordinator for testing timeout scenarios',
      instruction="""
      You are a test coordinator. When given a task, use the timeout_test_agent
      tool. This agent will timeout, demonstrating the timeout handling mechanism.
      After the timeout, provide a clear explanation to the user.
      """,
      tools=[
          TimeoutAgentTool(
              agent=timeout_test_agent,
              timeout=5.0,  # Short timeout to trigger timeout behavior
              timeout_error_message="Test agent timed out after 5 seconds",
          ),
      ],
  )
  
  runner = Runner(
      app_name="timeout_test",
      agent=timeout_coordinator,
  )
  
  query = "Perform a test task"
  print(f"\nQuery: {query}\n")
  print("Note: This will timeout after 5 seconds...\n")
  
  response = await runner.run_async(query)
  print(f"\nResponse: {response.text}\n")


async def test_failure_scenario():
  """Test failure scenario using test helpers."""
  print("=" * 60)
  print("Test 3: Failure Scenario")
  print("=" * 60)
  
  # Create a coordinator with failure_test_agent
  from agent import Agent
  
  failure_coordinator = Agent(
      name='failure_coordinator',
      model='gemini-2.5-flash-lite',
      description='Coordinator for testing failure scenarios',
      instruction="""
      You are a test coordinator. When given a task, use the failure_test_agent
      tool. This agent will fail, demonstrating the error handling mechanism.
      After the failure, provide a clear explanation to the user about what
      went wrong and what alternatives might be available.
      """,
      tools=[
          AgentTool(agent=failure_test_agent),
      ],
  )
  
  runner = Runner(
      app_name="failure_test",
      agent=failure_coordinator,
  )
  
  query = "Perform a test task"
  print(f"\nQuery: {query}\n")
  print("Note: This will trigger a failure...\n")
  
  response = await runner.run_async(query)
  print(f"\nResponse: {response.text}\n")


async def main():
  """Run all test scenarios."""
  if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY environment variable is not set.")
    print("Please create a .env file with your Google API key.")
    return
  
  print("\n" + "=" * 60)
  print("Resilience Sample Test Suite")
  print("=" * 60)
  print("\nThis script demonstrates various resilience patterns:")
  print("1. Normal operation")
  print("2. Timeout handling")
  print("3. Failure handling")
  print("\n" + "=" * 60 + "\n")
  
  try:
    # Test normal operation
    await test_normal_operation()
    
    # Test timeout scenario
    await test_timeout_scenario()
    
    # Test failure scenario
    await test_failure_scenario()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
    
  except Exception as e:
    print(f"\nError during testing: {e}")
    import traceback
    traceback.print_exc()


if __name__ == "__main__":
  asyncio.run(main())

