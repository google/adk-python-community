# Copyright 2026 pairsys.ai (DBA Goodmem.ai)
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

"""GoodMem Memory Service demo for ADK.

Run from the parent directory (contributing/samples/goodmem/):

  cd contributing/samples/goodmem
  adk web --memory_service_uri="goodmem://env" .

Then open http://localhost:8000 and select goodmem_memory_service_demo.

The memory service is created by adk web via the factory in goodmem/services.py.
Config (top_k, split_turn, timeout) is set there, not in this file.
See MEMORY_SERVICE.md "Usage" for programmatic config with Runner.
"""

from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import load_memory, preload_memory

async def save_to_memory(callback_context: CallbackContext) -> None:
    """Save new conversation turns to GoodMem after each agent response.

    ADK does use memory service to write to memory automatically. 
    ADK only writes to the memory service via a callback.
    
    add_session_to_memory() is a method of BaseMemoryService. 
    GoodmemMemoryService extends BaseMemoryService.

    The callback `after_agent_callback` is a method of Agent in ADK. 
    By passing add_session_to_memory() to after_agent_callback, 
    ADK will write to the memory service after every agent turn 
    (a turn is a user message and a model response).

    Without this callback, nothing would be stored
    and the agent would have no persistent memory to search later.
    """
    await callback_context.add_session_to_memory()


root_agent = Agent(
    model="gemini-2.5-flash",
    name="goodmem_memory_agent",
    instruction=(
        "You are a helpful assistant with persistent memory. "
        "Use load_memory to search for relevant memories from past conversations. "
        "Saving happens automatically after each response - do not try to call a save tool."
    ),
    after_agent_callback=save_to_memory,
    tools=[preload_memory, load_memory],
)
