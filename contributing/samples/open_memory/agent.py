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


from datetime import datetime

from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.load_memory_tool import load_memory_tool
from google.adk.tools.preload_memory_tool import preload_memory_tool


def update_current_time(callback_context: CallbackContext):
  callback_context.state['_time'] = datetime.now().isoformat()


async def auto_save_session_to_memory_callback(callback_context: CallbackContext):
  """Auto-saves the current session to memory after each agent turn.
  
  Since there's no automatic save (saves only happen when the PATCH /memory endpoint
  is called), this callback is a simple workaround for testing that saves memories
  after each agent turn.
  """
  if callback_context._invocation_context.memory_service:
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session)


root_agent = Agent(
    model='gemini-2.5-flash',
    name='open_memory_agent',
    description='agent that has access to memory tools with OpenMemory.',
    before_agent_callback=update_current_time,
    after_agent_callback=auto_save_session_to_memory_callback,
    instruction=(
        'You are an agent that helps user answer questions. You have access to memory tools.\n'
        'You can use the memory tools to look up the information in the memory. Current time: {_time}'
    ),
    tools=[load_memory_tool, preload_memory_tool],
)

