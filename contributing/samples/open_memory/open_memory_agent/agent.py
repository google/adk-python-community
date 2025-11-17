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
from google.adk.tools import load_memory, preload_memory


def update_current_time(callback_context: CallbackContext):
  callback_context.state['_time'] = datetime.now().isoformat()

root_agent = Agent(
    model='gemini-2.5-flash',
    name='open_memory_agent',
    description='agent that has access to memory tools with OpenMemory via get_fast_api_app.',
    before_agent_callback=update_current_time,
    instruction=(
        'You are an agent that helps user answer questions. You have access to memory tools.\n'
        'When the user asks a question you do not know the answer to, you MUST use the load_memory tool to search for relevant information from past conversations.\n'
        'If the first search does not find relevant information, try different search terms or keywords related to the question.\n'
        'Current time: {_time}'
    ),
    tools=[preload_memory, load_memory],
)

