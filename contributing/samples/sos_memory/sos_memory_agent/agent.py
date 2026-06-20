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

"""SOS Memory Agent - Agent with semantic memory via SOS Mirror API.

This sample demonstrates using SOS (Sovereign Operating System) Mirror
for agent memory. SOS provides:

- FRC Physics: Memories ranked by Frequency, Recency, and Context relevance
- Lineage Tracking: Cryptographic provenance for every memory
- Multi-Agent Isolation: Each agent has its own memory namespace
- Semantic Search: Vector-based similarity matching
"""

from datetime import datetime
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import load_memory, preload_memory


def update_current_time(callback_context: CallbackContext):
    """Update the current time in agent state for temporal awareness."""
    callback_context.state['_time'] = datetime.now().isoformat()


root_agent = Agent(
    model='gemini-2.0-flash',
    name='sos_memory_agent',
    description=(
        'Agent with semantic memory powered by SOS Mirror. '
        'Uses FRC physics for intelligent memory retrieval and '
        'lineage tracking for memory provenance.'
    ),
    before_agent_callback=update_current_time,
    instruction=(
        'You are a helpful assistant with access to persistent semantic memory.\n\n'
        'Your memory is powered by SOS Mirror, which uses FRC physics to retrieve '
        'the most relevant memories based on:\n'
        '- Frequency: How often a memory has been accessed\n'
        '- Recency: How recent the memory is\n'
        '- Context: Semantic relevance to the current query\n\n'
        'When the user asks a question:\n'
        '1. First, use load_memory to search for relevant past conversations\n'
        '2. If the first search yields no results, try different keywords\n'
        '3. Use remembered context to provide personalized responses\n'
        '4. Reference past interactions when relevant ("As we discussed before...")\n\n'
        'Current time: {_time}'
    ),
    tools=[preload_memory, load_memory],
)
