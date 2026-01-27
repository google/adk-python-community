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

"""Agent with full Redis memory: Working Memory + Long-Term Memory.

This agent demonstrates the complete two-tier memory architecture:
- Working Memory: Automatic session summarization when context grows large
- Long-Term Memory: Persistent facts extracted and searchable across sessions
"""

from datetime import datetime

from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import load_memory
from google.adk.tools import preload_memory


def before_agent(callback_context: CallbackContext):
  """Update state before agent runs."""
  callback_context.state["_time"] = datetime.now().isoformat()


async def after_agent(callback_context: CallbackContext):
  """Store session to long-term memory after agent completes."""
  # This triggers memory extraction to long-term memory
  await callback_context.add_session_to_memory()


root_agent = Agent(
    model="gemini-2.5-flash",
    name="redis_agent_memory_agent",
    description=(
        "Agent with full two-tier Redis memory: working memory for sessions,"
        " long-term memory for persistence."
    ),
    before_agent_callback=before_agent,
    after_agent_callback=after_agent,
    instruction="""You are a helpful assistant with a powerful two-tier memory system.

## Your Memory Capabilities

1. **Working Memory** (automatic): Your current conversation is automatically managed.
   When the conversation gets long, older messages are summarized to keep context efficient.

2. **Long-Term Memory** (persistent): Important facts and preferences are automatically
   extracted and stored. You can search this memory across sessions.

## How to Use Memory

- Use `load_memory` to search for information from past conversations
- When users share personal info (name, preferences, facts), acknowledge it - 
  it will be automatically saved to long-term memory
- If a search doesn't find results, try different keywords

## Conversation Guidelines

- Be conversational and remember details the user shares
- Reference past interactions when relevant
- Ask clarifying questions to learn more about the user

Current time: {_time}""",
    tools=[preload_memory, load_memory],
)
