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


import os

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk_community.tools.goodmem import GoodmemSaveTool
from google.adk_community.tools.goodmem import GoodmemFetchTool

# Initialize Goodmem tools
goodmem_save_tool = GoodmemSaveTool(
    base_url=os.getenv("GOODMEM_BASE_URL"),
    api_key=os.getenv("GOODMEM_API_KEY"),
    embedder_id=os.getenv("GOODMEM_EMBEDDER_ID"),  # Optional, only needed if you wanna pin a specific embedder from multiple embedders
    debug=False
)
goodmem_fetch_tool = GoodmemFetchTool(
    base_url=os.getenv("GOODMEM_BASE_URL"),
    api_key=os.getenv("GOODMEM_API_KEY"),
    embedder_id=os.getenv("GOODMEM_EMBEDDER_ID"),  # Optional, only needed if you wanna pin a specific embedder from multiple embedders
    top_k=5,  # Default number of memories to retrieve
    debug=False
)

# Create root agent with Goodmem tools
root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='goodmem_tools_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
    tools=[goodmem_save_tool, goodmem_fetch_tool]
)

# Create App (this is what adk run looks for)
app = App(
    name='goodmem_tools_demo',
    root_agent=root_agent,
)
