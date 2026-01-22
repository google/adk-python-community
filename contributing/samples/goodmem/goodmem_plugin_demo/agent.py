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

# Example code for using the Goodmem Chat Plugin in an ADK app
# For usage, see PLUGIN.md.

import os

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk_community.plugins.goodmem import GoodmemChatPlugin

root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
)

# Initialize the Goodmem Chat Plugin
goodmem_chat_plugin = GoodmemChatPlugin(
    base_url=os.getenv("GOODMEM_BASE_URL"),
    api_key=os.getenv("GOODMEM_API_KEY"),
    embedder_id=os.getenv("EMBEDDER_ID"),  # Optional, if not provided, will fetch the first embedder from API. If no embedders are fetched, will raise an error.
    top_k=5,  # Optional: number of top-k most relevant entries to retrieve (default: 5)
    debug=False  # Optional: set to True to enable debug mode
)

# Create App with the plugin (this is what adk run looks for)
app = App(
    name='goodmem_plugin_demo_agent',
    root_agent=root_agent,
    plugins=[goodmem_chat_plugin]
)
