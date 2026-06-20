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

"""Main entry point for SOS Memory Agent sample.

This sample demonstrates how to use SOS Mirror as a memory backend
for ADK agents. SOS provides semantic memory with FRC physics and
lineage tracking for memory provenance.

Prerequisites:
    1. Set up SOS Mirror API (https://github.com/servathadi/sos)
    2. Set environment variables:
       - SOS_MIRROR_URL: URL of SOS Mirror API (default: http://localhost:8844)
       - SOS_API_KEY: API key for authentication
       - GOOGLE_API_KEY: Gemini API key

Usage:
    # Start the agent server
    python main.py

    # Or use ADK CLI
    adk web sos_memory_agent
"""

import os
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk_community.memory import SOSMemoryService, SOSMemoryServiceConfig

from sos_memory_agent import root_agent


def create_memory_service() -> SOSMemoryService:
    """Create and configure SOS Memory service."""
    base_url = os.environ.get("SOS_MIRROR_URL", "http://localhost:8844")
    api_key = os.environ.get("SOS_API_KEY", "")

    if not api_key:
        raise ValueError(
            "SOS_API_KEY environment variable is required. "
            "Get your API key from your SOS Mirror deployment."
        )

    config = SOSMemoryServiceConfig(
        search_top_k=10,
        timeout=30.0,
        user_content_salience=0.8,
        model_content_salience=0.7,
        enable_lineage_tracking=True,
    )

    return SOSMemoryService(
        base_url=base_url,
        api_key=api_key,
        agent_id="sos_memory_agent",
        config=config,
    )


def main():
    """Run the SOS Memory Agent."""
    memory_service = create_memory_service()
    session_service = InMemorySessionService()

    runner = Runner(
        agent=root_agent,
        app_name="sos_memory_sample",
        session_service=session_service,
        memory_service=memory_service,
    )

    # Start the web interface
    runner.run()


if __name__ == "__main__":
    main()
