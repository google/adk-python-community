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

"""Example of using Redis Agent Memory Service with get_fast_api_app."""

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from urllib.parse import urlparse

from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.service_registry import get_service_registry
from google.adk_community.memory import (
    RedisAgentMemoryService,
    RedisAgentMemoryServiceConfig,
)

# Load environment variables from .env file if it exists
load_dotenv()


def redis_agent_memory_factory(uri: str, **kwargs):
    """Factory function for creating RedisAgentMemoryService from URI."""
    parsed = urlparse(uri)
    location = parsed.netloc + parsed.path
    base_url = (
        location
        if location.startswith(("http://", "https://"))
        else f"http://{location}"
    )

    # Get configuration from environment variables
    config = RedisAgentMemoryServiceConfig(
        api_base_url=base_url,
        default_namespace=os.getenv("REDIS_AGENT_MEMORY_NAMESPACE", "adk_sample"),
        extraction_strategy=os.getenv(
            "REDIS_AGENT_MEMORY_EXTRACTION_STRATEGY", "discrete"
        ),
        recency_boost=os.getenv(
            "REDIS_AGENT_MEMORY_RECENCY_BOOST", "true"
        ).lower()
        == "true",
        semantic_weight=float(
            os.getenv("REDIS_AGENT_MEMORY_SEMANTIC_WEIGHT", "0.8")
        ),
        recency_weight=float(os.getenv("REDIS_AGENT_MEMORY_RECENCY_WEIGHT", "0.2")),
    )

    return RedisAgentMemoryService(config=config)


# Register Redis Agent Memory service factory for redis-agent-memory:// URI scheme
get_service_registry().register_memory_service(
    "redis-agent-memory", redis_agent_memory_factory
)

# Build Redis Agent Memory URI from environment variables
base_url = (
    os.getenv("REDIS_AGENT_MEMORY_URL", "http://localhost:8000")
    .replace("http://", "")
    .replace("https://", "")
)
MEMORY_SERVICE_URI = f"redis-agent-memory://{base_url}"

# Create the FastAPI app using get_fast_api_app
app: FastAPI = get_fast_api_app(
    agents_dir=".",
    memory_service_uri=MEMORY_SERVICE_URI,
    web=True,
)


if __name__ == "__main__":
    # Use the PORT environment variable provided by Cloud Run, defaulting to 8000
    port = int(os.environ.get("PORT", 8080))
    print(f"""
Starting Redis Agent Memory Sample
===================================
ADK Server:          http://localhost:{port}
Memory Server:       {os.getenv('REDIS_AGENT_MEMORY_URL', 'http://localhost:8000')}
Namespace:           {os.getenv('REDIS_AGENT_MEMORY_NAMESPACE', 'adk_sample')}
Extraction Strategy: {os.getenv('REDIS_AGENT_MEMORY_EXTRACTION_STRATEGY', 'discrete')}
""")
    uvicorn.run(app, host="0.0.0.0", port=port)

