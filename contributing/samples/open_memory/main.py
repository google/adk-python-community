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

"""Example of using OpenMemory with get_fast_api_app."""

import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from urllib.parse import urlparse
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.service_registry import get_service_registry
from google.adk_community.memory import OpenMemoryService

# Load environment variables from .env file if it exists
load_dotenv()

# Register OpenMemory service factory for openmemory:// URI scheme
def openmemory_factory(uri: str, **kwargs):
    parsed = urlparse(uri)
    location = parsed.netloc + parsed.path
    base_url = location if location.startswith(('http://', 'https://')) else f'http://{location}'
    api_key = os.getenv('OPENMEMORY_API_KEY', '')
    if not api_key:
        raise ValueError("OpenMemory API key required. Set OPENMEMORY_API_KEY environment variable.")
    return OpenMemoryService(base_url=base_url, api_key=api_key)

get_service_registry().register_memory_service("openmemory", openmemory_factory)

# Build OpenMemory URI from environment variables (API key comes from env var)
base_url = os.getenv('OPENMEMORY_BASE_URL', 'http://localhost:8080').replace('http://', '').replace('https://', '')
MEMORY_SERVICE_URI = f"openmemory://{base_url}"


# Create the FastAPI app using get_fast_api_app
app: FastAPI = get_fast_api_app(
    agents_dir=".",
    memory_service_uri=MEMORY_SERVICE_URI,
    web=True,
)

if __name__ == '__main__':
    # Use the PORT environment variable provided by Cloud Run, defaulting to 8000
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)

