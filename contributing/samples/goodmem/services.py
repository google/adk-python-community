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

"""Register GoodmemMemoryService with the ADK service registry.

adk web loads services.py from the agents root (this directory when you run
'adk web .' from contributing/samples/goodmem/). This registration must happen
before the server resolves --memory_service_uri="goodmem://env".

Edit the GoodmemMemoryService(...) call below to set top_k, timeout,
split_turn, or debug.

For how to use this file, see goodmem/goodmem_memory_service_demo/agent.py.
"""

import os

from google.adk.cli.service_registry import get_service_registry
from google.adk_community.memory.goodmem import GoodmemMemoryService


def _goodmem_factory(uri: str, **kwargs):
    return GoodmemMemoryService(
        base_url=os.getenv("GOODMEM_BASE_URL"),
        api_key=os.getenv("GOODMEM_API_KEY"),
        embedder_id=os.getenv("GOODMEM_EMBEDDER_ID"),
        top_k=5,
        timeout=30.0,
        split_turn=True,
        debug=False,
    )


get_service_registry().register_memory_service("goodmem", _goodmem_factory)
