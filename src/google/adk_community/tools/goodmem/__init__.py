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

"""Goodmem tools module for ADK."""

from google.adk_community.plugins.goodmem import GoodmemClient
from .goodmem_tools import goodmem_fetch
from .goodmem_tools import goodmem_save
from .goodmem_tools import GoodmemFetchResponse
from .goodmem_tools import GoodmemFetchTool
from .goodmem_tools import GoodmemSaveResponse
from .goodmem_tools import GoodmemSaveTool
from .goodmem_tools import MemoryItem

__all__ = [
    "GoodmemClient",
    "goodmem_save",
    "goodmem_fetch",
    "GoodmemSaveResponse",
    "GoodmemSaveTool",
    "GoodmemFetchResponse",
    "GoodmemFetchTool",
    "MemoryItem",
]
