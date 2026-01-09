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

"""Tools module for ADK Community integrations.

This module uses lazy loading to provide helpful error messages when
optional dependencies are not installed.
"""

__all__ = [
    "BaseRedisSearchTool",
    "VectorizedSearchTool",
    "RedisVectorSearchTool",
    "RedisHybridSearchTool",
    "RedisRangeSearchTool",
    "RedisTextSearchTool",
    "RedisVectorQueryConfig",
    "RedisHybridQueryConfig",
    "RedisRangeQueryConfig",
]

# Redis tool names for lazy loading
_REDIS_TOOLS = {
    "BaseRedisSearchTool",
    "VectorizedSearchTool",
    "RedisVectorSearchTool",
    "RedisHybridSearchTool",
    "RedisRangeSearchTool",
    "RedisTextSearchTool",
}

# Redis config names for lazy loading
_REDIS_CONFIGS = {
    "RedisVectorQueryConfig",
    "RedisHybridQueryConfig",
    "RedisRangeQueryConfig",
    "RedisTextQueryConfig",
}


def __getattr__(name: str):
  """Lazy load tools to provide helpful error messages."""
  if name in _REDIS_TOOLS:
    try:
      from .redis import BaseRedisSearchTool
      from .redis import RedisHybridSearchTool
      from .redis import RedisRangeSearchTool
      from .redis import RedisTextSearchTool
      from .redis import RedisVectorSearchTool
      from .redis import VectorizedSearchTool

      globals().update({
          "BaseRedisSearchTool": BaseRedisSearchTool,
          "VectorizedSearchTool": VectorizedSearchTool,
          "RedisVectorSearchTool": RedisVectorSearchTool,
          "RedisHybridSearchTool": RedisHybridSearchTool,
          "RedisRangeSearchTool": RedisRangeSearchTool,
          "RedisTextSearchTool": RedisTextSearchTool,
      })
      return globals()[name]
    except ImportError as e:
      raise ImportError(
          f"{name} requires redisvl. "
          "Install with: pip install google-adk-community[redis-vl]"
      ) from e
  if name in _REDIS_CONFIGS:
    from .redis import RedisHybridQueryConfig
    from .redis import RedisRangeQueryConfig
    from .redis import RedisTextQueryConfig
    from .redis import RedisVectorQueryConfig

    globals().update({
        "RedisVectorQueryConfig": RedisVectorQueryConfig,
        "RedisHybridQueryConfig": RedisHybridQueryConfig,
        "RedisRangeQueryConfig": RedisRangeQueryConfig,
        "RedisTextQueryConfig": RedisTextQueryConfig,
    })
    return globals()[name]
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
