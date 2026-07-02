# Copyright 2026 Google LLC
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

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import uuid

from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types
import pytest

from google.adk_community.memory.milvus_memory_service import MilvusMemoryService
from google.adk_community.memory.milvus_memory_service import MilvusMemoryServiceConfig

_VOCAB = ("milvus", "vector", "memory", "cooking")


def _keyword_embedding(texts):
  vectors = []
  for text in texts:
    lowered = text.lower()
    vector = [float(lowered.count(word)) for word in _VOCAB]
    if not any(vector):
      vector[-1] = 0.01
    vectors.append(vector)
  return vectors


def _session() -> Session:
  return Session(
      app_name="milvus-memory-e2e",
      user_id="user-1",
      id="session-1",
      last_update_time=1000,
      events=[
          Event(
              id="event-1",
              invocation_id="inv-1",
              author="user",
              timestamp=12345,
              content=types.Content(
                  parts=[
                      types.Part(text="Milvus stores vector memory for agents.")
                  ]
              ),
          ),
          Event(
              id="event-2",
              invocation_id="inv-2",
              author="user",
              timestamp=12346,
              content=types.Content(
                  parts=[types.Part(text="I enjoy cooking pasta.")]
              ),
          ),
      ],
  )


async def _drop_collection(service: MilvusMemoryService) -> None:
  try:
    await asyncio.to_thread(
        service._client.drop_collection,  # pylint: disable=protected-access
        collection_name=service._config.collection_name,  # pylint: disable=protected-access
    )
  except Exception:
    pass


async def _run_memory_e2e(config: MilvusMemoryServiceConfig) -> None:
  service = MilvusMemoryService(
      embedding_function=_keyword_embedding,
      config=config,
  )
  try:
    await service.add_session_to_memory(_session())

    result = await service.search_memory(
        app_name="milvus-memory-e2e",
        user_id="user-1",
        query="vector memory",
    )
    assert result.memories
    assert (
        "Milvus stores vector memory"
        in result.memories[0].content.parts[0].text
    )

    isolated_result = await service.search_memory(
        app_name="milvus-memory-e2e",
        user_id="user-2",
        query="vector memory",
    )
    assert isolated_result.memories == []
  finally:
    await _drop_collection(service)
    await service.close()


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_MILVUS_LITE_E2E") != "1",
    reason="Set RUN_MILVUS_LITE_E2E=1 to run Milvus Lite E2E.",
)
async def test_milvus_lite_memory_e2e(tmp_path: Path):
  collection_name = f"adk_memory_e2e_{uuid.uuid4().hex[:8]}"
  await _run_memory_e2e(
      MilvusMemoryServiceConfig(
          uri=str(tmp_path / "milvus_lite.db"),
          collection_name=collection_name,
          dimension=len(_VOCAB),
          consistency_level="Strong",
      )
  )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("ZILLIZ_URI") or not os.getenv("ZILLIZ_TOKEN"),
    reason="Set ZILLIZ_URI and ZILLIZ_TOKEN to run Zilliz Cloud E2E.",
)
async def test_zilliz_cloud_memory_e2e():
  collection_name = f"adk_memory_e2e_{uuid.uuid4().hex[:8]}"
  await _run_memory_e2e(
      MilvusMemoryServiceConfig(
          uri=os.environ["ZILLIZ_URI"],
          token=os.environ["ZILLIZ_TOKEN"],
          db_name=os.getenv("ZILLIZ_DB_NAME") or os.getenv("MILVUS_DB_NAME"),
          collection_name=collection_name,
          dimension=len(_VOCAB),
          consistency_level="Strong",
      )
  )
