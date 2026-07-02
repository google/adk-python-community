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
from google.genai import Client
from google.genai import types
import httpx
import pytest

from google.adk_community.memory.milvus_memory_service import MilvusMemoryService
from google.adk_community.memory.milvus_memory_service import MilvusMemoryServiceConfig

_VOCAB = ("milvus", "vector", "memory", "cooking")
_OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)
_OPENAI_EMBEDDING_DIMENSION = int(
    os.getenv("OPENAI_EMBEDDING_DIMENSION", "1536")
)
_GOOGLE_EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001"
)
_GOOGLE_EMBEDDING_DIMENSION = int(
    os.getenv("GOOGLE_EMBEDDING_DIMENSION", "3072")
)


def _keyword_embedding(texts):
  vectors = []
  for text in texts:
    lowered = text.lower()
    vector = [float(lowered.count(word)) for word in _VOCAB]
    if not any(vector):
      vector[-1] = 0.01
    vectors.append(vector)
  return vectors


def _openai_embedding(texts):
  response = httpx.post(
      "https://api.openai.com/v1/embeddings",
      headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
      json={
          "model": _OPENAI_EMBEDDING_MODEL,
          "input": list(texts),
      },
      timeout=30,
  )
  if response.status_code >= 400:
    raise RuntimeError(
        "OpenAI embeddings request failed with "
        f"{response.status_code}: {response.text[:500]}"
    )
  data = sorted(response.json()["data"], key=lambda item: item["index"])
  return [item["embedding"] for item in data]


def _google_embedding(texts):
  api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
  client = Client(api_key=api_key)
  try:
    response = client.models.embed_content(
        model=_GOOGLE_EMBEDDING_MODEL,
        contents=list(texts),
    )
  except Exception as exc:
    if "User location is not supported" in str(exc):
      pytest.skip("Google embeddings API is unavailable from this location.")
    raise
  return [list(embedding.values) for embedding in response.embeddings]


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


async def _run_memory_e2e(
    config: MilvusMemoryServiceConfig,
    embedding_function=_keyword_embedding,
) -> None:
  service = MilvusMemoryService(
      embedding_function=embedding_function,
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
    assert any(
        "Milvus stores vector memory" in memory.content.parts[0].text
        for memory in result.memories
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


def _lite_config(
    tmp_path: Path, *, dimension: int
) -> MilvusMemoryServiceConfig:
  collection_name = f"adk_memory_e2e_{uuid.uuid4().hex[:8]}"
  return MilvusMemoryServiceConfig(
      uri=str(tmp_path / "milvus_lite.db"),
      collection_name=collection_name,
      dimension=dimension,
      consistency_level="Strong",
  )


def _zilliz_config(*, dimension: int) -> MilvusMemoryServiceConfig:
  collection_name = f"adk_memory_e2e_{uuid.uuid4().hex[:8]}"
  return MilvusMemoryServiceConfig(
      uri=os.environ["ZILLIZ_URI"],
      token=os.environ["ZILLIZ_TOKEN"],
      db_name=os.getenv("ZILLIZ_DB_NAME") or os.getenv("MILVUS_DB_NAME"),
      collection_name=collection_name,
      dimension=dimension,
      consistency_level="Strong",
  )


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_MILVUS_LITE_E2E") != "1",
    reason="Set RUN_MILVUS_LITE_E2E=1 to run Milvus Lite E2E.",
)
async def test_milvus_lite_memory_e2e(tmp_path: Path):
  await _run_memory_e2e(
      _lite_config(tmp_path, dimension=len(_VOCAB)),
  )


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_MILVUS_LITE_E2E") != "1",
    reason="Set RUN_MILVUS_LITE_E2E=1 to run Milvus Lite E2E.",
)
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run OpenAI embeddings E2E.",
)
async def test_milvus_lite_memory_openai_embedding_e2e(tmp_path: Path):
  await _run_memory_e2e(
      _lite_config(tmp_path, dimension=_OPENAI_EMBEDDING_DIMENSION),
      embedding_function=_openai_embedding,
  )


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_MILVUS_LITE_E2E") != "1",
    reason="Set RUN_MILVUS_LITE_E2E=1 to run Milvus Lite E2E.",
)
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
    reason="Set GEMINI_API_KEY or GOOGLE_API_KEY to run Google embeddings E2E.",
)
async def test_milvus_lite_memory_google_embedding_e2e(tmp_path: Path):
  await _run_memory_e2e(
      _lite_config(tmp_path, dimension=_GOOGLE_EMBEDDING_DIMENSION),
      embedding_function=_google_embedding,
  )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("ZILLIZ_URI") or not os.getenv("ZILLIZ_TOKEN"),
    reason="Set ZILLIZ_URI and ZILLIZ_TOKEN to run Zilliz Cloud E2E.",
)
async def test_zilliz_cloud_memory_e2e():
  await _run_memory_e2e(
      _zilliz_config(dimension=len(_VOCAB)),
  )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("ZILLIZ_URI") or not os.getenv("ZILLIZ_TOKEN"),
    reason="Set ZILLIZ_URI and ZILLIZ_TOKEN to run Zilliz Cloud E2E.",
)
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run OpenAI embeddings E2E.",
)
async def test_zilliz_cloud_memory_openai_embedding_e2e():
  await _run_memory_e2e(
      _zilliz_config(dimension=_OPENAI_EMBEDDING_DIMENSION),
      embedding_function=_openai_embedding,
  )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("ZILLIZ_URI") or not os.getenv("ZILLIZ_TOKEN"),
    reason="Set ZILLIZ_URI and ZILLIZ_TOKEN to run Zilliz Cloud E2E.",
)
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
    reason="Set GEMINI_API_KEY or GOOGLE_API_KEY to run Google embeddings E2E.",
)
async def test_zilliz_cloud_memory_google_embedding_e2e():
  await _run_memory_e2e(
      _zilliz_config(dimension=_GOOGLE_EMBEDDING_DIMENSION),
      embedding_function=_google_embedding,
  )
