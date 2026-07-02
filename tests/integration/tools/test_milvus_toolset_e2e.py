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

from google.genai import Client
import httpx
import pytest

from google.adk_community.tools.milvus import MilvusToolset
from google.adk_community.tools.milvus import MilvusVectorStore
from google.adk_community.tools.milvus import MilvusVectorStoreSettings

_VOCAB = ("milvus", "local", "cloud", "production")
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


def _lite_settings(
    tmp_path: Path, *, dimension: int
) -> MilvusVectorStoreSettings:
  return MilvusVectorStoreSettings(
      uri=str(tmp_path / "milvus_toolset.db"),
      collection_name=f"adk_rag_e2e_{uuid.uuid4().hex[:8]}",
      dimension=dimension,
      consistency_level="Strong",
  )


def _zilliz_settings(*, dimension: int) -> MilvusVectorStoreSettings:
  return MilvusVectorStoreSettings(
      uri=os.environ["ZILLIZ_URI"],
      token=os.environ["ZILLIZ_TOKEN"],
      db_name=os.getenv("ZILLIZ_DB_NAME") or os.getenv("MILVUS_DB_NAME"),
      collection_name=f"adk_rag_e2e_{uuid.uuid4().hex[:8]}",
      dimension=dimension,
      consistency_level="Strong",
  )


async def _drop_collection(vector_store: MilvusVectorStore) -> None:
  try:
    await asyncio.to_thread(
        vector_store._client.drop_collection,  # pylint: disable=protected-access
        collection_name=vector_store._settings.collection_name,  # pylint: disable=protected-access
    )
  except Exception:
    pass


async def _run_toolset_e2e(
    settings: MilvusVectorStoreSettings,
    embedding_function,
) -> None:
  vector_store = MilvusVectorStore(
      embedding_function=embedding_function,
      settings=settings,
  )
  try:
    await vector_store.add_texts_async(
        [
            "Milvus Lite is useful for local RAG development.",
            "Zilliz Cloud provides managed Milvus for production workloads.",
        ],
        metadatas=[
            {"source": "milvus-lite"},
            {"source": "zilliz-cloud"},
        ],
        ids=["local-doc", "cloud-doc"],
    )
    toolset = MilvusToolset(vector_store=vector_store)
    tools = await toolset.get_tools_with_prefix()
    assert [tool.name for tool in tools] == ["milvus_similarity_search"]

    result = await tools[0].run_async(
        args={"query": "managed cloud production Milvus"},
        tool_context=None,
    )

    assert result["status"] == "SUCCESS"
    assert any(
        "Zilliz Cloud provides managed Milvus" in row["content"]
        for row in result["rows"]
    )
  finally:
    await _drop_collection(vector_store)
    await vector_store.close()


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_MILVUS_LITE_E2E") != "1",
    reason="Set RUN_MILVUS_LITE_E2E=1 to run Milvus Lite E2E.",
)
async def test_milvus_lite_toolset_e2e(tmp_path: Path):
  await _run_toolset_e2e(
      _lite_settings(tmp_path, dimension=len(_VOCAB)),
      _keyword_embedding,
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
@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_EMBEDDING_E2E") != "1",
    reason="Set RUN_OPENAI_EMBEDDING_E2E=1 to run OpenAI embeddings E2E.",
)
async def test_milvus_lite_toolset_openai_embedding_e2e(tmp_path: Path):
  await _run_toolset_e2e(
      _lite_settings(tmp_path, dimension=_OPENAI_EMBEDDING_DIMENSION),
      _openai_embedding,
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
@pytest.mark.skipif(
    os.getenv("RUN_GOOGLE_EMBEDDING_E2E") != "1",
    reason="Set RUN_GOOGLE_EMBEDDING_E2E=1 to run Google embeddings E2E.",
)
async def test_milvus_lite_toolset_google_embedding_e2e(tmp_path: Path):
  await _run_toolset_e2e(
      _lite_settings(tmp_path, dimension=_GOOGLE_EMBEDDING_DIMENSION),
      _google_embedding,
  )


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_ZILLIZ_CLOUD_E2E") != "1",
    reason="Set RUN_ZILLIZ_CLOUD_E2E=1 to run Zilliz Cloud E2E.",
)
@pytest.mark.skipif(
    not os.getenv("ZILLIZ_URI") or not os.getenv("ZILLIZ_TOKEN"),
    reason="Set ZILLIZ_URI and ZILLIZ_TOKEN to run Zilliz Cloud E2E.",
)
async def test_zilliz_cloud_toolset_e2e():
  await _run_toolset_e2e(
      _zilliz_settings(dimension=len(_VOCAB)),
      _keyword_embedding,
  )


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_ZILLIZ_CLOUD_E2E") != "1",
    reason="Set RUN_ZILLIZ_CLOUD_E2E=1 to run Zilliz Cloud E2E.",
)
@pytest.mark.skipif(
    not os.getenv("ZILLIZ_URI") or not os.getenv("ZILLIZ_TOKEN"),
    reason="Set ZILLIZ_URI and ZILLIZ_TOKEN to run Zilliz Cloud E2E.",
)
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run OpenAI embeddings E2E.",
)
@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_EMBEDDING_E2E") != "1",
    reason="Set RUN_OPENAI_EMBEDDING_E2E=1 to run OpenAI embeddings E2E.",
)
async def test_zilliz_cloud_toolset_openai_embedding_e2e():
  await _run_toolset_e2e(
      _zilliz_settings(dimension=_OPENAI_EMBEDDING_DIMENSION),
      _openai_embedding,
  )


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_ZILLIZ_CLOUD_E2E") != "1",
    reason="Set RUN_ZILLIZ_CLOUD_E2E=1 to run Zilliz Cloud E2E.",
)
@pytest.mark.skipif(
    not os.getenv("ZILLIZ_URI") or not os.getenv("ZILLIZ_TOKEN"),
    reason="Set ZILLIZ_URI and ZILLIZ_TOKEN to run Zilliz Cloud E2E.",
)
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
    reason="Set GEMINI_API_KEY or GOOGLE_API_KEY to run Google embeddings E2E.",
)
@pytest.mark.skipif(
    os.getenv("RUN_GOOGLE_EMBEDDING_E2E") != "1",
    reason="Set RUN_GOOGLE_EMBEDDING_E2E=1 to run Google embeddings E2E.",
)
async def test_zilliz_cloud_toolset_google_embedding_e2e():
  await _run_toolset_e2e(
      _zilliz_settings(dimension=_GOOGLE_EMBEDDING_DIMENSION),
      _google_embedding,
  )
