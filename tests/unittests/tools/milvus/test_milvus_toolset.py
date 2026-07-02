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

from unittest.mock import MagicMock

import pytest

from google.adk_community.tools.milvus import milvus_toolset as milvus_module
from google.adk_community.tools.milvus import MilvusToolset
from google.adk_community.tools.milvus import MilvusToolSettings
from google.adk_community.tools.milvus import MilvusVectorStore
from google.adk_community.tools.milvus import MilvusVectorStoreSettings


class FakeDataType:
  VARCHAR = "VARCHAR"
  FLOAT_VECTOR = "FLOAT_VECTOR"
  JSON = "JSON"


class FakeSchema:

  def __init__(self):
    self.fields = []

  def add_field(self, **kwargs):
    self.fields.append(kwargs)


class FakeIndexParams:

  def __init__(self):
    self.indexes = []

  def add_index(self, **kwargs):
    self.indexes.append(kwargs)


@pytest.fixture
def fake_milvus(monkeypatch):
  client = MagicMock()
  client.has_collection.return_value = False
  client.create_schema.return_value = FakeSchema()
  client.prepare_index_params.return_value = FakeIndexParams()
  client.describe_collection.return_value = {
      "auto_id": False,
      "fields": [
          {"name": "id", "type": FakeDataType.VARCHAR, "is_primary": True},
          {
              "name": "embedding",
              "type": FakeDataType.FLOAT_VECTOR,
              "params": {"dim": 3},
          },
          {"name": "content", "type": FakeDataType.VARCHAR},
          {"name": "source", "type": FakeDataType.VARCHAR},
          {"name": "metadata", "type": FakeDataType.JSON},
      ],
  }

  class FakeMilvusClient:

    def __new__(cls, **kwargs):
      client.init_kwargs = kwargs
      return client

  monkeypatch.setattr(
      milvus_module,
      "_load_pymilvus",
      lambda: (FakeMilvusClient, FakeDataType),
  )
  return client


def embedding_function(texts):
  return [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(texts)]


async def async_embedding_function(texts):
  return embedding_function(texts)


def create_settings(**kwargs):
  return MilvusVectorStoreSettings(
      uri="memory.db",
      collection_name="rag_collection",
      dimension=3,
      **kwargs,
  )


def create_vector_store(fake_milvus, **kwargs):
  _ = fake_milvus
  return MilvusVectorStore(
      embedding_function=embedding_function,
      settings=create_settings(**kwargs),
  )


def test_vector_store_creates_collection(fake_milvus):
  client = fake_milvus
  vector_store = create_vector_store(fake_milvus, consistency_level="Strong")

  assert vector_store is not None
  assert client.init_kwargs == {"uri": "memory.db"}
  client.create_collection.assert_called_once()
  create_kwargs = client.create_collection.call_args.kwargs
  assert create_kwargs["collection_name"] == "rag_collection"
  assert create_kwargs["consistency_level"] == "Strong"
  schema = client.create_schema.return_value
  assert {field["field_name"] for field in schema.fields} == {
      "id",
      "embedding",
      "content",
      "source",
      "metadata",
  }
  assert client.prepare_index_params.return_value.indexes == [{
      "field_name": "embedding",
      "index_type": "AUTOINDEX",
      "metric_type": "COSINE",
  }]


def test_existing_collection_dimension_mismatch_raises(fake_milvus):
  client = fake_milvus
  client.has_collection.return_value = True
  client.describe_collection.return_value["fields"][1]["params"] = {"dim": 8}

  with pytest.raises(ValueError, match="vector dimension 8"):
    create_vector_store(fake_milvus)


def test_invalid_field_name_raises(fake_milvus):
  _ = fake_milvus
  settings = create_settings(content_field="bad-name")

  with pytest.raises(ValueError, match="valid identifiers"):
    MilvusVectorStore(
        embedding_function=embedding_function,
        settings=settings,
    )


def test_add_texts_upserts_records(fake_milvus):
  client = fake_milvus
  vector_store = create_vector_store(fake_milvus)

  result = vector_store.add_texts(
      ["Milvus stores vectors.", "ADK tools retrieve context."],
      metadatas=[
          {"source": "doc-1", "section": 1},
          {"source": "doc-2", "section": 2},
      ],
      ids=["id-1", "id-2"],
  )

  assert result == {"status": "SUCCESS", "inserted_count": 2}
  client.upsert.assert_called_once()
  records = client.upsert.call_args.kwargs["data"]
  assert records[0] == {
      "id": "id-1",
      "embedding": [1.0, 0.0, 0.0],
      "content": "Milvus stores vectors.",
      "source": "doc-1",
      "metadata": {"source": "doc-1", "section": 1},
  }
  assert records[1]["embedding"] == [2.0, 0.0, 0.0]


def test_add_texts_metadata_length_mismatch_raises(fake_milvus):
  vector_store = create_vector_store(fake_milvus)

  with pytest.raises(ValueError, match="one item per content"):
    vector_store.add_texts(["one", "two"], metadatas=[{}])


def test_similarity_search_returns_rows(fake_milvus):
  client = fake_milvus
  vector_store = create_vector_store(fake_milvus, search_top_k=3)
  client.search.return_value = [[{
      "id": "id-1",
      "distance": 0.12,
      "entity": {
          "content": "Milvus stores vectors.",
          "source": "doc-1",
          "metadata": {"section": 1},
      },
  }]]

  result = vector_store.similarity_search("vector database")

  assert result == {
      "status": "SUCCESS",
      "rows": [{
          "id": "id-1",
          "content": "Milvus stores vectors.",
          "source": "doc-1",
          "metadata": {"section": 1},
          "distance": 0.12,
      }],
  }
  search_kwargs = client.search.call_args.kwargs
  assert search_kwargs["collection_name"] == "rag_collection"
  assert search_kwargs["data"] == [[1.0, 0.0, 0.0]]
  assert search_kwargs["limit"] == 3
  assert "filter" not in search_kwargs


@pytest.mark.asyncio
async def test_toolset_returns_prefixed_similarity_search_tool(fake_milvus):
  client = fake_milvus
  client.search.return_value = [[{
      "id": "id-1",
      "distance": 0.12,
      "entity": {
          "content": "Milvus stores vectors.",
          "source": "doc-1",
          "metadata": {},
      },
  }]]
  toolset = MilvusToolset(
      embedding_function=async_embedding_function,
      milvus_tool_settings=MilvusToolSettings(
          vector_store_settings=create_settings()
      ),
  )

  tools = await toolset.get_tools_with_prefix()

  assert [tool.name for tool in tools] == ["milvus_similarity_search"]
  result = await tools[0].run_async(
      args={"query": "vector database"},
      tool_context=None,
  )
  assert result["status"] == "SUCCESS"
  assert result["rows"][0]["content"] == "Milvus stores vectors."


@pytest.mark.asyncio
async def test_toolset_reuses_vector_store_with_custom_description(
    fake_milvus,
):
  vector_store = create_vector_store(fake_milvus)
  toolset = MilvusToolset(
      vector_store=vector_store,
      milvus_tool_settings=MilvusToolSettings(
          similarity_search_description="Search indexed support content.",
      ),
  )

  tools = await toolset.get_tools()

  assert [tool.name for tool in tools] == ["similarity_search"]
  assert tools[0].description == "Search indexed support content."


@pytest.mark.asyncio
async def test_toolset_filter_can_include_similarity_search(fake_milvus):
  toolset = MilvusToolset(
      embedding_function=embedding_function,
      milvus_tool_settings=MilvusToolSettings(
          vector_store_settings=create_settings()
      ),
      tool_filter=["similarity_search"],
  )

  tools = await toolset.get_tools()

  assert [tool.name for tool in tools] == ["similarity_search"]


@pytest.mark.asyncio
async def test_toolset_filter_can_exclude_similarity_search(fake_milvus):
  toolset = MilvusToolset(
      embedding_function=embedding_function,
      milvus_tool_settings=MilvusToolSettings(
          vector_store_settings=create_settings()
      ),
      tool_filter=["other_tool"],
  )

  assert await toolset.get_tools() == []


@pytest.mark.asyncio
async def test_close_closes_client(fake_milvus):
  client = fake_milvus
  vector_store = create_vector_store(fake_milvus)

  await vector_store.close()

  client.close.assert_called_once()
