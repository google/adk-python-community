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

from google.adk.events.event import Event
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions.session import Session
from google.genai import types
import pytest

from google.adk_community.memory import milvus_memory_service as milvus_module
from google.adk_community.memory.milvus_memory_service import MilvusMemoryService
from google.adk_community.memory.milvus_memory_service import MilvusMemoryServiceConfig


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
  schema = FakeSchema()
  index_params = FakeIndexParams()
  client.create_schema.return_value = schema
  client.prepare_index_params.return_value = index_params
  client.describe_collection.return_value = {
      "auto_id": False,
      "fields": [
          {"name": "id", "type": FakeDataType.VARCHAR, "is_primary": True},
          {
              "name": "embedding",
              "type": FakeDataType.FLOAT_VECTOR,
              "params": {"dim": 3},
          },
          {"name": "text", "type": FakeDataType.VARCHAR},
          {"name": "app_name", "type": FakeDataType.VARCHAR},
          {"name": "user_id", "type": FakeDataType.VARCHAR},
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
  return client, schema, index_params


def embedding_function(texts):
  return [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(texts)]


async def async_embedding_function(texts):
  return embedding_function(texts)


def create_service(fake_milvus, **kwargs):
  _ = fake_milvus
  return MilvusMemoryService(
      embedding_function=embedding_function,
      dimension=3,
      uri="memory.db",
      **kwargs,
  )


def test_constructor_creates_collection(fake_milvus):
  client, schema, index_params = fake_milvus

  service = MilvusMemoryService(
      embedding_function=embedding_function,
      dimension=3,
      uri="memory.db",
      token="token",
      db_name="default",
      collection_name="memory_collection",
      consistency_level="Strong",
  )

  assert service is not None
  assert client.init_kwargs == {
      "uri": "memory.db",
      "token": "token",
      "db_name": "default",
  }
  client.create_collection.assert_called_once()
  create_kwargs = client.create_collection.call_args.kwargs
  assert create_kwargs["collection_name"] == "memory_collection"
  assert create_kwargs["consistency_level"] == "Strong"
  assert {field["field_name"] for field in schema.fields} >= {
      "id",
      "embedding",
      "text",
      "app_name",
      "user_id",
      "metadata",
  }
  assert index_params.indexes == [{
      "field_name": "embedding",
      "index_type": "AUTOINDEX",
      "metric_type": "COSINE",
  }]


def test_existing_collection_dimension_mismatch_raises(fake_milvus):
  client, _, _ = fake_milvus
  client.has_collection.return_value = True
  client.describe_collection.return_value = {
      "auto_id": False,
      "fields": [
          {"name": "id", "type": FakeDataType.VARCHAR, "is_primary": True},
          {
              "name": "embedding",
              "type": FakeDataType.FLOAT_VECTOR,
              "params": {"dim": 8},
          },
          {"name": "text", "type": FakeDataType.VARCHAR},
          {"name": "app_name", "type": FakeDataType.VARCHAR},
          {"name": "user_id", "type": FakeDataType.VARCHAR},
      ],
  }

  with pytest.raises(ValueError, match="vector dimension 8"):
    create_service(fake_milvus)


def test_existing_collection_auto_id_raises(fake_milvus):
  client, _, _ = fake_milvus
  client.has_collection.return_value = True
  client.describe_collection.return_value = {
      "auto_id": True,
      "fields": [
          {"name": "id", "type": FakeDataType.VARCHAR, "is_primary": True},
          {
              "name": "embedding",
              "type": FakeDataType.FLOAT_VECTOR,
              "params": {"dim": 3},
          },
          {"name": "text", "type": FakeDataType.VARCHAR},
          {"name": "app_name", "type": FakeDataType.VARCHAR},
          {"name": "user_id", "type": FakeDataType.VARCHAR},
      ],
  }

  with pytest.raises(ValueError, match="auto_id=False"):
    create_service(fake_milvus)


def test_existing_collection_primary_key_mismatch_raises(fake_milvus):
  client, _, _ = fake_milvus
  client.has_collection.return_value = True
  client.describe_collection.return_value = {
      "auto_id": False,
      "fields": [
          {"name": "id", "type": FakeDataType.VARCHAR, "is_primary": False},
          {
              "name": "embedding",
              "type": FakeDataType.FLOAT_VECTOR,
              "params": {"dim": 3},
          },
          {"name": "text", "type": FakeDataType.VARCHAR},
          {"name": "app_name", "type": FakeDataType.VARCHAR},
          {"name": "user_id", "type": FakeDataType.VARCHAR},
      ],
  }

  with pytest.raises(ValueError, match="primary key"):
    create_service(fake_milvus)


def test_existing_collection_type_mismatch_raises(fake_milvus):
  client, _, _ = fake_milvus
  client.has_collection.return_value = True
  client.describe_collection.return_value = {
      "auto_id": False,
      "fields": [
          {"name": "id", "type": FakeDataType.VARCHAR, "is_primary": True},
          {
              "name": "embedding",
              "type": FakeDataType.VARCHAR,
              "params": {"dim": 3},
          },
          {"name": "text", "type": FakeDataType.VARCHAR},
          {"name": "app_name", "type": FakeDataType.VARCHAR},
          {"name": "user_id", "type": FakeDataType.VARCHAR},
      ],
  }

  with pytest.raises(ValueError, match="embedding.*expected FLOAT_VECTOR"):
    create_service(fake_milvus)


def test_invalid_field_name_raises(fake_milvus):
  _ = fake_milvus
  config = MilvusMemoryServiceConfig(
      dimension=3,
      app_name_field="app-name",
  )

  with pytest.raises(ValueError, match="valid identifiers"):
    MilvusMemoryService(
        embedding_function=embedding_function,
        config=config,
        uri="memory.db",
    )


def test_duplicate_field_name_raises(fake_milvus):
  _ = fake_milvus
  config = MilvusMemoryServiceConfig(
      dimension=3,
      user_id_field="app_name",
  )

  with pytest.raises(ValueError, match="must be unique"):
    MilvusMemoryService(
        embedding_function=embedding_function,
        config=config,
        uri="memory.db",
    )


@pytest.mark.asyncio
async def test_add_session_to_memory_upserts_text_events(fake_milvus):
  client, _, _ = fake_milvus
  service = create_service(fake_milvus)
  session = Session(
      app_name="test-app",
      user_id="test-user",
      id="session-1",
      last_update_time=1000,
      events=[
          Event(
              id="event-1",
              invocation_id="inv-1",
              author="user",
              timestamp=12345,
              content=types.Content(
                  parts=[types.Part(text="Milvus stores vectors.")]
              ),
          ),
          Event(
              id="event-2",
              invocation_id="inv-2",
              author="model",
              timestamp=12346,
              content=types.Content(
                  parts=[types.Part(text="Semantic search is supported.")]
              ),
          ),
          Event(id="event-empty", author="user", timestamp=12347),
          Event(
              id="event-tool",
              author="agent",
              timestamp=12348,
              content=types.Content(
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(name="lookup")
                      )
                  ]
              ),
          ),
      ],
  )

  await service.add_session_to_memory(session)

  client.upsert.assert_called_once()
  records = client.upsert.call_args.kwargs["data"]
  assert len(records) == 2
  assert records[0]["app_name"] == "test-app"
  assert records[0]["user_id"] == "test-user"
  assert records[0]["session_id"] == "session-1"
  assert records[0]["event_id"] == "event-1"
  assert records[0]["source"] == "adk_event"
  assert records[0]["metadata"]["invocation_id"] == "inv-1"
  assert records[1]["embedding"] == [2.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_add_memory_upserts_direct_memory(fake_milvus):
  client, _, _ = fake_milvus
  service = create_service(fake_milvus)
  memory = MemoryEntry(
      id="memory-1",
      author="user",
      timestamp="2026-01-01T00:00:00Z",
      content=types.Content(parts=[types.Part(text="Remember Milvus.")]),
      custom_metadata={"kind": "fact"},
  )

  await service.add_memory(
      app_name="test-app",
      user_id="test-user",
      memories=[memory],
      custom_metadata={"source_name": "manual"},
  )

  records = client.upsert.call_args.kwargs["data"]
  assert records == [{
      "id": "memory-1",
      "embedding": [1.0, 0.0, 0.0],
      "text": "Remember Milvus.",
      "app_name": "test-app",
      "user_id": "test-user",
      "session_id": "",
      "event_id": "",
      "author": "user",
      "timestamp": "2026-01-01T00:00:00Z",
      "source": "adk_memory",
      "metadata": {
          "source_name": "manual",
          "kind": "fact",
          "source": "adk_memory",
      },
  }]


@pytest.mark.asyncio
async def test_search_memory_returns_entries(fake_milvus):
  client, _, _ = fake_milvus
  service = MilvusMemoryService(
      embedding_function=async_embedding_function,
      dimension=3,
      uri="memory.db",
      search_top_k=3,
  )
  client.search.return_value = [[{
      "entity": {
          "text": "Milvus supports semantic memory.",
          "author": "user",
          "timestamp": "2026-01-01T00:00:00Z",
          "metadata": {"source": "adk_event"},
      }
  }]]

  result = await service.search_memory(
      app_name='app "quoted"',
      user_id="user-1",
      query="semantic memory",
  )

  assert len(result.memories) == 1
  assert result.memories[0].content.parts[0].text == (
      "Milvus supports semantic memory."
  )
  assert result.memories[0].custom_metadata == {"source": "adk_event"}
  search_kwargs = client.search.call_args.kwargs
  assert search_kwargs["filter"] == (
      'app_name == "app \\"quoted\\"" and user_id == "user-1"'
  )
  assert search_kwargs["limit"] == 3
  assert search_kwargs["data"] == [[1.0, 0.0, 0.0]]


@pytest.mark.asyncio
async def test_embedding_dimension_mismatch_raises(fake_milvus):
  client, _, _ = fake_milvus

  def bad_embedding_function(texts):
    return [[1.0] for _ in texts]

  service = MilvusMemoryService(
      embedding_function=bad_embedding_function,
      dimension=3,
      uri="memory.db",
  )

  with pytest.raises(ValueError, match="vector dimension 1"):
    await service.add_memory(
        app_name="test-app",
        user_id="test-user",
        memories=[
            MemoryEntry(
                content=types.Content(parts=[types.Part(text="Bad vector.")])
            )
        ],
    )
  client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_close_closes_client(fake_milvus):
  client, _, _ = fake_milvus
  service = create_service(fake_milvus)

  await service.close()

  client.close.assert_called_once()
