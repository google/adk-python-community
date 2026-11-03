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

import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.events.event import Event
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions.session import Session
from google.genai import types
import pytest

pytest.importorskip('qdrant_client')

from qdrant_client import AsyncQdrantClient
from qdrant_client import models

from google.adk_community.memory.qdrant_memory_service import QdrantMemoryService
from google.adk_community.memory.qdrant_memory_service import QdrantMemoryServiceConfig

APP = 'test-app'
USER = 'test-user'
DIM = 8


def _event(event_id, author, text=None, thought=None, timestamp=12345.0):
  parts = []
  if thought:
    parts.append(types.Part(text=thought, thought=True))
  if text:
    parts.append(types.Part(text=text))
  return Event(
      id=event_id,
      invocation_id=f'inv-{event_id}',
      author=author,
      timestamp=timestamp,
      content=types.Content(role='user', parts=parts) if parts else None,
  )


SESSION = Session(
    app_name=APP,
    user_id=USER,
    id='session-1',
    last_update_time=1000,
    events=[
        _event('e1', 'user', 'I like Python.', thought='secret reasoning'),
        _event('e2', 'model', 'Python is great.'),
        _event('e3', 'user'),  # no content, skipped
        Event(
            id='e4',
            invocation_id='inv-e4',
            author='model',
            content=types.Content(
                parts=[
                    types.Part(function_call=types.FunctionCall(name='tool'))
                ]
            ),
        ),  # no text, skipped
    ],
)


async def fake_embed(texts):
  """Deterministic bag-of-words embedding for tests."""
  vectors = []
  for text in texts:
    vec = [0.0] * DIM
    for word in text.lower().replace('.', ' ').split():
      vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    vectors.append(vec)
  return vectors


@pytest.fixture
def mock_client():
  client = MagicMock(spec=AsyncQdrantClient)
  client.collection_exists = AsyncMock(return_value=False)
  client.create_collection = AsyncMock()
  client.create_payload_index = AsyncMock()
  client.upsert = AsyncMock()
  client.delete = AsyncMock()
  client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))
  return client


@pytest.fixture
def service(mock_client):
  return QdrantMemoryService(
      client=mock_client, embedding_function=AsyncMock(side_effect=fake_embed)
  )


def test_requires_client_and_embedding_function(mock_client):
  with pytest.raises(ValueError):
    QdrantMemoryService(client=None, embedding_function=fake_embed)
  with pytest.raises(ValueError):
    QdrantMemoryService(client=mock_client, embedding_function=None)


def test_missing_qdrant_client_raises_import_error(mock_client):
  with patch.dict('sys.modules', {'qdrant_client': None}):
    with pytest.raises(ImportError, match=r'google-adk-community\[qdrant\]'):
      QdrantMemoryService(client=mock_client, embedding_function=fake_embed)


async def test_add_session_creates_collection_and_upserts(service, mock_client):
  await service.add_session_to_memory(SESSION)

  mock_client.create_collection.assert_awaited_once()
  kwargs = mock_client.create_collection.call_args.kwargs
  assert kwargs['collection_name'] == 'adk_memory'
  assert kwargs['vectors_config'].size == DIM
  assert kwargs['vectors_config'].distance == models.Distance.COSINE
  indexed = {
      c.kwargs['field_name']
      for c in mock_client.create_payload_index.call_args_list
  }
  assert indexed == {'app_name', 'user_id'}

  points = mock_client.upsert.call_args.kwargs['points']
  assert len(points) == 2
  payload = points[0].payload
  assert payload['app_name'] == APP
  assert payload['user_id'] == USER
  assert payload['session_id'] == 'session-1'
  assert payload['event_id'] == 'e1'
  assert payload['text'] == 'I like Python.'
  assert payload['content']['parts'] == [{'text': 'I like Python.'}]
  assert 'secret reasoning' not in str(payload)


async def test_collection_created_once(service, mock_client):
  await service.add_session_to_memory(SESSION)
  await service.add_session_to_memory(SESSION)
  assert mock_client.create_collection.await_count == 1
  assert mock_client.collection_exists.await_count == 1


async def test_point_ids_are_deterministic_and_scoped(service, mock_client):
  await service.add_session_to_memory(SESSION)
  first = [p.id for p in mock_client.upsert.call_args.kwargs['points']]
  await service.add_session_to_memory(SESSION)
  second = [p.id for p in mock_client.upsert.call_args.kwargs['points']]
  assert first == second

  other_user = SESSION.model_copy(update={'user_id': 'other'})
  await service.add_session_to_memory(other_user)
  third = [p.id for p in mock_client.upsert.call_args.kwargs['points']]
  assert set(first).isdisjoint(third)


async def test_no_create_when_disabled(mock_client):
  service = QdrantMemoryService(
      client=mock_client,
      embedding_function=fake_embed,
      config=QdrantMemoryServiceConfig(create_collection=False),
  )
  await service.add_session_to_memory(SESSION)
  mock_client.create_collection.assert_not_awaited()
  mock_client.upsert.assert_awaited_once()


async def test_configured_vector_size_and_distance(mock_client):
  service = QdrantMemoryService(
      client=mock_client,
      embedding_function=fake_embed,
      config=QdrantMemoryServiceConfig(vector_size=2 * DIM, distance='Dot'),
  )
  await service.add_session_to_memory(SESSION)
  config = mock_client.create_collection.call_args.kwargs['vectors_config']
  assert config.size == 2 * DIM  # configured size wins over inferred
  assert config.distance == models.Distance.DOT


async def test_empty_session_does_not_call_qdrant(service, mock_client):
  session = SESSION.model_copy(update={'events': [_event('x', 'user')]})
  await service.add_session_to_memory(session)
  mock_client.upsert.assert_not_awaited()
  service._embedding_function.assert_not_awaited()


async def test_embedding_count_mismatch_raises(mock_client):
  async def bad_embed(texts):
    return [[0.1] * DIM]

  service = QdrantMemoryService(
      client=mock_client, embedding_function=bad_embed
  )
  with pytest.raises(ValueError, match='returned 1 vectors for 2 texts'):
    await service.add_session_to_memory(SESSION)


async def test_add_events_to_memory(service, mock_client):
  await service.add_events_to_memory(
      app_name=APP,
      user_id=USER,
      events=[_event('e9', 'user', 'New fact.')],
      custom_metadata={'source': 'delta'},
  )
  [point] = mock_client.upsert.call_args.kwargs['points']
  assert point.payload['session_id'] is None
  assert point.payload['custom_metadata'] == {'source': 'delta'}


async def test_add_memory(service, mock_client):
  await service.add_memory(
      app_name=APP,
      user_id=USER,
      memories=[
          MemoryEntry(
              id='fact-1',
              content=types.Content(parts=[types.Part(text='Likes tea.')]),
              author='user',
              custom_metadata={'kind': 'preference'},
          ),
          MemoryEntry(content=types.Content(parts=[])),  # skipped
      ],
      custom_metadata={'source': 'explicit'},
  )
  [point] = mock_client.upsert.call_args.kwargs['points']
  assert point.payload['text'] == 'Likes tea.'
  assert point.payload['custom_metadata'] == {
      'kind': 'preference',
      'source': 'explicit',
  }


async def test_search_scopes_by_app_and_user(service, mock_client):
  mock_client.collection_exists.return_value = True
  mock_client.query_points.return_value = SimpleNamespace(
      points=[
          models.ScoredPoint(
              id='5c56c793-69f3-4fbf-87e6-c4bf54c28c26',
              version=1,
              score=0.9,
              payload={
                  'content': {'parts': [{'text': 'I like Python.'}]},
                  'author': 'user',
                  'timestamp': '2025-01-01T00:00:00',
                  'custom_metadata': {'k': 'v'},
              },
          )
      ]
  )

  result = await service.search_memory(
      app_name=APP, user_id=USER, query='python'
  )

  kwargs = mock_client.query_points.call_args.kwargs
  assert kwargs['limit'] == 10
  conditions = {c.key: c.match.value for c in kwargs['query_filter'].must}
  assert conditions == {'app_name': APP, 'user_id': USER}
  [memory] = result.memories
  assert memory.id == '5c56c793-69f3-4fbf-87e6-c4bf54c28c26'
  assert memory.content.parts[0].text == 'I like Python.'
  assert memory.author == 'user'
  assert memory.timestamp == '2025-01-01T00:00:00'
  assert memory.custom_metadata == {'k': 'v'}


async def test_search_without_collection_returns_empty(service, mock_client):
  result = await service.search_memory(app_name=APP, user_id=USER, query='x')
  assert result.memories == []
  mock_client.query_points.assert_not_awaited()


async def test_delete_memory_is_scoped(service, mock_client):
  mock_client.collection_exists.return_value = True
  memory_id = '5c56c793-69f3-4fbf-87e6-c4bf54c28c26'
  await service.delete_memory(app_name=APP, user_id=USER, memory_id=memory_id)
  selector = mock_client.delete.call_args.kwargs['points_selector']
  has_id, *fields = selector.filter.must
  assert has_id.has_id == [memory_id]
  assert {c.key: c.match.value for c in fields} == {
      'app_name': APP,
      'user_id': USER,
  }


@pytest.mark.parametrize('memory_id', [None, '', 'not-a-uuid'])
async def test_delete_memory_ignores_invalid_ids(
    service, mock_client, memory_id
):
  mock_client.collection_exists.return_value = True
  await service.delete_memory(app_name=APP, user_id=USER, memory_id=memory_id)
  mock_client.delete.assert_not_awaited()


@pytest.mark.filterwarnings('ignore:Payload indexes have no effect')
async def test_end_to_end_with_local_in_memory_qdrant():
  """Runs against qdrant-client's local in-memory mode (no server)."""
  client = AsyncQdrantClient(':memory:')
  service = QdrantMemoryService(client=client, embedding_function=fake_embed)
  try:
    await service.add_session_to_memory(SESSION)
    await service.add_session_to_memory(SESSION)  # idempotent
    await service.add_session_to_memory(
        SESSION.model_copy(update={'user_id': 'other'})
    )
    assert (await client.count('adk_memory')).count == 4

    result = await service.search_memory(
        app_name=APP, user_id=USER, query='I like Python.'
    )
    assert len(result.memories) == 2
    assert result.memories[0].content.parts[0].text == 'I like Python.'
    assert result.memories[0].author == 'user'

    await service.delete_memory(
        app_name=APP, user_id='other', memory_id=result.memories[0].id
    )
    assert (await client.count('adk_memory')).count == 4  # wrong scope

    await service.delete_memory(
        app_name=APP, user_id=USER, memory_id=result.memories[0].id
    )
    result = await service.search_memory(
        app_name=APP, user_id=USER, query='I like Python.'
    )
    assert [m.content.parts[0].text for m in result.memories] == [
        'Python is great.'
    ]
    assert (
        await service.search_memory(app_name='nope', user_id=USER, query='x')
    ).memories == []
  finally:
    await client.close()
