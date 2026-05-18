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

import asyncio
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.session import Session
import pytest
import pytest_asyncio

from google.adk_community.sessions.typesense_session_service import _session_doc_id
from google.adk_community.sessions.typesense_session_service import _user_state_doc_id
from google.adk_community.sessions.typesense_session_service import TypesenseSessionService


def _make_typesense_mock():
  """Build a minimal mock of the typesense module and its Client."""
  mock_ts = MagicMock()

  class _FakeObjectNotFound(Exception):
    pass

  mock_ts.exceptions.ObjectNotFound = _FakeObjectNotFound
  mock_client = MagicMock()
  mock_ts.Client.return_value = mock_client
  return mock_ts, mock_client, _FakeObjectNotFound


def _make_service(mock_ts, mock_client, ObjectNotFound):
  """Instantiate TypesenseSessionService bypassing __init__ for unit tests."""
  svc = TypesenseSessionService.__new__(TypesenseSessionService)
  svc._typesense = mock_ts
  svc._ObjectNotFound = ObjectNotFound
  svc._client = mock_client
  svc._sessions_col = 'adk_sessions'
  svc._events_col = 'adk_events'
  svc._app_states_col = 'adk_app_states'
  svc._user_states_col = 'adk_user_states'
  svc._collections_ready = True  # skip bootstrap in unit tests
  svc._collections_lock = asyncio.Lock()
  return svc


def _make_session_doc(
    app_name, user_id, session_id, state=None, update_time=1000.0
):
  doc_id = _session_doc_id(app_name, user_id, session_id)
  return {
      'id': doc_id,
      'app_name': app_name,
      'user_id': user_id,
      'session_id': session_id,
      'state': json.dumps(state or {}),
      'update_time': update_time,
  }


@pytest_asyncio.fixture
def ts_mock():
  return _make_typesense_mock()


@pytest_asyncio.fixture
def service(ts_mock):
  mock_ts, mock_client, ObjectNotFound = ts_mock
  return _make_service(mock_ts, mock_client, ObjectNotFound)


class TestTypesenseSessionService:

  @pytest.mark.asyncio
  async def test_create_session_returns_session(self, service, ts_mock):
    _, mock_client, _ = ts_mock
    service._get_doc_sync = MagicMock(return_value=None)
    mock_client.collections.__getitem__.return_value.documents.create = (
        MagicMock()
    )
    mock_client.collections.__getitem__.return_value.documents.upsert = (
        MagicMock()
    )

    session = await service.create_session(
        app_name='my_app',
        user_id='user1',
        state={'key': 'value'},
        session_id='sess1',
    )

    assert session.app_name == 'my_app'
    assert session.user_id == 'user1'
    assert session.id == 'sess1'
    assert session.state.get('key') == 'value'
    assert session.events == []

  @pytest.mark.asyncio
  async def test_create_session_generates_id_when_none(self, service, ts_mock):
    _, mock_client, _ = ts_mock
    service._get_doc_sync = MagicMock(return_value=None)
    mock_client.collections.__getitem__.return_value.documents.create = (
        MagicMock()
    )

    session = await service.create_session(app_name='app', user_id='u')

    assert session.id  # auto-generated UUID

  @pytest.mark.asyncio
  async def test_create_session_raises_if_already_exists(self, service):
    existing_doc = _make_session_doc('app', 'u', 'sess1')
    service._get_doc_sync = MagicMock(return_value=existing_doc)

    from google.adk.errors.already_exists_error import AlreadyExistsError

    with pytest.raises(AlreadyExistsError):
      await service.create_session(
          app_name='app', user_id='u', session_id='sess1'
      )

  @pytest.mark.asyncio
  async def test_create_session_splits_state_scopes(self, service, ts_mock):
    _, mock_client, _ = ts_mock
    upserted: list[dict] = []
    service._get_doc_sync = MagicMock(return_value=None)
    mock_client.collections.__getitem__.return_value.documents.upsert = (
        MagicMock(side_effect=lambda doc: upserted.append(doc))
    )
    mock_client.collections.__getitem__.return_value.documents.create = (
        MagicMock()
    )

    await service.create_session(
        app_name='app',
        user_id='u',
        session_id='s',
        state={
            'app:shared': 'app_val',
            'user:pref': 'user_val',
            'local': 'session_val',
            'temp:scratch': 'ignored',
        },
    )

    assert len(upserted) == 2
    app_upsert = next(d for d in upserted if 'user_id' not in d)
    assert json.loads(app_upsert['state']) == {'shared': 'app_val'}
    user_upsert = next(d for d in upserted if 'user_id' in d)
    assert json.loads(user_upsert['state']) == {'pref': 'user_val'}

  @pytest.mark.asyncio
  async def test_get_session_returns_none_if_missing(self, service):
    service._get_doc_sync = MagicMock(return_value=None)

    result = await service.get_session(
        app_name='app', user_id='u', session_id='nonexistent'
    )
    assert result is None

  @pytest.mark.asyncio
  async def test_get_session_merges_state_scopes(self, service, ts_mock):
    _, mock_client, _ = ts_mock

    session_doc = _make_session_doc('app', 'u', 's', state={'local': 'v'})
    app_state_doc = {
        'id': 'app',
        'app_name': 'app',
        'state': json.dumps({'shared': 'app_val'}),
        'update_time': 999.0,
    }
    user_state_doc = {
        'id': 'app||u',
        'app_name': 'app',
        'user_id': 'u',
        'state': json.dumps({'pref': 'user_val'}),
        'update_time': 999.0,
    }

    docs = {
        (service._sessions_col, _session_doc_id('app', 'u', 's')): session_doc,
        (service._app_states_col, 'app'): app_state_doc,
        (
            service._user_states_col,
            _user_state_doc_id('app', 'u'),
        ): user_state_doc,
    }
    service._get_doc_sync = MagicMock(
        side_effect=lambda col, did: docs.get((col, did))
    )
    mock_client.collections.__getitem__.return_value.documents.search = (
        MagicMock(return_value={'hits': []})
    )

    session = await service.get_session(
        app_name='app', user_id='u', session_id='s'
    )

    assert session is not None
    assert session.state['local'] == 'v'
    assert session.state['app:shared'] == 'app_val'
    assert session.state['user:pref'] == 'user_val'

  @pytest.mark.asyncio
  async def test_get_session_with_num_recent_events(self, service, ts_mock):
    _, mock_client, _ = ts_mock

    session_doc = _make_session_doc('app', 'u', 's')
    service._get_doc_sync = MagicMock(
        side_effect=lambda col, did: session_doc
        if col == service._sessions_col
        else None
    )

    event_json = Event(author='user', timestamp=1.0).model_dump_json(
        exclude_none=True
    )
    mock_client.collections.__getitem__.return_value.documents.search = (
        MagicMock(
            return_value={'hits': [{'document': {'event_data': event_json}}]}
        )
    )

    config = GetSessionConfig(num_recent_events=3)
    session = await service.get_session(
        app_name='app', user_id='u', session_id='s', config=config
    )

    assert len(session.events) == 1
    params = mock_client.collections.__getitem__.return_value.documents.search.call_args[
        0
    ][
        0
    ]
    assert params['per_page'] == 3
    assert params['sort_by'] == 'timestamp:desc'

  @pytest.mark.asyncio
  async def test_get_session_num_recent_events_zero_skips_search(
      self, service, ts_mock
  ):
    _, mock_client, _ = ts_mock

    session_doc = _make_session_doc('app', 'u', 's')
    service._get_doc_sync = MagicMock(
        side_effect=lambda col, did: session_doc
        if col == service._sessions_col
        else None
    )

    config = GetSessionConfig(num_recent_events=0)
    session = await service.get_session(
        app_name='app', user_id='u', session_id='s', config=config
    )

    assert session.events == []
    mock_client.collections.__getitem__.return_value.documents.search.assert_not_called()

  @pytest.mark.asyncio
  async def test_list_sessions_returns_all_for_app(self, service):
    docs = [
        _make_session_doc('app', 'u1', 's1'),
        _make_session_doc('app', 'u2', 's2'),
    ]

    async def fake_search_all(collection, filter_by, sort_by=None):
      return docs if collection == service._sessions_col else []

    service._search_all = fake_search_all
    service._get_app_state = AsyncMock(return_value={})
    service._get_user_state = AsyncMock(return_value={})

    response = await service.list_sessions(app_name='app')

    assert len(response.sessions) == 2
    assert {s.id for s in response.sessions} == {'s1', 's2'}
    for s in response.sessions:
      assert s.events == []

  @pytest.mark.asyncio
  async def test_list_sessions_filtered_by_user(self, service):
    docs = [_make_session_doc('app', 'u1', 's1')]

    async def fake_search_all(collection, filter_by, sort_by=None):
      if collection == service._sessions_col:
        assert 'u1' in filter_by
        return docs
      return []

    service._search_all = fake_search_all
    service._get_app_state = AsyncMock(return_value={})
    service._get_user_state = AsyncMock(return_value={})

    response = await service.list_sessions(app_name='app', user_id='u1')

    assert len(response.sessions) == 1
    assert response.sessions[0].user_id == 'u1'

  @pytest.mark.asyncio
  async def test_delete_session_removes_events_and_doc(self, service, ts_mock):
    _, mock_client, _ = ts_mock
    mock_client.collections.__getitem__.return_value.documents.delete = (
        MagicMock()
    )
    mock_client.collections.__getitem__.return_value.documents.__getitem__.return_value.delete = (
        MagicMock()
    )

    await service.delete_session(app_name='app', user_id='u', session_id='s')

    mock_client.collections.__getitem__.return_value.documents.delete.assert_called_once()
    mock_client.collections.__getitem__.return_value.documents.__getitem__.return_value.delete.assert_called_once()

  @pytest.mark.asyncio
  async def test_delete_session_ignores_missing_doc(self, service, ts_mock):
    _, mock_client, ObjectNotFound = ts_mock
    mock_client.collections.__getitem__.return_value.documents.delete = (
        MagicMock()
    )
    mock_client.collections.__getitem__.return_value.documents.__getitem__.return_value.delete = MagicMock(
        side_effect=ObjectNotFound()
    )

    # Must not raise
    await service.delete_session(
        app_name='app', user_id='u', session_id='missing'
    )

  @pytest.mark.asyncio
  async def test_append_event_persists_event(self, service, ts_mock):
    _, mock_client, _ = ts_mock
    session_doc = _make_session_doc('app', 'u', 's', update_time=1.0)
    service._get_doc_sync = MagicMock(return_value=session_doc)
    mock_client.collections.__getitem__.return_value.documents.__getitem__.return_value.update = (
        MagicMock()
    )
    mock_client.collections.__getitem__.return_value.documents.create = (
        MagicMock()
    )

    session = Session(
        app_name='app',
        user_id='u',
        id='s',
        state={},
        events=[],
        last_update_time=1.0,
    )
    event = Event(author='user', timestamp=2.0)

    result = await service.append_event(session=session, event=event)

    assert result is event
    assert session.last_update_time == 2.0
    assert len(session.events) == 1
    mock_client.collections.__getitem__.return_value.documents.create.assert_called_once()

  @pytest.mark.asyncio
  async def test_append_event_skips_partial_events(self, service, ts_mock):
    _, mock_client, _ = ts_mock
    session = Session(
        app_name='app',
        user_id='u',
        id='s',
        state={},
        events=[],
        last_update_time=1.0,
    )
    event = Event(author='user', timestamp=2.0, partial=True)

    result = await service.append_event(session=session, event=event)

    assert result is event
    mock_client.collections.__getitem__.return_value.documents.create.assert_not_called()

  @pytest.mark.asyncio
  async def test_append_event_raises_on_stale_session(self, service):
    session_doc = _make_session_doc('app', 'u', 's', update_time=5.0)
    service._get_doc_sync = MagicMock(return_value=session_doc)

    session = Session(
        app_name='app',
        user_id='u',
        id='s',
        state={},
        events=[],
        last_update_time=1.0,
    )
    event = Event(author='user', timestamp=6.0)

    with pytest.raises(ValueError, match='modified in storage'):
      await service.append_event(session=session, event=event)

  @pytest.mark.asyncio
  async def test_append_event_raises_if_session_missing(self, service):
    service._get_doc_sync = MagicMock(return_value=None)
    session = Session(
        app_name='app',
        user_id='u',
        id='ghost',
        state={},
        events=[],
        last_update_time=1.0,
    )
    event = Event(author='user', timestamp=2.0)

    with pytest.raises(ValueError, match='not found'):
      await service.append_event(session=session, event=event)

  @pytest.mark.asyncio
  async def test_append_event_updates_all_state_scopes(self, service, ts_mock):
    _, mock_client, _ = ts_mock
    session_doc = _make_session_doc('app', 'u', 's', update_time=1.0)
    service._get_doc_sync = MagicMock(return_value=session_doc)
    mock_client.collections.__getitem__.return_value.documents.__getitem__.return_value.update = (
        MagicMock()
    )
    mock_client.collections.__getitem__.return_value.documents.create = (
        MagicMock()
    )

    upserted: list[dict] = []
    mock_client.collections.__getitem__.return_value.documents.upsert = (
        MagicMock(side_effect=lambda doc: upserted.append(doc))
    )

    session = Session(
        app_name='app',
        user_id='u',
        id='s',
        state={},
        events=[],
        last_update_time=1.0,
    )
    event = Event(
        author='user',
        timestamp=2.0,
        actions=EventActions(
            state_delta={
                'app:shared': 'app_val',
                'user:pref': 'user_val',
                'local': 'session_val',
                'temp:scratch': 'ignored',
            }
        ),
    )

    await service.append_event(session=session, event=event)

    assert len(upserted) == 2
    app_u = next(d for d in upserted if 'user_id' not in d)
    assert json.loads(app_u['state']) == {'shared': 'app_val'}
    user_u = next(d for d in upserted if 'user_id' in d)
    assert json.loads(user_u['state']) == {'pref': 'user_val'}

    update_call = (
        mock_client.collections.__getitem__.return_value.documents.__getitem__.return_value.update.call_args
    )
    updated_fields = update_call[0][0]
    if 'state' in updated_fields:
      persisted = json.loads(updated_fields['state'])
      assert 'temp:scratch' not in persisted

  def test_missing_typesense_raises_import_error(self):
    with patch.dict(
        'sys.modules', {'typesense': None, 'typesense.exceptions': None}
    ):
      with pytest.raises(ImportError, match='typesense'):
        TypesenseSessionService(api_key='x')
