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

from datetime import datetime
from datetime import timezone
from decimal import Decimal
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

pytest.importorskip("pymongo")

from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions import State
from google.adk.sessions.base_session_service import GetSessionConfig
import pytest_asyncio

from google.adk_community.sessions.mongo_session_service import MongoKeys
from google.adk_community.sessions.mongo_session_service import MongoSessionService


class TestMongoSessionService:
  """Tests for MongoSessionService mirroring Redis coverage."""

  @pytest_asyncio.fixture
  async def mongo_service(self):
    """Create a Mongo session service with mocked collections."""
    sessions_collection = AsyncMock()
    sessions_collection.create_index = AsyncMock()
    sessions_collection.insert_one = AsyncMock()
    sessions_collection.find_one = AsyncMock(return_value=None)
    sessions_collection.find = MagicMock()
    sessions_collection.delete_one = AsyncMock()
    sessions_collection.update_one = AsyncMock(
        return_value=MagicMock(matched_count=1)
    )

    kv_collection = AsyncMock()
    kv_collection.find_one = AsyncMock(return_value=None)
    kv_collection.find = MagicMock()
    kv_collection.update_one = AsyncMock()

    db = MagicMock()

    def _get_collection(name: str):
      if name == "sessions":
        return sessions_collection
      if name == "session_state":
        return kv_collection
      raise KeyError(name)

    db.__getitem__.side_effect = _get_collection

    client = MagicMock()
    client.__getitem__.return_value = db

    service = MongoSessionService(database_name="test_db", client=client)
    return service, sessions_collection, kv_collection

  @pytest.mark.asyncio
  async def test_get_empty_session(self, mongo_service):
    """get_session should return None for missing sessions."""
    service, sessions_collection, _ = mongo_service
    sessions_collection.find_one.return_value = None

    session = await service.get_session(
        app_name="test_app", user_id="test_user", session_id="missing"
    )

    assert session is None
    sessions_collection.find_one.assert_awaited_once_with({
        "_id": MongoKeys.session("test_app", "test_user", "missing"),
    })
    assert service._indexes_built is True

  @pytest.mark.asyncio
  async def test_create_and_get_session(self, mongo_service):
    """create_session persists and get_session retrieves a session."""
    service, sessions_collection, kv_collection = mongo_service
    kv_collection.find_one.return_value = None

    session = await service.create_session(
        app_name="test_app",
        user_id="test_user",
        session_id="session-1",
        state={"key": "value"},
    )

    inserted_doc = sessions_collection.insert_one.await_args.args[0]
    assert inserted_doc["_id"] == MongoKeys.session(
        "test_app", "test_user", "session-1"
    )
    assert inserted_doc["state"] == {"key": "value"}

    sessions_collection.find_one.return_value = dict(inserted_doc)

    retrieved = await service.get_session(
        app_name="test_app", user_id="test_user", session_id="session-1"
    )

    assert retrieved is not None
    assert retrieved.id == "session-1"
    assert retrieved.app_name == session.app_name
    assert retrieved.user_id == session.user_id
    assert retrieved.state["key"] == "value"

    # Indexes should only be created once even when called from multiple methods.
    assert sessions_collection.create_index.await_count == 2

  @pytest.mark.asyncio
  async def test_list_sessions_merges_state_and_strips_events(
      self, mongo_service
  ):
    """list_sessions returns sessions without events but with merged state."""
    service, sessions_collection, kv_collection = mongo_service
    app_name = "test_app"
    user_id = "user1"

    docs = [
        {
            "_id": MongoKeys.session(app_name, user_id, "s1"),
            "app_name": app_name,
            "user_id": user_id,
            "id": "s1",
            "state": {"session_key": "v1"},
        },
        {
            "_id": MongoKeys.session(app_name, user_id, "s2"),
            "app_name": app_name,
            "user_id": user_id,
            "id": "s2",
            "state": {"session_key": "v2"},
        },
    ]

    cursor = MagicMock()
    cursor.to_list = AsyncMock(return_value=docs)
    sessions_collection.find.return_value = cursor

    async def _kv_find_one(query):
      if query["_id"] == MongoKeys.app_state(app_name):
        return {"_id": query["_id"], "state": {"theme": "dark"}}
      if query["_id"] == MongoKeys.user_state(app_name, user_id):
        return {"_id": query["_id"], "state": {"pref": "value"}}
      return None

    kv_collection.find_one.side_effect = _kv_find_one

    response = await service.list_sessions(app_name=app_name, user_id=user_id)

    sessions_collection.find.assert_called_once_with(
        {"app_name": app_name, "user_id": user_id}, projection={"events": False}
    )

    assert len(response.sessions) == 2
    for doc, sess in zip(docs, response.sessions):
      assert sess.id == doc["id"]
      assert sess.events == []
      assert sess.state["session_key"] == doc["state"]["session_key"]
      assert sess.state[State.APP_PREFIX + "theme"] == "dark"
      assert sess.state[State.USER_PREFIX + "pref"] == "value"

  @pytest.mark.asyncio
  async def test_session_state_management_on_append_event(self, mongo_service):
    """append_event should persist state deltas and mutate in-memory session."""
    service, sessions_collection, kv_collection = mongo_service
    app_name = "test_app"
    user_id = "test_user"
    session_id = "session-123"
    kv_collection.find_one.return_value = None

    session = await service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={"initial_key": "initial_value"},
    )

    event = Event(
        invocation_id="invocation",
        author="user",
        timestamp=datetime.now().astimezone(timezone.utc).timestamp(),
        actions=EventActions(
            state_delta={
                f"{State.APP_PREFIX}key": "app_value",
                f"{State.USER_PREFIX}key1": "user_value",
                "temp:key": "temp_value",
                "initial_key": "updated_value",
            }
        ),
    )

    await service.append_event(session=session, event=event)

    assert session.state[State.APP_PREFIX + "key"] == "app_value"
    assert session.state[State.USER_PREFIX + "key1"] == "user_value"
    assert session.state["initial_key"] == "updated_value"
    assert session.state.get("temp:key") is None

    # App and user deltas are stored in the kv collection.
    assert kv_collection.update_one.await_count == 2
    kv_update_filters = [
        call.args[0] for call in kv_collection.update_one.await_args_list
    ]
    kv_updates = [
        call.args[1] for call in kv_collection.update_one.await_args_list
    ]
    assert {"_id": MongoKeys.app_state(app_name)} in kv_update_filters
    assert {"_id": MongoKeys.user_state(app_name, user_id)} in kv_update_filters
    assert any(
        update.get("$set", {}).get("state.key") == "app_value"
        for update in kv_updates
    )
    assert any(
        update.get("$set", {}).get("state.key1") == "user_value"
        for update in kv_updates
    )

    update_filter, update_doc = sessions_collection.update_one.await_args.args
    assert update_filter == {
        "_id": MongoKeys.session(app_name, user_id, session_id)
    }
    assert update_doc["$set"]["state.initial_key"] == "updated_value"
    # Temp state should not be persisted.
    assert "state.temp:key" not in update_doc.get("$set", {})
    assert "state.temp:key" not in update_doc.get("$unset", {})

  @pytest.mark.asyncio
  async def test_get_session_with_config(self, mongo_service):
    """get_session applies after_timestamp and num_recent_events filters."""
    service, sessions_collection, kv_collection = mongo_service
    kv_collection.find_one.return_value = None

    events = [
        Event(author="user", timestamp=float(i)).model_dump(
            mode="json", exclude_none=True
        )
        for i in range(1, 6)
    ]
    doc = {
        "_id": MongoKeys.session("app", "user", "session"),
        "app_name": "app",
        "user_id": "user",
        "id": "session",
        "events": events,
        "state": {},
    }

    sessions_collection.find_one.return_value = doc

    config = GetSessionConfig(num_recent_events=3)
    filtered = await service.get_session(
        app_name="app", user_id="user", session_id="session", config=config
    )
    assert [e.timestamp for e in filtered.events] == [3.0, 4.0, 5.0]

    config = GetSessionConfig(after_timestamp=3.0)
    filtered = await service.get_session(
        app_name="app", user_id="user", session_id="session", config=config
    )
    assert [e.timestamp for e in filtered.events] == [4.0, 5.0]

  @pytest.mark.asyncio
  async def test_delete_session(self, mongo_service):
    """delete_session removes session documents."""
    service, sessions_collection, _ = mongo_service

    await service.delete_session(
        app_name="test_app", user_id="user", session_id="session-1"
    )

    sessions_collection.delete_one.assert_awaited_once_with(
        {"_id": MongoKeys.session("test_app", "user", "session-1")}
    )

  @pytest.mark.asyncio
  async def test_non_bson_state_values_are_normalized(self, mongo_service):
    """Session, app, and user state use the shared JSON serializer."""
    service, sessions_collection, kv_collection = mongo_service
    kv_collection.find_one.return_value = None
    state = {
        "session_set": {"value"},
        "session_decimal": Decimal("1.25"),
        "session_datetime": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "session_bytes": b"bytes",
        f"{State.APP_PREFIX}app_set": {"value"},
        f"{State.USER_PREFIX}user_set": {"value"},
    }

    await service.create_session(
        app_name="app", user_id="user", session_id="session", state=state
    )

    inserted = sessions_collection.insert_one.await_args.args[0]
    assert inserted["state"] == {
        "session_set": ["value"],
        "session_decimal": 1.25,
        "session_datetime": "2025-01-01T00:00:00+00:00",
        "session_bytes": "Ynl0ZXM=",
    }
    updates = [
        call.args[1] for call in kv_collection.update_one.await_args_list
    ]
    set_values = {
        key: value
        for update in updates
        for key, value in update.get("$set", {}).items()
    }
    assert set_values["state.app_set"] == ["value"]
    assert set_values["state.user_set"] == ["value"]

  @pytest.mark.asyncio
  async def test_event_is_stored_as_nested_document(self, mongo_service):
    """Events remain addressable Mongo documents after normalization."""
    service, sessions_collection, kv_collection = mongo_service
    kv_collection.find_one.return_value = None
    session = await service.create_session(
        app_name="app", user_id="user", session_id="session"
    )

    event = Event(author="user", timestamp=1.0)
    await service.append_event(session, event)

    update = sessions_collection.update_one.await_args.args[1]
    stored_event = update["$push"]["events"]
    assert isinstance(stored_event, dict)
    assert stored_event["author"] == "user"
    assert not isinstance(stored_event, bytes)

  @pytest.mark.asyncio
  async def test_list_sessions_fetches_shared_state_once(self, mongo_service):
    """Listing sessions for one user does not issue per-session state reads."""
    service, sessions_collection, kv_collection = mongo_service
    app_name = "app"
    user_id = "user"
    docs = [
        {
            "_id": MongoKeys.session(app_name, user_id, f"s{i}"),
            "app_name": app_name,
            "user_id": user_id,
            "id": f"s{i}",
            "state": {},
        }
        for i in range(3)
    ]
    session_cursor = MagicMock()
    session_cursor.to_list = AsyncMock(return_value=docs)
    sessions_collection.find.return_value = session_cursor
    kv_collection.find_one.side_effect = [
        {"_id": MongoKeys.app_state(app_name), "state": {"a": 1}},
        {"_id": MongoKeys.user_state(app_name, user_id), "state": {"u": 2}},
    ]

    response = await service.list_sessions(app_name=app_name, user_id=user_id)

    assert len(response.sessions) == 3
    assert kv_collection.find_one.await_count == 2
    assert all(
        session.state[State.APP_PREFIX + "a"] == 1
        for session in response.sessions
    )
    assert all(
        session.state[State.USER_PREFIX + "u"] == 2
        for session in response.sessions
    )

  @pytest.mark.asyncio
  async def test_list_sessions_bulk_fetches_users(self, mongo_service):
    """Listing all users uses one state query and attributes state correctly."""
    service, sessions_collection, kv_collection = mongo_service
    app_name = "app"
    docs = [
        {
            "_id": MongoKeys.session(app_name, user_id, session_id),
            "app_name": app_name,
            "user_id": user_id,
            "id": session_id,
            "state": {},
        }
        for user_id, session_id in (("user1", "s1"), ("user2", "s2"))
    ]
    session_cursor = MagicMock()
    session_cursor.to_list = AsyncMock(return_value=docs)
    sessions_collection.find.return_value = session_cursor
    user_cursor = MagicMock()
    user_cursor.to_list = AsyncMock(
        return_value=[
            {
                "_id": MongoKeys.user_state(app_name, user_id),
                "state": {"value": user_id},
            }
            for user_id in ("user1", "user2")
        ]
    )
    kv_collection.find.return_value = user_cursor
    kv_collection.find_one.return_value = None

    response = await service.list_sessions(app_name=app_name)

    assert kv_collection.find.call_count == 1
    assert [
        session.state[State.USER_PREFIX + "value"]
        for session in response.sessions
    ] == [
        "user1",
        "user2",
    ]
