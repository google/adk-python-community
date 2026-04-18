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

from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types
import pytest

MOCK_APP_NAME = "test-app"
MOCK_USER_ID = "user-123"
MOCK_SESSION_ID = "session-456"


def _make_session(events=None):
  return Session(
      app_name=MOCK_APP_NAME,
      user_id=MOCK_USER_ID,
      id=MOCK_SESSION_ID,
      last_update_time=1000,
      events=events or [],
  )


def _text_event(event_id, author, text):
  return Event(
      id=event_id,
      invocation_id=f"inv-{event_id}",
      author=author,
      timestamp=12345,
      content=types.Content(parts=[types.Part(text=text)]),
  )


def _tool_call_event(event_id, name, args=None):
  return Event(
      id=event_id,
      invocation_id=f"inv-{event_id}",
      author="model",
      timestamp=12346,
      content=types.Content(
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name=name, args=args or {}
                  )
              )
          ]
      ),
  )


def _tool_response_event(event_id, name, response=None):
  return Event(
      id=event_id,
      invocation_id=f"inv-{event_id}",
      author="tool",
      timestamp=12347,
      content=types.Content(
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name=name, response=response or {"status": "ok"}
                  )
              )
          ]
      ),
  )


MOCK_SESSION_TEXT_ONLY = _make_session([
    _text_event("e1", "user", "I prefer dark mode."),
    _text_event("e2", "model", "Noted, dark mode preference saved."),
])

MOCK_SESSION_WITH_TOOLS = _make_session([
    _text_event("e1", "user", "Run depreciation for Q4"),
    _tool_call_event("e2", "run_depreciation", {"quarter": "Q4"}),
    _tool_response_event("e3", "run_depreciation", {"status": "success"}),
    _text_event("e4", "model", "Depreciation complete for Q4."),
])

MOCK_SESSION_EMPTY = _make_session([])

MOCK_SESSION_SINGLE_MSG = _make_session([
    _text_event("e1", "user", "Hello"),
])


class MockStoreItem:
  """Mock item returned by store.asearch()."""

  def __init__(self, content, updated_at=None):
    self.value = {"content": content}
    self.updated_at = updated_at or datetime(2025, 1, 1)


@pytest.fixture
def mock_store():
  store = MagicMock()
  store.asearch = AsyncMock(return_value=[])
  return store


@pytest.fixture
def mock_manager():
  mgr = MagicMock()
  mgr.ainvoke = AsyncMock(return_value=None)
  return mgr


@pytest.fixture
def service(mock_store):
  from google.adk_community.memory.langmem_memory_service import (
      LangMemMemoryService,
  )

  return LangMemMemoryService(
      store=mock_store,
      extraction_model="openai:gpt-4o-mini",
  )


class TestLangMemMemoryServiceInit:

  def test_default_configuration(self, mock_store):
    from google.adk_community.memory.langmem_memory_service import (
        LangMemMemoryService,
    )

    svc = LangMemMemoryService(
        store=mock_store, extraction_model="openai:gpt-4o-mini"
    )
    assert svc._extraction_model == "openai:gpt-4o-mini"
    assert svc._namespace_prefix == ("memories",)
    assert svc._max_semantic == 10
    assert svc._max_episodic == 5
    assert svc._enable_episodic is True

  def test_custom_configuration(self, mock_store):
    from google.adk_community.memory.langmem_memory_service import (
        LangMemMemoryService,
    )

    svc = LangMemMemoryService(
        store=mock_store,
        extraction_model="google:gemini-2.0-flash",
        namespace_prefix=("myapp", "data"),
        semantic_instructions="Custom semantic",
        episodic_instructions="Custom episodic",
        max_semantic_results=20,
        max_episodic_results=8,
        enable_episodic=False,
    )
    assert svc._extraction_model == "google:gemini-2.0-flash"
    assert svc._namespace_prefix == ("myapp", "data")
    assert svc._semantic_instructions == "Custom semantic"
    assert svc._episodic_instructions == "Custom episodic"
    assert svc._max_semantic == 20
    assert svc._max_episodic == 8
    assert svc._enable_episodic is False

  def test_import_error_when_langmem_missing(self, mock_store):
    from google.adk_community.memory.langmem_memory_service import (
        LangMemMemoryService,
    )

    svc = LangMemMemoryService(
        store=mock_store, extraction_model="openai:gpt-4o-mini"
    )
    with patch.dict("sys.modules", {"langmem": None}):
      with pytest.raises(ImportError, match="langmem is required"):
        svc._make_manager(("ns",), "instructions")


class TestAddSessionToMemory:

  @pytest.mark.asyncio
  async def test_extracts_semantic_and_episodic(
      self, service, mock_manager
  ):
    with patch.object(
        service, "_make_manager", return_value=mock_manager
    ) as make_mock:
      await service.add_session_to_memory(MOCK_SESSION_WITH_TOOLS)

      assert make_mock.call_count == 2
      semantic_call = make_mock.call_args_list[0]
      assert semantic_call[0][0] == (
          "memories",
          MOCK_USER_ID,
          "semantic",
      )
      episodic_call = make_mock.call_args_list[1]
      assert episodic_call[0][0] == (
          "memories",
          MOCK_USER_ID,
          "episodic",
      )
      assert mock_manager.ainvoke.call_count == 2

  @pytest.mark.asyncio
  async def test_text_only_session_skips_episodic(
      self, service, mock_manager
  ):
    with patch.object(
        service, "_make_manager", return_value=mock_manager
    ) as make_mock:
      await service.add_session_to_memory(MOCK_SESSION_TEXT_ONLY)

      assert make_mock.call_count == 1
      semantic_call = make_mock.call_args_list[0]
      assert "semantic" in semantic_call[0][0]

  @pytest.mark.asyncio
  async def test_empty_session_is_noop(self, service, mock_manager):
    with patch.object(
        service, "_make_manager", return_value=mock_manager
    ) as make_mock:
      await service.add_session_to_memory(MOCK_SESSION_EMPTY)
      assert make_mock.call_count == 0

  @pytest.mark.asyncio
  async def test_single_message_is_noop(self, service, mock_manager):
    with patch.object(
        service, "_make_manager", return_value=mock_manager
    ) as make_mock:
      await service.add_session_to_memory(MOCK_SESSION_SINGLE_MSG)
      assert make_mock.call_count == 0

  @pytest.mark.asyncio
  async def test_no_user_id_is_noop(self, service, mock_manager):
    session = Session(
        app_name=MOCK_APP_NAME,
        user_id="",
        id=MOCK_SESSION_ID,
        last_update_time=1000,
        events=[
            _text_event("e1", "user", "Hello"),
            _text_event("e2", "model", "Hi"),
        ],
    )
    with patch.object(
        service, "_make_manager", return_value=mock_manager
    ) as make_mock:
      await service.add_session_to_memory(session)
      assert make_mock.call_count == 0

  @pytest.mark.asyncio
  async def test_episodic_disabled(self, mock_store, mock_manager):
    from google.adk_community.memory.langmem_memory_service import (
        LangMemMemoryService,
    )

    svc = LangMemMemoryService(
        store=mock_store,
        extraction_model="openai:gpt-4o-mini",
        enable_episodic=False,
    )
    with patch.object(
        svc, "_make_manager", return_value=mock_manager
    ) as make_mock:
      await svc.add_session_to_memory(MOCK_SESSION_WITH_TOOLS)

      assert make_mock.call_count == 1
      assert "semantic" in make_mock.call_args_list[0][0][0]

  @pytest.mark.asyncio
  async def test_extraction_error_is_logged_not_raised(
      self, service
  ):
    failing_manager = MagicMock()
    failing_manager.ainvoke = AsyncMock(
        side_effect=RuntimeError("LLM failure")
    )
    with patch.object(
        service, "_make_manager", return_value=failing_manager
    ):
      await service.add_session_to_memory(MOCK_SESSION_TEXT_ONLY)

  @pytest.mark.asyncio
  async def test_custom_namespace_prefix(
      self, mock_store, mock_manager
  ):
    from google.adk_community.memory.langmem_memory_service import (
        LangMemMemoryService,
    )

    svc = LangMemMemoryService(
        store=mock_store,
        extraction_model="openai:gpt-4o-mini",
        namespace_prefix=("org", "team"),
    )
    with patch.object(
        svc, "_make_manager", return_value=mock_manager
    ) as make_mock:
      await svc.add_session_to_memory(MOCK_SESSION_TEXT_ONLY)

      ns = make_mock.call_args_list[0][0][0]
      assert ns == ("org", "team", MOCK_USER_ID, "semantic")


class TestSearchMemory:

  @pytest.mark.asyncio
  async def test_merges_semantic_and_episodic(
      self, service, mock_store
  ):
    semantic_items = [
        MockStoreItem("User likes dark mode", datetime(2025, 3, 1)),
        MockStoreItem("User is in California", datetime(2025, 2, 1)),
    ]
    episodic_items = [
        MockStoreItem("Ran depreciation for Q4", datetime(2025, 4, 1)),
    ]

    async def fake_search(namespace, query, limit):
      if "semantic" in namespace:
        return semantic_items
      return episodic_items

    mock_store.asearch = AsyncMock(side_effect=fake_search)

    result = await service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query="dark mode",
    )

    assert len(result.memories) == 3
    texts = [m.content.parts[0].text for m in result.memories]
    assert "User likes dark mode" in texts
    assert "User is in California" in texts
    assert "Ran depreciation for Q4" in texts

  @pytest.mark.asyncio
  async def test_search_handles_store_error(
      self, service, mock_store
  ):
    mock_store.asearch = AsyncMock(
        side_effect=RuntimeError("Connection refused")
    )

    result = await service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query="anything",
    )

    assert len(result.memories) == 0

  @pytest.mark.asyncio
  async def test_search_handles_partial_failure(
      self, service, mock_store
  ):
    call_count = 0

    async def partial_fail(namespace, query, limit):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return [MockStoreItem("Good result")]
      raise RuntimeError("Episodic store down")

    mock_store.asearch = AsyncMock(side_effect=partial_fail)

    result = await service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query="test",
    )

    assert len(result.memories) == 1
    assert result.memories[0].content.parts[0].text == "Good result"

  @pytest.mark.asyncio
  async def test_search_respects_max_results(
      self, mock_store
  ):
    from google.adk_community.memory.langmem_memory_service import (
        LangMemMemoryService,
    )

    svc = LangMemMemoryService(
        store=mock_store,
        extraction_model="openai:gpt-4o-mini",
        max_semantic_results=2,
        max_episodic_results=1,
        enable_episodic=False,
    )

    items = [
        MockStoreItem(f"fact-{i}", datetime(2025, 1, i + 1))
        for i in range(5)
    ]
    mock_store.asearch = AsyncMock(return_value=items)

    result = await svc.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query="test",
    )

    assert len(result.memories) == 2

  @pytest.mark.asyncio
  async def test_search_filters_empty_values(
      self, mock_store
  ):
    from google.adk_community.memory.langmem_memory_service import (
        LangMemMemoryService,
    )

    svc = LangMemMemoryService(
        store=mock_store,
        extraction_model="openai:gpt-4o-mini",
        enable_episodic=False,
    )
    items = [
        MockStoreItem("real content"),
        MockStoreItem(""),
        MockStoreItem("   "),
    ]
    mock_store.asearch = AsyncMock(return_value=items)

    result = await svc.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query="test",
    )

    assert len(result.memories) == 1
    assert (
        result.memories[0].content.parts[0].text == "real content"
    )

  @pytest.mark.asyncio
  async def test_search_namespace_scoping(
      self, service, mock_store
  ):
    mock_store.asearch = AsyncMock(return_value=[])

    await service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query="test",
    )

    calls = mock_store.asearch.call_args_list
    assert len(calls) == 2
    assert calls[0][0][0] == (
        "memories",
        MOCK_USER_ID,
        "semantic",
    )
    assert calls[1][0][0] == (
        "memories",
        MOCK_USER_ID,
        "episodic",
    )


class TestExtractMessages:

  def test_text_messages_extracted(self, service):
    text_msgs, episode_msgs = service._extract_messages(
        MOCK_SESSION_TEXT_ONLY
    )
    assert len(text_msgs) == 2
    assert text_msgs[0] == {
        "role": "user",
        "content": "I prefer dark mode.",
    }
    assert text_msgs[1] == {
        "role": "assistant",
        "content": "Noted, dark mode preference saved.",
    }
    assert episode_msgs == []

  def test_tool_calls_included_in_episodic(self, service):
    text_msgs, episode_msgs = service._extract_messages(
        MOCK_SESSION_WITH_TOOLS
    )
    assert len(text_msgs) == 2
    assert len(episode_msgs) == 1
    episode_text = episode_msgs[0]["content"]
    assert "Tool call sequence:" in episode_text
    assert "run_depreciation" in episode_text
    assert "Tool result:" in episode_text

  def test_empty_session_returns_empty(self, service):
    text_msgs, episode_msgs = service._extract_messages(
        MOCK_SESSION_EMPTY
    )
    assert text_msgs == []
    assert episode_msgs == []

  def test_thought_parts_excluded(self, service):
    session = _make_session([
        Event(
            id="e1",
            invocation_id="inv-e1",
            author="model",
            timestamp=12345,
            content=types.Content(
                parts=[
                    types.Part(text="thinking...", thought=True),
                    types.Part(text="Actual response"),
                ]
            ),
        ),
    ])
    text_msgs, _ = service._extract_messages(session)
    assert len(text_msgs) == 1
    assert text_msgs[0]["content"] == "Actual response"
