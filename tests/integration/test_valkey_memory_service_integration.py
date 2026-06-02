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

"""Integration tests for ValkeyMemoryService.

Requires a running Valkey instance with the Search module loaded.
Set VALKEY_HOST and VALKEY_PORT environment variables if not using
defaults (localhost:6379).

The valkey-bundle image (valkey/valkey-bundle) includes the Search
module. Run with:

    podman run -d --name valkey-test -p 6379:6379 valkey/valkey-bundle:9.1
    pytest tests/integration/test_valkey_memory_service_integration.py -v
"""

from __future__ import annotations

import asyncio
import os
import uuid

from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types
import pytest

from google.adk_community.memory.valkey_memory_service import ValkeyMemoryService
from google.adk_community.memory.valkey_memory_service import ValkeyMemoryServiceConfig

VALKEY_HOST = os.environ.get("VALKEY_HOST", "localhost")
VALKEY_PORT = int(os.environ.get("VALKEY_PORT", "6379"))


def _requires_valkey():
  """Check if Valkey is available, skip if not."""
  try:
    import glide  # noqa: F401
  except ImportError:
    pytest.skip("valkey-glide not installed")


@pytest.fixture
async def valkey_client():
  """Create a connected valkey-glide client."""
  _requires_valkey()
  from glide import GlideClient
  from glide import GlideClientConfiguration
  from glide import NodeAddress

  config = GlideClientConfiguration(
      addresses=[NodeAddress(host=VALKEY_HOST, port=VALKEY_PORT)],
      client_name="adk_memory_integration_test_client",
  )
  client = await GlideClient.create(config)
  yield client
  await client.close()


@pytest.fixture
async def memory_service(valkey_client):
  """Create ValkeyMemoryService with a unique prefix for test isolation."""
  from glide import ft

  test_prefix = f"test:memory:{uuid.uuid4().hex[:8]}"
  index_name = f"test_idx_{uuid.uuid4().hex[:8]}"
  config = ValkeyMemoryServiceConfig(
      key_prefix=test_prefix,
      index_name=index_name,
      search_top_k=10,
  )
  service = ValkeyMemoryService(client=valkey_client, config=config)
  await service.create_index()

  # Small delay for index to be ready
  await asyncio.sleep(0.1)

  yield service

  # Cleanup: drop the index and delete test keys
  try:
    await ft.dropindex(valkey_client, index_name)
  except Exception:
    pass
  try:
    keys = await valkey_client.custom_command(["KEYS", f"{test_prefix}:*"])
    if keys:
      for key in keys:
        key_str = key.decode() if isinstance(key, bytes) else key
        await valkey_client.custom_command(["DEL", key_str])
  except Exception:
    pass


def _make_session(app_name: str, user_id: str) -> Session:
  """Create a test session with events."""
  return Session(
      app_name=app_name,
      user_id=user_id,
      id=f"session-{uuid.uuid4().hex[:8]}",
      last_update_time=1000,
      events=[
          Event(
              id="event-1",
              invocation_id="inv-1",
              author="user",
              timestamp=12345,
              content=types.Content(
                  parts=[types.Part(text="I enjoy learning Python.")]
              ),
          ),
          Event(
              id="event-2",
              invocation_id="inv-2",
              author="model",
              timestamp=12346,
              content=types.Content(
                  parts=[
                      types.Part(
                          text="Python is versatile and beginner-friendly."
                      )
                  ]
              ),
          ),
          Event(
              id="event-3",
              invocation_id="inv-3",
              author="user",
              timestamp=12347,
              content=types.Content(
                  parts=[
                      types.Part(
                          text="What about Rust for systems programming?"
                      )
                  ]
              ),
          ),
      ],
  )


@pytest.mark.asyncio
class TestValkeyMemoryServiceIntegration:
  """Integration tests for ValkeyMemoryService with a real Valkey instance."""

  async def test_add_and_search_memories(self, memory_service):
    """Test adding a session and searching for memories."""
    session = _make_session("test-app", "user-1")

    await memory_service.add_session_to_memory(session)
    await asyncio.sleep(0.5)  # Wait for indexing

    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="Python",
    )

    assert len(result.memories) >= 1
    texts = [m.content.parts[0].text for m in result.memories]
    assert any("Python" in t for t in texts)

  async def test_search_returns_empty_for_no_match(self, memory_service):
    """Test that search returns empty when no memories match."""
    session = _make_session("test-app", "user-1")
    await memory_service.add_session_to_memory(session)
    await asyncio.sleep(0.5)

    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="JavaScript framework Angular",
    )

    assert len(result.memories) == 0

  async def test_user_isolation(self, memory_service):
    """Test that memories are isolated between users."""
    session1 = _make_session("test-app", "user-1")
    session2 = Session(
        app_name="test-app",
        user_id="user-2",
        id="session-other",
        last_update_time=1000,
        events=[
            Event(
                id="event-other",
                invocation_id="inv-other",
                author="user",
                timestamp=12345,
                content=types.Content(
                    parts=[types.Part(text="I prefer Java over everything.")]
                ),
            ),
        ],
    )

    await memory_service.add_session_to_memory(session1)
    await memory_service.add_session_to_memory(session2)
    await asyncio.sleep(0.5)

    # user-1 should not see user-2's memories
    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="Java",
    )
    assert len(result.memories) == 0

    # user-2 should see their own Java memory
    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-2",
        query="Java",
    )
    assert len(result.memories) == 1

  async def test_app_isolation(self, memory_service):
    """Test that memories are isolated between applications."""
    session1 = Session(
        app_name="app-one",
        user_id="user-1",
        id="session-app1",
        last_update_time=1000,
        events=[
            Event(
                id="event-app1",
                invocation_id="inv-app1",
                author="user",
                timestamp=12345,
                content=types.Content(
                    parts=[types.Part(text="Kubernetes orchestration tips.")]
                ),
            ),
        ],
    )
    session2 = Session(
        app_name="app-two",
        user_id="user-1",
        id="session-app2",
        last_update_time=1000,
        events=[
            Event(
                id="event-app2",
                invocation_id="inv-app2",
                author="user",
                timestamp=12345,
                content=types.Content(
                    parts=[types.Part(text="Docker container best practices.")]
                ),
            ),
        ],
    )

    await memory_service.add_session_to_memory(session1)
    await memory_service.add_session_to_memory(session2)
    await asyncio.sleep(0.5)

    # app-one should only see its own memories
    result = await memory_service.search_memory(
        app_name="app-one",
        user_id="user-1",
        query="Kubernetes",
    )
    assert len(result.memories) == 1

    # app-two should not see app-one's memories
    result = await memory_service.search_memory(
        app_name="app-two",
        user_id="user-1",
        query="Kubernetes",
    )
    assert len(result.memories) == 0

  async def test_multiple_sessions_accumulate(self, memory_service):
    """Test that multiple sessions accumulate memories."""
    session1 = _make_session("test-app", "user-1")
    session2 = Session(
        app_name="test-app",
        user_id="user-1",
        id="session-2",
        last_update_time=2000,
        events=[
            Event(
                id="event-extra",
                invocation_id="inv-extra",
                author="user",
                timestamp=22345,
                content=types.Content(
                    parts=[types.Part(text="Python web frameworks are useful.")]
                ),
            ),
        ],
    )

    await memory_service.add_session_to_memory(session1)
    await memory_service.add_session_to_memory(session2)
    await asyncio.sleep(0.5)

    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="Python",
    )

    # Should find memories from both sessions
    assert len(result.memories) >= 3

  async def test_search_empty_store(self, memory_service):
    """Test searching when no memories have been added."""
    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="anything",
    )

    assert len(result.memories) == 0

  async def test_search_case_insensitive(self, memory_service):
    """Test that full-text search is case-insensitive."""
    session = Session(
        app_name="test-app",
        user_id="user-1",
        id="session-case",
        last_update_time=1000,
        events=[
            Event(
                id="event-case",
                invocation_id="inv-case",
                author="user",
                timestamp=12345,
                content=types.Content(
                    parts=[
                        types.Part(
                            text="VALKEY is a high performance datastore."
                        )
                    ]
                ),
            ),
        ],
    )

    await memory_service.add_session_to_memory(session)
    await asyncio.sleep(0.5)

    # Search with lowercase should find uppercase content
    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="valkey",
    )
    assert len(result.memories) == 1

  async def test_search_top_k_limit(self, valkey_client):
    """Test that search_top_k limits the number of results."""
    from glide import ft

    test_prefix = f"test:topk:{uuid.uuid4().hex[:8]}"
    index_name = f"test_topk_idx_{uuid.uuid4().hex[:8]}"
    config = ValkeyMemoryServiceConfig(
        key_prefix=test_prefix,
        index_name=index_name,
        search_top_k=3,
    )
    service = ValkeyMemoryService(client=valkey_client, config=config)
    await service.create_index()
    await asyncio.sleep(0.1)

    # Add more events than top_k
    events = [
        Event(
            id=f"event-{i}",
            invocation_id=f"inv-{i}",
            author="user",
            timestamp=12345 + i,
            content=types.Content(
                parts=[types.Part(text=f"Python tip number {i} is great.")]
            ),
        )
        for i in range(6)
    ]
    session = Session(
        app_name="test-app",
        user_id="user-1",
        id="session-topk",
        last_update_time=1000,
        events=events,
    )

    await service.add_session_to_memory(session)
    await asyncio.sleep(0.5)

    result = await service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="Python",
    )

    # Should return at most 3 (search_top_k)
    assert len(result.memories) <= 3
    assert len(result.memories) >= 1

    # Cleanup
    try:
      await ft.dropindex(valkey_client, index_name)
    except Exception:
      pass
    try:
      keys = await valkey_client.custom_command(["KEYS", f"{test_prefix}:*"])
      if keys:
        for key in keys:
          key_str = key.decode() if isinstance(key, bytes) else key
          await valkey_client.custom_command(["DEL", key_str])
    except Exception:
      pass

  async def test_events_without_text_are_filtered(self, memory_service):
    """Test that function_call and empty events are not stored."""
    session = Session(
        app_name="test-app",
        user_id="user-1",
        id="session-filter",
        last_update_time=1000,
        events=[
            # Function call event - should be filtered
            Event(
                id="event-func",
                invocation_id="inv-func",
                author="agent",
                timestamp=12345,
                content=types.Content(
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(name="search_tool")
                        )
                    ]
                ),
            ),
            # Empty event - should be filtered
            Event(
                id="event-empty",
                invocation_id="inv-empty",
                author="user",
                timestamp=12346,
            ),
            # Valid text event - should be stored
            Event(
                id="event-text",
                invocation_id="inv-text",
                author="user",
                timestamp=12347,
                content=types.Content(
                    parts=[types.Part(text="This is valid content.")]
                ),
            ),
        ],
    )

    await memory_service.add_session_to_memory(session)
    await asyncio.sleep(0.5)

    # Only the text event should be searchable
    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="valid content",
    )
    assert len(result.memories) == 1

    # Function call content should not appear
    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="search_tool",
    )
    assert len(result.memories) == 0

  async def test_memory_entry_metadata(self, memory_service):
    """Test that returned MemoryEntry has correct metadata."""
    session = Session(
        app_name="test-app",
        user_id="user-1",
        id="session-meta",
        last_update_time=1000,
        events=[
            Event(
                id="event-meta",
                invocation_id="inv-meta",
                author="user",
                timestamp=99999,
                content=types.Content(
                    parts=[types.Part(text="Metadata verification test.")]
                ),
            ),
        ],
    )

    await memory_service.add_session_to_memory(session)
    await asyncio.sleep(0.5)

    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="Metadata verification",
    )

    assert len(result.memories) == 1
    entry = result.memories[0]
    assert entry.content.parts[0].text == "Metadata verification test."
    assert entry.author == "user"
    assert entry.timestamp == "99999"

  async def test_create_index_idempotent(self, valkey_client):
    """Test that calling create_index multiple times is safe."""
    from glide import ft

    index_name = f"test_idem_idx_{uuid.uuid4().hex[:8]}"
    test_prefix = f"test:idem:{uuid.uuid4().hex[:8]}"
    config = ValkeyMemoryServiceConfig(
        key_prefix=test_prefix,
        index_name=index_name,
    )
    service = ValkeyMemoryService(client=valkey_client, config=config)

    # First call should succeed
    await service.create_index()
    assert service._index_created is True

    # Second call should not raise
    await service.create_index()
    assert service._index_created is True

    # Cleanup
    try:
      await ft.dropindex(valkey_client, index_name)
    except Exception:
      pass
