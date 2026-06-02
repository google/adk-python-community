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

Requires a running Valkey instance. Set VALKEY_HOST and VALKEY_PORT
environment variables if not using defaults (localhost:6379).

Run with:
    pytest tests/integration/test_valkey_memory_service_integration.py -v
"""

from __future__ import annotations

import os
import uuid

from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.adk_community.memory.valkey_memory_service import (
    ValkeyMemoryService,
    ValkeyMemoryServiceConfig,
)
from google.genai import types
import pytest

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
  from glide import GlideClient, GlideClientConfiguration, NodeAddress

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
  test_prefix = f"test:memory:{uuid.uuid4().hex[:8]}"
  config = ValkeyMemoryServiceConfig(
      key_prefix=test_prefix,
      search_top_k=10,
  )
  service = ValkeyMemoryService(client=valkey_client, config=config)
  yield service

  # Cleanup: delete test keys
  list_key = f"{test_prefix}:*"
  try:
    # Use KEYS to find all test keys and delete them
    keys = await valkey_client.custom_command(["KEYS", list_key])
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
              id='event-1',
              invocation_id='inv-1',
              author='user',
              timestamp=12345,
              content=types.Content(
                  parts=[types.Part(text='I enjoy learning Python.')]
              ),
          ),
          Event(
              id='event-2',
              invocation_id='inv-2',
              author='model',
              timestamp=12346,
              content=types.Content(
                  parts=[
                      types.Part(
                          text='Python is versatile and beginner-friendly.'
                      )
                  ]
              ),
          ),
          Event(
              id='event-3',
              invocation_id='inv-3',
              author='user',
              timestamp=12347,
              content=types.Content(
                  parts=[
                      types.Part(
                          text='What about Rust for systems programming?'
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

    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="JavaScript framework",
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
                id='event-other',
                invocation_id='inv-other',
                author='user',
                timestamp=12345,
                content=types.Content(
                    parts=[types.Part(text='I prefer Java over everything.')]
                ),
            ),
        ],
    )

    await memory_service.add_session_to_memory(session1)
    await memory_service.add_session_to_memory(session2)

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
                id='event-extra',
                invocation_id='inv-extra',
                author='user',
                timestamp=22345,
                content=types.Content(
                    parts=[
                        types.Part(text='Python web frameworks are useful.')
                    ]
                ),
            ),
        ],
    )

    await memory_service.add_session_to_memory(session1)
    await memory_service.add_session_to_memory(session2)

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
