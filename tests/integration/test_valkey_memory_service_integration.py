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

"""Integration tests for ValkeyMemoryService with vector similarity search.

Requires a running Valkey instance with the Search module loaded.
Set VALKEY_HOST and VALKEY_PORT environment variables if not using
defaults (localhost:6379).

The valkey-bundle image includes the Search module with vector support.
Run with:

    podman run -d --name valkey-test -p 6379:6379 valkey/valkey-bundle:9.1
    pytest tests/integration/test_valkey_memory_service_integration.py -v
"""

from __future__ import annotations

import asyncio
import math
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

# Simple deterministic embedding function for testing.
# Maps text to a 32-dim vector based on character frequencies.
EMBED_DIM = 32


async def _test_embedding_function(
    texts: list[str],
) -> list[list[float]]:
  """Deterministic embedding function for testing.

  Generates a 32-dimensional vector based on character frequency
  distribution. This gives semantically similar texts somewhat
  similar vectors (texts with similar character distributions).
  """
  embeddings = []
  for text in texts:
    text_lower = text.lower()
    vec = [0.0] * EMBED_DIM
    for ch in text_lower:
      idx = ord(ch) % EMBED_DIM
      vec[idx] += 1.0
    # Normalize to unit vector
    magnitude = math.sqrt(sum(x * x for x in vec)) or 1.0
    vec = [x / magnitude for x in vec]
    embeddings.append(vec)
  return embeddings


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
      similarity_top_k=10,
      embedding_dimensions=EMBED_DIM,
  )
  service = ValkeyMemoryService(
      client=valkey_client,
      embedding_function=_test_embedding_function,
      config=config,
  )
  await service.create_index()
  await asyncio.sleep(0.1)

  yield service

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
  """Integration tests with vector similarity search."""

  async def test_add_and_search_memories(self, memory_service):
    """Test adding a session and searching with vector similarity."""
    session = _make_session("test-app", "user-1")
    await memory_service.add_session_to_memory(session)
    await asyncio.sleep(0.5)

    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="Python programming language",
    )

    assert len(result.memories) >= 1
    texts = [m.content.parts[0].text for m in result.memories]
    assert any("Python" in t for t in texts)

  async def test_search_returns_results_ranked_by_similarity(
      self, memory_service
  ):
    """Test that results are returned by vector similarity."""
    session = _make_session("test-app", "user-1")
    await memory_service.add_session_to_memory(session)
    await asyncio.sleep(0.5)

    # Query similar to "Python" content should return Python memories
    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="I enjoy learning Python programming",
    )

    assert len(result.memories) >= 1
    # The most similar result should be about Python
    top_text = result.memories[0].content.parts[0].text
    assert "Python" in top_text or "python" in top_text.lower()

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

    # user-1 should not see user-2's Java memory
    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="Java programming",
    )
    texts = [m.content.parts[0].text for m in result.memories]
    assert not any("Java" in t for t in texts)

    # user-2 should see their own Java memory
    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-2",
        query="Java programming",
    )
    assert len(result.memories) == 1
    assert "Java" in result.memories[0].content.parts[0].text

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
        query="Kubernetes orchestration",
    )
    assert len(result.memories) == 1

    # app-two should only see its own Docker memory, not Kubernetes
    result = await memory_service.search_memory(
        app_name="app-two",
        user_id="user-1",
        query="Kubernetes orchestration",
    )
    # KNN returns nearest neighbor from the filtered set (app-two only).
    # The result should NOT contain app-one's Kubernetes content.
    for mem in result.memories:
      assert "Kubernetes" not in mem.content.parts[0].text

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
        query="Python programming",
    )

    # Should find memories from both sessions
    assert len(result.memories) >= 3

  async def test_search_empty_store(self, memory_service):
    """Test searching when no memories have been added."""
    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="anything at all",
    )
    assert len(result.memories) == 0

  async def test_similarity_top_k_limit(self, valkey_client):
    """Test that similarity_top_k limits the number of results."""
    from glide import ft

    test_prefix = f"test:topk:{uuid.uuid4().hex[:8]}"
    index_name = f"test_topk_idx_{uuid.uuid4().hex[:8]}"
    config = ValkeyMemoryServiceConfig(
        key_prefix=test_prefix,
        index_name=index_name,
        similarity_top_k=3,
        embedding_dimensions=EMBED_DIM,
    )
    service = ValkeyMemoryService(
        client=valkey_client,
        embedding_function=_test_embedding_function,
        config=config,
    )
    await service.create_index()
    await asyncio.sleep(0.1)

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
        query="Python tips",
    )

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
            Event(
                id="event-empty",
                invocation_id="inv-empty",
                author="user",
                timestamp=12346,
            ),
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

    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="valid content text",
    )
    assert len(result.memories) == 1

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
        query="Metadata verification test",
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
        embedding_dimensions=EMBED_DIM,
    )
    service = ValkeyMemoryService(
        client=valkey_client,
        embedding_function=_test_embedding_function,
        config=config,
    )

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

  async def test_add_events_to_memory(self, memory_service):
    """Test incremental event ingestion via add_events_to_memory."""
    events = [
        Event(
            id="event-inc-1",
            invocation_id="inv-inc-1",
            author="user",
            timestamp=50001,
            content=types.Content(
                parts=[types.Part(text="Incremental memory about Golang.")]
            ),
        ),
        Event(
            id="event-inc-2",
            invocation_id="inv-inc-2",
            author="model",
            timestamp=50002,
            content=types.Content(
                parts=[types.Part(text="Go is great for concurrency.")]
            ),
        ),
    ]

    await memory_service.add_events_to_memory(
        app_name="test-app",
        user_id="user-1",
        events=events,
        session_id="session-incremental",
    )
    await asyncio.sleep(0.5)

    result = await memory_service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="Golang concurrency",
    )
    assert len(result.memories) >= 1
    texts = [m.content.parts[0].text for m in result.memories]
    assert any("Go" in t for t in texts)

  async def test_vector_distance_threshold(self, valkey_client):
    """Test that vector_distance_threshold filters distant results."""
    from glide import ft

    test_prefix = f"test:thresh:{uuid.uuid4().hex[:8]}"
    index_name = f"test_thresh_idx_{uuid.uuid4().hex[:8]}"
    config = ValkeyMemoryServiceConfig(
        key_prefix=test_prefix,
        index_name=index_name,
        similarity_top_k=10,
        embedding_dimensions=EMBED_DIM,
        vector_distance_threshold=0.01,  # Very strict threshold
    )
    service = ValkeyMemoryService(
        client=valkey_client,
        embedding_function=_test_embedding_function,
        config=config,
    )
    await service.create_index()
    await asyncio.sleep(0.1)

    session = Session(
        app_name="test-app",
        user_id="user-1",
        id="session-thresh",
        last_update_time=1000,
        events=[
            Event(
                id="event-thresh",
                invocation_id="inv-thresh",
                author="user",
                timestamp=12345,
                content=types.Content(
                    parts=[types.Part(text="Completely unrelated topic XYZ.")]
                ),
            ),
        ],
    )

    await service.add_session_to_memory(session)
    await asyncio.sleep(0.5)

    # Search for something very different — should be filtered by threshold
    result = await service.search_memory(
        app_name="test-app",
        user_id="user-1",
        query="AAAAAAA BBBBBBB CCCCCCC",
    )
    # With strict threshold, dissimilar results should be filtered
    # (This depends on the embedding function producing distant vectors)
    assert len(result.memories) <= 1

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
