# Evaluation: Custom `client.py` vs [goodmem-client](https://pypi.org/project/goodmem-client/) PyPI SDK

This doc compares keeping the in-repo `GoodmemClient` (httpx-based) vs adopting the official [goodmem-client](https://pypi.org/project/goodmem-client/) Python SDK (OpenAPI-generated, v1.5.10).

---

## Current custom client surface

| Method | Purpose |
|--------|--------|
| `create_space(space_name, embedder_id)` | Create space with default chunking |
| `list_spaces(name=None)` | List spaces, optional **name filter** (server-side `nameFilter`) |
| `list_embedders()` | List embedders (for lazy embedder resolution) |
| `insert_memory(space_id, content, content_type, metadata)` | Text memory |
| `insert_memory_binary(space_id, content_bytes, content_type, metadata)` | Binary/multipart memory |
| `retrieve_memories(query, space_ids, request_size)` | **POST** retrieve, returns **list** of parsed NDJSON chunks |
| `get_memory_by_id(memory_id)` | Single memory by ID |
| `get_memories_batch(memory_ids)` | **POST** `/v1/memories:batchGet` |

Plugin and tools use: **space-by-name** (list then pick), **sync** retrieve returning a **list**, **batch get**, and **binary insert**.

---

## Pros of switching to goodmem-client

- **Official / maintained**  
  Aligns with the [GoodMem server API](https://pypi.org/project/goodmem-client/) (e.g. server v1.0.224). Bug fixes and new endpoints (filter expressions, streaming, post-processors) show up in the SDK.

- **API coverage**  
  Covers spaces, memories, embedders, batch get, streaming retrieval, filters, etc.  
  [MemoriesApi](https://pypi.org/project/goodmem-client/) includes `batch_get_memory`, `get_memory`, `create_memory`; [SpacesApi](https://pypi.org/project/goodmem-client/) has `list_spaces`, `create_space`; [EmbeddersApi](https://pypi.org/project/goodmem-client/) has `list_embedders`.

- **Structured types**  
  OpenAPI-generated request/response models (e.g. `Space`, `Memory`, `BatchMemoryRetrievalRequest`) instead of raw dicts.

- **Less custom HTTP code**  
  No manual NDJSON parsing, URL encoding, or multipart building if the SDK exposes the same operations.

- **Future features**  
  Filter expressions, streaming (`MemoryStreamClient`), post-processors (e.g. Chat), OCR, etc. are documented and supported in the SDK.

- **Single dependency for Goodmem**  
  One `goodmem-client` dependency instead of maintaining our own HTTP client and keeping it in sync with the API.

---

## Cons / tradeoffs

- **Different call patterns**  
  Our code expects a small, sync API (e.g. `retrieve_memories` → list of chunks). The SDK emphasizes **streaming** (`MemoryStreamClient.retrieve_memory_stream`) and may expose **GET** vs **POST** retrieve differently. We’d need a thin wrapper that:
  - Calls the appropriate retrieve API (e.g. `retrieve_memory_advanced` or streaming).
  - Collects results into a **list** so plugin/tools don’t need to change.

- **“Get space ID from name” not a single call**  
  The SDK’s `SpacesApi.list_spaces()` returns a list of spaces; it may or may not support a `name_filter` (or equivalent) in the generated client. Either way we’d implement a small helper, e.g.:
  - `get_space_id_by_name(name) -> str | None`: call `list_spaces` (with filter if the API supports it), then find the space with `name == space_name` and return its ID. If the SDK doesn’t support server-side name filter, we filter in Python after listing.

- **Binary / multipart memory**  
  Our `insert_memory_binary` does multipart upload. We’d need to map that to the SDK’s memory creation (e.g. `MemoriesApi.create_memory` with the right payload/API for binary content). If the SDK only has a different shape, we keep a small adapter.

- **Response shape**  
  SDK returns typed models (e.g. Pydantic/dataclasses), not plain dicts. Plugin and tools use `space.get("spaceId")`, `response.get("memories", [])`, etc. We’d either:
  - Use SDK types and update call sites to use attributes, or
  - Add a thin “dict-like” adapter so existing code stays mostly unchanged.

- **Dependency and versioning**  
  We add `goodmem-client` and pin a version (e.g. `>=1.5.10, <2`). Upgrades may change method names or signatures (OpenAPI regen); we’d run tests and adjust wrappers.

- **Debug flag**  
  Our client has a `debug` flag and prints. The SDK uses `Configuration` and its own patterns; we’d either wrap the SDK and keep our debug prints in the wrapper or rely on the SDK’s logging.

---

## Gaps you’d implement on top of the SDK

1. **Space ID by name**  
   - `get_space_id_by_name(name: str) -> Optional[str]`:  
     Call `SpacesApi.list_spaces()` (with name filter parameter if present in the generated client).  
     If no name filter: list and return the first `space.space_id` where `space.name == name`.  
     Return `None` if not found.

2. **Sync “retrieve and return list”**  
   - If the SDK only offers streaming: consume `retrieve_memory_stream` (or equivalent), collect events into a list of chunk dicts, and expose e.g. `retrieve_memories_list(query, space_ids, request_size) -> List[Dict]` so the plugin’s `before_model_callback` and tools keep the same interface.

3. **Binary memory creation**  
   - If the SDK’s `create_memory` doesn’t match our multipart usage, add a helper that builds the request (or uses the right SDK method) so we still have a single “insert binary memory” entry point.

4. **Batch get**  
   - SDK has `MemoriesApi.batch_get_memory` ([POST /v1/memories:batchGet](https://pypi.org/project/goodmem-client/)). We’d call it and, if needed, map the response to a list of dicts for existing code.

5. **Pagination for list_spaces**  
   - Our client paginates with `nextToken`. If the SDK’s `list_spaces` returns one page, we’d implement a small loop (or use SDK pagination if provided) so “list all spaces” / “find by name” still works with many spaces.

---

## Recommendation summary

- **Staying with the custom client** is reasonable if you want minimal dependencies, full control over request/response shapes, and no churn from SDK upgrades. You already have batch get, retrieve-as-list, and name-filtered list; maintenance is mainly keeping in sync with the Goodmem API when it changes.

- **Switching to goodmem-client** is attractive if you want to rely on the official client for correctness and new features (filters, streaming, post-processors). The extra work is a **thin wrapper layer** that:
  - Provides `get_space_id_by_name` (and optionally `list_spaces` with name filter if the API supports it).
  - Exposes a sync “retrieve → list of chunks” and “batch get → list of dicts” so the plugin and tools don’t need to deal with streaming or SDK types.
  - Maps binary insert to the SDK’s create-memory API.
  - Keeps your existing `GoodmemClient`-style interface (or a close equivalent) so plugin and tools change as little as possible.

If you adopt the SDK, do it behind a small facade (e.g. `GoodmemClient` implemented via goodmem-client) so call sites stay the same and you can swap or reimplement the backend later.
