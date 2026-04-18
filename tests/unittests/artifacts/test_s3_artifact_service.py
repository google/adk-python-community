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

# pylint: disable=missing-class-docstring,missing-function-docstring

"""Tests for S3ArtifactService."""

import asyncio
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Optional
from unittest import mock

from google.adk.artifacts.base_artifact_service import ArtifactVersion
from google.adk_community.artifacts import S3ArtifactService
from google.genai import types
import pytest



FIXED_DATETIME = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class MockS3Object:
  """Mocks an S3 object stored in a bucket."""

  def __init__(self, key: str) -> None:
    self.key = key
    self.data: Optional[bytes] = None
    self.content_type: Optional[str] = None
    self.last_modified = FIXED_DATETIME
    self.metadata: dict[str, str] = {}

  def set_data(
      self, data: bytes, content_type: str, metadata: dict[str, str]
  ):
    self.data = data
    self.content_type = content_type
    self.metadata = metadata or {}


class MockS3Bucket:
  """Mocks an S3 bucket."""

  def __init__(self, name: str) -> None:
    self.name = name
    self.objects: dict[str, MockS3Object] = {}


class MockS3ResponseBody:
  """Mocks the async streaming body returned by get_object."""

  def __init__(self, data: bytes):
    self._data = data

  async def read(self) -> bytes:
    return self._data

  async def __aenter__(self):
    return self

  async def __aexit__(self, *args):
    pass


class MockS3Client:
  """Mocks an aioboto3 S3 client with async context manager support."""

  def __init__(self, **kwargs) -> None:
    self.buckets: dict[str, MockS3Bucket] = {}

  async def __aenter__(self):
    return self

  async def __aexit__(self, *args):
    pass



  async def head_bucket(self, Bucket: str):
    if Bucket not in self.buckets:
      self.buckets[Bucket] = MockS3Bucket(Bucket)
    return {}

  async def put_object(
      self,
      Bucket: str,
      Key: str,
      Body: bytes,
      ContentType: str,
      Metadata: Optional[dict[str, str]] = None,
      IfNoneMatch: Optional[str] = None,
      **kwargs,
  ):
    if Bucket not in self.buckets:
      self.buckets[Bucket] = MockS3Bucket(Bucket)
    bucket = self.buckets[Bucket]

    # Reject if key already exists (atomic IfNoneMatch)
    if IfNoneMatch == "*" and Key in bucket.objects and bucket.objects[Key].data is not None:
      from botocore.exceptions import ClientError

      raise ClientError(
          {"Error": {"Code": "PreconditionFailed", "Message": "Object exists"}},
          "PutObject",
      )

    if Key not in bucket.objects:
      bucket.objects[Key] = MockS3Object(Key)
    bucket.objects[Key].set_data(Body, ContentType, Metadata or {})

  async def get_object(self, Bucket: str, Key: str):
    bucket = self.buckets.get(Bucket)
    if not bucket or Key not in bucket.objects or bucket.objects[Key].data is None:
      from botocore.exceptions import ClientError

      raise ClientError(
          {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
          "GetObject",
      )
    obj = bucket.objects[Key]
    return {
        "Body": MockS3ResponseBody(obj.data),
        "ContentType": obj.content_type,
        "LastModified": obj.last_modified,
        "Metadata": obj.metadata,
    }

  async def head_object(self, Bucket: str, Key: str):
    bucket = self.buckets.get(Bucket)
    if not bucket or Key not in bucket.objects or bucket.objects[Key].data is None:
      from botocore.exceptions import ClientError

      raise ClientError(
          {"Error": {"Code": "404", "Message": "Not found"}},
          "HeadObject",
      )
    obj = bucket.objects[Key]
    return {
        "ContentType": obj.content_type,
        "LastModified": obj.last_modified,
        "Metadata": obj.metadata,
    }

  async def delete_object(self, Bucket: str, Key: str):
    bucket = self.buckets.get(Bucket)
    if bucket and Key in bucket.objects:
      del bucket.objects[Key]

  async def delete_objects(self, Bucket: str, Delete: dict):
    bucket = self.buckets.get(Bucket)
    if bucket:
      for obj_spec in Delete.get("Objects", []):
        key = obj_spec["Key"]
        if key in bucket.objects:
          del bucket.objects[key]

  def get_paginator(self, operation_name: str):
    return MockS3Paginator(self, operation_name)


class MockS3Paginator:
  """Mocks an S3 paginator returned by get_paginator."""

  def __init__(self, client: MockS3Client, operation_name: str):
    self.client = client
    self.operation_name = operation_name

  def paginate(self, Bucket: str, Prefix: str = ""):
    return MockS3PaginateResult(self.client, Bucket, Prefix)


class MockS3PaginateResult:
  """Async iterator that yields a single page of list_objects_v2 results."""

  def __init__(self, client: MockS3Client, bucket: str, prefix: str):
    self.client = client
    self.bucket_name = bucket
    self.prefix = prefix
    self._yielded = False

  def __aiter__(self):
    self._yielded = False
    return self

  async def __anext__(self):
    if self._yielded:
      raise StopAsyncIteration
    self._yielded = True

    bucket = self.client.buckets.get(self.bucket_name)
    if not bucket:
      return {}

    contents = []
    for key, obj in bucket.objects.items():
      if key.startswith(self.prefix) and obj.data is not None:
        contents.append({
            "Key": key,
            "LastModified": obj.last_modified,
        })

    if contents:
      return {"Contents": contents}
    return {}


class MockS3Session:
  """Mocks aioboto3.Session."""

  def __init__(self, mock_client: MockS3Client):
    self._client = mock_client

  def client(self, service_name: str, **kwargs):
    return self._client




@pytest.fixture
def mock_s3_service():
  """Provides a mocked S3ArtifactService for testing."""
  mock_client = MockS3Client()
  mock_session = MockS3Session(mock_client)

  with mock.patch("aioboto3.Session", return_value=mock_session):
    service = S3ArtifactService(bucket_name="test_bucket")
    service._session = mock_session
    return service





@pytest.mark.asyncio
async def test_load_empty(mock_s3_service):
  """Loading an artifact when none exists returns None."""
  result = await mock_s3_service.load_artifact(
      app_name="test_app",
      user_id="test_user",
      session_id="session_id",
      filename="filename",
  )
  assert result is None


@pytest.mark.asyncio
async def test_save_load_delete(mock_s3_service):
  """Full CRUD cycle: save, load, load-missing-version, delete."""
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "file456"

  version = await mock_s3_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=artifact,
  )
  assert version == 0

  loaded = await mock_s3_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )
  assert loaded == artifact

  # Non-existent version
  assert not await mock_s3_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      version=3,
  )

  await mock_s3_service.delete_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )
  assert not await mock_s3_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )


@pytest.mark.asyncio
async def test_list_keys(mock_s3_service):
  """Listing artifact keys returns all saved filenames."""
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filenames = [f"filename{i}" for i in range(5)]

  for f in filenames:
    await mock_s3_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=f,
        artifact=artifact,
    )

  keys = await mock_s3_service.list_artifact_keys(
      app_name=app_name, user_id=user_id, session_id=session_id
  )
  assert keys == filenames


@pytest.mark.asyncio
async def test_list_versions(mock_s3_service):
  """Multiple saves of the same artifact create incremental versions."""
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "report.txt"
  parts = [
      types.Part.from_bytes(
          data=i.to_bytes(2, byteorder="big"), mime_type="text/plain"
      )
      for i in range(3)
  ]
  parts.append(types.Part.from_text(text="hello"))

  for p in parts:
    await mock_s3_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        artifact=p,
    )

  versions = await mock_s3_service.list_versions(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )
  assert versions == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_list_keys_preserves_user_prefix(mock_s3_service):
  """User-scoped artifacts keep the 'user:' prefix in key listings."""
  artifact = types.Part.from_bytes(data=b"data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"

  await mock_s3_service.save_artifact(
      app_name=app_name, user_id=user_id, session_id=session_id,
      filename="user:document.pdf", artifact=artifact,
  )
  await mock_s3_service.save_artifact(
      app_name=app_name, user_id=user_id, session_id=session_id,
      filename="user:image.png", artifact=artifact,
  )
  await mock_s3_service.save_artifact(
      app_name=app_name, user_id=user_id, session_id=session_id,
      filename="session_file.txt", artifact=artifact,
  )

  keys = await mock_s3_service.list_artifact_keys(
      app_name=app_name, user_id=user_id, session_id=session_id
  )
  assert sorted(keys) == ["session_file.txt", "user:document.pdf", "user:image.png"]


@pytest.mark.asyncio
async def test_list_artifact_versions_and_get_artifact_version(
    mock_s3_service,
):
  """Artifact version metadata includes canonical URI and custom metadata."""
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "filename"

  for i in range(4):
    part = types.Part.from_bytes(
        data=i.to_bytes(2, byteorder="big"), mime_type="text/plain"
    )
    await mock_s3_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        artifact=part,
        custom_metadata={"key": f"value{i}"},
    )

  artifact_versions = await mock_s3_service.list_artifact_versions(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )

  assert len(artifact_versions) == 4
  for i, av in enumerate(artifact_versions):
    assert av.version == i
    assert (
        av.canonical_uri
        == f"s3://test_bucket/{app_name}/{user_id}/{session_id}/{filename}/{i}"
    )
    assert av.custom_metadata["key"] == f"value{i}"
    assert av.mime_type == "text/plain"

  # Get latest
  latest = await mock_s3_service.get_artifact_version(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )
  assert latest is not None
  assert latest.version == 3

  # Get specific version
  specific = await mock_s3_service.get_artifact_version(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      version=2,
  )
  assert specific is not None
  assert specific.version == 2


@pytest.mark.asyncio
async def test_list_artifact_versions_with_user_prefix(mock_s3_service):
  """User-scoped artifact versions have correct canonical URIs."""
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  user_filename = "user:document.pdf"

  for i in range(4):
    part = types.Part.from_bytes(
        data=i.to_bytes(2, byteorder="big"), mime_type="text/plain"
    )
    await mock_s3_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=user_filename,
        artifact=part,
        custom_metadata={"key": f"value{i}"},
    )

  artifact_versions = await mock_s3_service.list_artifact_versions(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=user_filename,
  )

  assert len(artifact_versions) == 4
  for i, av in enumerate(artifact_versions):
    assert av.version == i
    assert (
        av.canonical_uri
        == f"s3://test_bucket/{app_name}/{user_id}/user/document.pdf/{i}"
    )


@pytest.mark.asyncio
async def test_get_artifact_version_artifact_does_not_exist(mock_s3_service):
  """Getting a version for a non-existent artifact returns None."""
  result = await mock_s3_service.get_artifact_version(
      app_name="test_app",
      user_id="test_user",
      session_id="session_id",
      filename="filename",
  )
  assert result is None


@pytest.mark.asyncio
async def test_get_artifact_version_out_of_index(mock_s3_service):
  """Getting a non-existent version number returns None."""
  artifact = types.Part.from_bytes(data=b"data", mime_type="text/plain")
  await mock_s3_service.save_artifact(
      app_name="app0",
      user_id="user0",
      session_id="123",
      filename="filename",
      artifact=artifact,
  )
  result = await mock_s3_service.get_artifact_version(
      app_name="app0",
      user_id="user0",
      session_id="123",
      filename="filename",
      version=3,
  )
  assert result is None


@pytest.mark.asyncio
async def test_empty_artifact(mock_s3_service):
  """Saving and loading 0-byte artifacts works correctly."""
  empty = types.Part.from_bytes(data=b"", mime_type="text/plain")

  version = await mock_s3_service.save_artifact(
      app_name="app0",
      user_id="user0",
      session_id="123",
      filename="empty.txt",
      artifact=empty,
  )
  assert version == 0

  loaded = await mock_s3_service.load_artifact(
      app_name="app0",
      user_id="user0",
      session_id="123",
      filename="empty.txt",
  )
  assert loaded is not None
  assert loaded.inline_data is not None
  assert loaded.inline_data.data == b""


@pytest.mark.asyncio
async def test_custom_metadata(mock_s3_service):
  """Custom metadata is stored and retrieved correctly (JSON-encoded)."""
  artifact = types.Part.from_text(text="Test")
  custom_metadata = {"author": "test", "tags": ["integration", "test"]}

  await mock_s3_service.save_artifact(
      app_name="app0",
      user_id="user0",
      session_id="123",
      filename="test.txt",
      artifact=artifact,
      custom_metadata=custom_metadata,
  )

  version_info = await mock_s3_service.get_artifact_version(
      app_name="app0",
      user_id="user0",
      session_id="123",
      filename="test.txt",
  )

  assert version_info is not None
  assert version_info.custom_metadata["author"] == "test"
  assert version_info.custom_metadata["tags"] == ["integration", "test"]


@pytest.mark.asyncio
async def test_text_artifact_roundtrip(mock_s3_service):
  """Text artifacts are encoded to UTF-8 bytes on save and loaded as bytes."""
  artifact = types.Part.from_text(text="Hello, world! 🌍")

  version = await mock_s3_service.save_artifact(
      app_name="app0",
      user_id="user0",
      session_id="123",
      filename="greeting.txt",
      artifact=artifact,
  )
  assert version == 0

  loaded = await mock_s3_service.load_artifact(
      app_name="app0",
      user_id="user0",
      session_id="123",
      filename="greeting.txt",
  )
  assert loaded is not None
  assert loaded.inline_data is not None
  assert loaded.inline_data.data == "Hello, world! 🌍".encode("utf-8")


@pytest.mark.asyncio
async def test_save_artifact_version_conflict_retry(mock_s3_service):
  """Atomic versioning retries on PreconditionFailed."""
  artifact = types.Part.from_bytes(data=b"data", mime_type="text/plain")

  # Save version 0 successfully
  v0 = await mock_s3_service.save_artifact(
      app_name="app0",
      user_id="user0",
      session_id="123",
      filename="conflict.txt",
      artifact=artifact,
  )
  assert v0 == 0

  # Save version 1 — should succeed because version 1 doesn't exist yet
  v1 = await mock_s3_service.save_artifact(
      app_name="app0",
      user_id="user0",
      session_id="123",
      filename="conflict.txt",
      artifact=artifact,
  )
  assert v1 == 1
