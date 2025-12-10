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

from datetime import datetime
from typing import Any
from typing import Optional
from unittest import mock

from google.adk.artifacts.base_artifact_service import ArtifactVersion
from google.adk_community.artifacts import S3ArtifactService
from google.genai import types
import pytest

# Define a fixed datetime object for consistent testing
FIXED_DATETIME = datetime(2025, 1, 1, 12, 0, 0)


class MockS3Object:
  """Mocks an S3 object."""

  def __init__(self, key: str) -> None:
    self.key = key
    self.data: Optional[bytes] = None
    self.content_type: Optional[str] = None
    self.last_modified = FIXED_DATETIME
    self.metadata: dict[str, Any] = {}

  def set_data(self, data: bytes, content_type: str, metadata: dict[str, Any]):
    """Sets the object data."""
    self.data = data
    self.content_type = content_type
    self.metadata = metadata or {}


class MockS3Bucket:
  """Mocks an S3 bucket."""

  def __init__(self, name: str) -> None:
    self.name = name
    self.objects: dict[str, MockS3Object] = {}


class MockS3Client:
  """Mocks the boto3 S3 client."""

  def __init__(self, **kwargs) -> None:
    self.buckets: dict[str, MockS3Bucket] = {}
    self.exceptions = type(
        "Exceptions", (), {"NoSuchKey": KeyError, "NoSuchBucket": Exception}
    )()

  def head_bucket(self, Bucket: str):
    """Mocks head_bucket call."""
    if Bucket not in self.buckets:
      self.buckets[Bucket] = MockS3Bucket(Bucket)
    return {}

  def put_object(
      self,
      Bucket: str,
      Key: str,
      Body: bytes,
      ContentType: str,
      Metadata: Optional[dict[str, str]] = None,
      **kwargs,
  ):
    """Mocks put_object call."""
    if Bucket not in self.buckets:
      self.buckets[Bucket] = MockS3Bucket(Bucket)
    bucket = self.buckets[Bucket]
    if Key not in bucket.objects:
      bucket.objects[Key] = MockS3Object(Key)
    bucket.objects[Key].set_data(Body, ContentType, Metadata or {})

  def get_object(self, Bucket: str, Key: str):
    """Mocks get_object call."""
    bucket = self.buckets.get(Bucket)
    if not bucket or Key not in bucket.objects:
      raise self.exceptions.NoSuchKey(f"Object {Key} not found")
    obj = bucket.objects[Key]
    if obj.data is None:
      raise self.exceptions.NoSuchKey(f"Object {Key} not found")

    class MockBody:

      def __init__(self, data: bytes):
        self._data = data

      def read(self) -> bytes:
        return self._data

    return {
        "Body": MockBody(obj.data),
        "ContentType": obj.content_type,
        "LastModified": obj.last_modified,
        "Metadata": obj.metadata,
    }

  def head_object(self, Bucket: str, Key: str):
    """Mocks head_object call."""
    bucket = self.buckets.get(Bucket)
    if not bucket or Key not in bucket.objects:
      raise self.exceptions.NoSuchKey(f"Object {Key} not found")
    obj = bucket.objects[Key]
    if obj.data is None:
      raise self.exceptions.NoSuchKey(f"Object {Key} not found")
    return {
        "ContentType": obj.content_type,
        "LastModified": obj.last_modified,
        "Metadata": obj.metadata,
    }

  def delete_object(self, Bucket: str, Key: str):
    """Mocks delete_object call."""
    bucket = self.buckets.get(Bucket)
    if bucket and Key in bucket.objects:
      del bucket.objects[Key]

  def list_objects_v2(self, Bucket: str, Prefix: str = ""):
    """Mocks list_objects_v2 call."""
    bucket = self.buckets.get(Bucket)
    if not bucket:
      return {}

    contents = []
    for key, obj in bucket.objects.items():
      if key.startswith(Prefix) and obj.data is not None:
        contents.append({"Key": key})

    if contents:
      return {"Contents": contents}
    return {}


@pytest.fixture
def mock_s3_service():
  """Provides a mocked S3ArtifactService for testing."""
  with mock.patch("boto3.client", return_value=MockS3Client()):
    return S3ArtifactService(bucket_name="test_bucket")


@pytest.mark.asyncio
async def test_load_empty(mock_s3_service):
  """Tests loading an artifact when none exists."""
  assert not await mock_s3_service.load_artifact(
      app_name="test_app",
      user_id="test_user",
      session_id="session_id",
      filename="filename",
  )


@pytest.mark.asyncio
async def test_save_load_delete(mock_s3_service):
  """Tests saving, loading, and deleting an artifact."""
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "file456"

  await mock_s3_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=artifact,
  )
  assert (
      await mock_s3_service.load_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
      )
      == artifact
  )

  # Attempt to load a version that doesn't exist
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
  """Tests listing keys in the artifact service."""
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "filename"
  filenames = [filename + str(i) for i in range(5)]

  for f in filenames:
    await mock_s3_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=f,
        artifact=artifact,
    )

  assert (
      await mock_s3_service.list_artifact_keys(
          app_name=app_name, user_id=user_id, session_id=session_id
      )
      == filenames
  )


@pytest.mark.asyncio
async def test_list_versions(mock_s3_service):
  """Tests listing versions of an artifact."""
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "with/slash/filename"
  versions = [
      types.Part.from_bytes(
          data=i.to_bytes(2, byteorder="big"), mime_type="text/plain"
      )
      for i in range(3)
  ]
  versions.append(types.Part.from_text(text="hello"))

  for i in range(4):
    await mock_s3_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        artifact=versions[i],
    )

  response_versions = await mock_s3_service.list_versions(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )

  assert response_versions == list(range(4))


@pytest.mark.asyncio
async def test_list_keys_preserves_user_prefix(mock_s3_service):
  """Tests that list_artifact_keys preserves 'user:' prefix in returned names."""
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"

  # Save artifacts with "user:" prefix (cross-session artifacts)
  await mock_s3_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="user:document.pdf",
      artifact=artifact,
  )

  await mock_s3_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="user:image.png",
      artifact=artifact,
  )

  # Save session-scoped artifact without prefix
  await mock_s3_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="session_file.txt",
      artifact=artifact,
  )

  # List artifacts should return names with "user:" prefix for user-scoped
  artifact_keys = await mock_s3_service.list_artifact_keys(
      app_name=app_name, user_id=user_id, session_id=session_id
  )

  # Should contain prefixed names and session file
  expected_keys = ["session_file.txt", "user:document.pdf", "user:image.png"]
  assert sorted(artifact_keys) == sorted(expected_keys)


@pytest.mark.asyncio
async def test_list_artifact_versions_and_get_artifact_version(
    mock_s3_service,
):
  """Tests listing artifact versions and getting a specific version."""
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "filename"
  versions = [
      types.Part.from_bytes(
          data=i.to_bytes(2, byteorder="big"), mime_type="text/plain"
      )
      for i in range(4)
  ]

  for i in range(4):
    custom_metadata = {"key": "value" + str(i)}
    await mock_s3_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        artifact=versions[i],
        custom_metadata=custom_metadata,
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
    assert av.canonical_uri == f"s3://test_bucket/{app_name}/{user_id}/{session_id}/{filename}/{i}"
    assert av.custom_metadata["key"] == f"value{i}"
    assert av.mime_type == "text/plain"

  # Get latest artifact version when version is not specified
  latest = await mock_s3_service.get_artifact_version(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )
  assert latest is not None
  assert latest.version == 3

  # Get artifact version by version number
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
  """Tests listing artifact versions with user prefix."""
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  user_scoped_filename = "user:document.pdf"
  versions = [
      types.Part.from_bytes(
          data=i.to_bytes(2, byteorder="big"), mime_type="text/plain"
      )
      for i in range(4)
  ]

  for i in range(4):
    custom_metadata = {"key": "value" + str(i)}
    await mock_s3_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=user_scoped_filename,
        artifact=versions[i],
        custom_metadata=custom_metadata,
    )

  artifact_versions = await mock_s3_service.list_artifact_versions(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=user_scoped_filename,
  )

  assert len(artifact_versions) == 4
  for i, av in enumerate(artifact_versions):
    assert av.version == i
    # User-scoped: {app}/{user}/user/document.pdf/{version}
    assert av.canonical_uri == f"s3://test_bucket/{app_name}/{user_id}/user/document.pdf/{i}"


@pytest.mark.asyncio
async def test_get_artifact_version_artifact_does_not_exist(mock_s3_service):
  """Tests getting an artifact version when artifact does not exist."""
  assert not await mock_s3_service.get_artifact_version(
      app_name="test_app",
      user_id="test_user",
      session_id="session_id",
      filename="filename",
  )


@pytest.mark.asyncio
async def test_get_artifact_version_out_of_index(mock_s3_service):
  """Tests loading an artifact with an out-of-index version."""
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "filename"
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")

  await mock_s3_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=artifact,
  )

  # Attempt to get a version that doesn't exist
  assert not await mock_s3_service.get_artifact_version(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      version=3,
  )


@pytest.mark.asyncio
async def test_special_characters_in_filename(mock_s3_service):
  """Tests URL encoding for special characters in filenames."""
  artifact = types.Part(text="Test content")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  # Filename with special characters that need encoding
  filename = "my file/with:special&chars.txt"

  version = await mock_s3_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=artifact,
  )

  loaded = await mock_s3_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )

  assert loaded is not None
  # Loaded artifacts come back as inline_data (bytes), not text
  assert loaded.inline_data is not None
  assert loaded.inline_data.data == b"Test content"


@pytest.mark.asyncio
async def test_custom_metadata(mock_s3_service):
  """Tests custom metadata storage and retrieval."""
  artifact = types.Part(text="Test")
  custom_metadata = {"author": "test", "tags": "integration,test"}

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
  assert version_info.custom_metadata["tags"] == "integration,test"


@pytest.mark.asyncio
async def test_empty_artifact(mock_s3_service):
  """Tests saving and loading empty (0-byte) artifacts."""
  # Create empty artifact
  empty_artifact = types.Part.from_bytes(data=b"", mime_type="text/plain")
  
  version = await mock_s3_service.save_artifact(
      app_name="app0",
      user_id="user0",
      session_id="123",
      filename="empty.txt",
      artifact=empty_artifact,
  )
  
  assert version == 0
  
  # Load empty artifact - should succeed, not return None
  loaded = await mock_s3_service.load_artifact(
      app_name="app0",
      user_id="user0",
      session_id="123",
      filename="empty.txt",
  )
  
  assert loaded is not None
  assert loaded.inline_data is not None
  assert loaded.inline_data.data == b""

