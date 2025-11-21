# pylint: disable=missing-class-docstring,missing-function-docstring

"""Tests for the artifact service."""

import asyncio
from datetime import datetime
import enum
from pathlib import Path
import random
import sys
from unittest.mock import patch

from botocore.exceptions import ClientError
from google.adk.artifacts.base_artifact_service import ArtifactVersion
from google.genai import types
import pytest

from google.adk_community.artifacts.s3_artifact_service import S3ArtifactService

Enum = enum.Enum

# Define a fixed datetime object to be returned by datetime.now()
FIXED_DATETIME = datetime(2025, 1, 1, 12, 0, 0)


class ArtifactServiceType(Enum):
  S3 = "S3"


class MockBody:

  def __init__(self, data: bytes):
    self._data = data

  async def read(self) -> bytes:
    return self._data

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc, tb):
    pass


class MockAsyncS3Object:

  def __init__(self, key):
    self.key = key
    self.data = None
    self.content_type = None
    self.metadata = {}
    self.last_modified = FIXED_DATETIME

  async def put(self, Body, ContentType=None, Metadata=None):
    self.data = Body if isinstance(Body, bytes) else Body.encode("utf-8")
    self.content_type = ContentType
    self.metadata = Metadata or {}

  async def get(self):
    if self.data is None:
      raise ClientError(
          {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}},
          operation_name="GetObject",
      )
    return {
        "Body": MockBody(self.data),
        "ContentType": self.content_type,
        "Metadata": self.metadata,
        "LastModified": self.last_modified,
    }


class MockAsyncS3Bucket:

  def __init__(self, name):
    self.name = name
    self.objects = {}

  def object(self, key):
    if key not in self.objects:
      self.objects[key] = MockAsyncS3Object(key)
    return self.objects[key]

  async def listed_keys(self, prefix=None):
    return [
        k
        for k, obj in self.objects.items()
        if obj.data is not None and (prefix is None or k.startswith(prefix))
    ]


class MockAsyncS3Client:

  def __init__(self):
    self.buckets = {}

  def get_bucket(self, bucket_name):
    if bucket_name not in self.buckets:
      self.buckets[bucket_name] = MockAsyncS3Bucket(bucket_name)
    return self.buckets[bucket_name]

  async def put_object(
      self, Bucket, Key, Body, ContentType=None, Metadata=None, IfNoneMatch=None
  ):
    await asyncio.sleep(random.uniform(0, 0.05))
    bucket = self.get_bucket(Bucket)
    obj_exists = Key in bucket.objects and bucket.objects[Key].data is not None

    if IfNoneMatch == "*" and obj_exists:
      raise ClientError(
          {"Error": {"Code": "PreconditionFailed", "Message": "Object exists"}},
          operation_name="PutObject",
      )

    await bucket.object(Key).put(
        Body=Body, ContentType=ContentType, Metadata=Metadata
    )

  async def get_object(self, Bucket, Key):
    bucket = self.get_bucket(Bucket)
    obj = bucket.object(Key)
    return await obj.get()

  async def delete_object(self, Bucket, Key):
    bucket = self.get_bucket(Bucket)
    bucket.objects.pop(Key, None)

  async def delete_objects(self, Bucket, Delete):
    bucket = self.get_bucket(Bucket)
    for item in Delete.get("Objects", []):
      key = item.get("Key")
      if key in bucket.objects:
        bucket.objects.pop(key)

  async def list_objects_v2(self, Bucket, Prefix=None):
    bucket = self.get_bucket(Bucket)
    keys = await bucket.listed_keys(Prefix)
    return {
        "KeyCount": len(keys),
        "Contents": [
            {"Key": k, "LastModified": bucket.objects[k].last_modified}
            for k in keys
        ],
    }

  async def head_object(self, Bucket, Key):
    obj = await self.get_object(Bucket, Key)
    return {
        "ContentType": obj["ContentType"],
        "Metadata": obj.get("Metadata", {}),
        "LastModified": obj.get("LastModified"),
    }

  def get_paginator(self, operation_name):
    if operation_name != "list_objects_v2":
      raise NotImplementedError(
          f"Paginator for {operation_name} not implemented"
      )

    class MockAsyncPaginator:

      def __init__(self, client, Bucket, Prefix=None):
        self.client = client
        self.Bucket = Bucket
        self.Prefix = Prefix

      async def __aiter__(self):
        response = await self.client.list_objects_v2(
            Bucket=self.Bucket, Prefix=self.Prefix
        )
        contents = response.get("Contents", [])
        page_size = 2
        for i in range(0, len(contents), page_size):
          yield {
              "KeyCount": len(contents[i : i + page_size]),
              "Contents": contents[i : i + page_size],
          }

    class MockPaginator:

      def paginate(inner_self, Bucket, Prefix=None):
        return MockAsyncPaginator(self, Bucket, Prefix)

    return MockPaginator()


def mock_s3_artifact_service(monkeypatch):
  mock_s3_client = MockAsyncS3Client()

  class MockAioboto3:

    class Session:

      def client(self, *args, **kwargs):
        class MockClientCtx:

          async def __aenter__(self_inner):
            return mock_s3_client

          async def __aexit__(self_inner, exc_type, exc, tb):
            pass

        return MockClientCtx()

  monkeypatch.setitem(sys.modules, "aioboto3", MockAioboto3)
  artifact_service = S3ArtifactService(bucket_name="test_bucket")
  return artifact_service


@pytest.fixture
def artifact_service_factory(tmp_path: Path, monkeypatch):
  """Provides an artifact service constructor bound to the test tmp path."""

  def factory(
      service_type: ArtifactServiceType = ArtifactServiceType.S3,
  ):
    if service_type == ArtifactServiceType.S3:
      return mock_s3_artifact_service(monkeypatch)

  return factory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type",
    [
        ArtifactServiceType.S3,
    ],
)
async def test_load_empty(service_type, artifact_service_factory):
  """Tests loading an artifact when none exists."""
  artifact_service = artifact_service_factory(service_type)
  assert not await artifact_service.load_artifact(
      app_name="test_app",
      user_id="test_user",
      session_id="session_id",
      filename="filename",
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type",
    [
        ArtifactServiceType.S3,
    ],
)
async def test_save_load_delete(service_type, artifact_service_factory):
  """Tests saving, loading, and deleting an artifact."""
  artifact_service = artifact_service_factory(service_type)
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "file456"

  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=artifact,
  )
  assert (
      await artifact_service.load_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
      )
      == artifact
  )

  # Attempt to load a version that doesn't exist
  assert not await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      version=3,
  )

  await artifact_service.delete_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )
  assert not await artifact_service.load_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type",
    [
        ArtifactServiceType.S3,
    ],
)
async def test_list_keys(service_type, artifact_service_factory):
  """Tests listing keys in the artifact service."""
  artifact_service = artifact_service_factory(service_type)
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "filename"
  filenames = [filename + str(i) for i in range(5)]

  for f in filenames:
    await artifact_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=f,
        artifact=artifact,
    )

  assert (
      await artifact_service.list_artifact_keys(
          app_name=app_name, user_id=user_id, session_id=session_id
      )
      == filenames
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type",
    [
        ArtifactServiceType.S3,
    ],
)
async def test_list_versions(service_type, artifact_service_factory):
  """Tests listing versions of an artifact."""
  artifact_service = artifact_service_factory(service_type)

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
    await artifact_service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        artifact=versions[i],
    )

  response_versions = await artifact_service.list_versions(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
  )

  assert response_versions == list(range(4))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type",
    [
        ArtifactServiceType.S3,
    ],
)
async def test_list_keys_preserves_user_prefix(
    service_type, artifact_service_factory
):
  """Tests that list_artifact_keys preserves 'user:' prefix in returned names."""
  artifact_service = artifact_service_factory(service_type)
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")
  app_name = "app0"
  user_id = "user0"
  session_id = "123"

  # Save artifacts with "user:" prefix (cross-session artifacts)
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="user:document.pdf",
      artifact=artifact,
  )

  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="user:image.png",
      artifact=artifact,
  )

  # Save session-scoped artifact without prefix
  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename="session_file.txt",
      artifact=artifact,
  )

  # List artifacts should return names with "user:" prefix for user-scoped artifacts
  artifact_keys = await artifact_service.list_artifact_keys(
      app_name=app_name, user_id=user_id, session_id=session_id
  )

  # Should contain prefixed names and session file
  expected_keys = ["user:document.pdf", "user:image.png", "session_file.txt"]
  assert sorted(artifact_keys) == sorted(expected_keys)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type",
    [
        ArtifactServiceType.S3,
    ],
)
async def test_list_artifact_versions_and_get_artifact_version(
    service_type, artifact_service_factory
):
  """Tests listing artifact versions and getting a specific version."""
  artifact_service = artifact_service_factory(service_type)
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

  with patch(
      "google.adk.artifacts.base_artifact_service.datetime"
  ) as mock_datetime:
    mock_datetime.now.return_value = FIXED_DATETIME

    for i in range(4):
      custom_metadata = {"key": "value" + str(i)}
      await artifact_service.save_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
          artifact=versions[i],
          custom_metadata=custom_metadata,
      )

    artifact_versions = await artifact_service.list_artifact_versions(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )

    expected_artifact_versions = []
    for i in range(4):
      metadata = {"key": "value" + str(i)}
      if service_type == ArtifactServiceType.S3:
        uri = (
            f"s3://test_bucket/{app_name}/{user_id}/{session_id}/{filename}/{i}"
        )
      else:
        uri = f"memory://apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{filename}/versions/{i}"
      expected_artifact_versions.append(
          ArtifactVersion(
              version=i,
              canonical_uri=uri,
              custom_metadata=metadata,
              mime_type="text/plain",
              create_time=FIXED_DATETIME.timestamp(),
          )
      )
    assert artifact_versions == expected_artifact_versions

    # Get latest artifact version when version is not specified
    assert (
        await artifact_service.get_artifact_version(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
        )
        == expected_artifact_versions[-1]
    )

    # Get artifact version by version number
    assert (
        await artifact_service.get_artifact_version(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
            version=2,
        )
        == expected_artifact_versions[2]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type",
    [
        ArtifactServiceType.S3,
    ],
)
async def test_list_artifact_versions_with_user_prefix(
    service_type, artifact_service_factory
):
  """Tests listing artifact versions with user prefix."""
  artifact_service = artifact_service_factory(service_type)
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

  with patch(
      "google.adk.artifacts.base_artifact_service.datetime"
  ) as mock_datetime:
    mock_datetime.now.return_value = FIXED_DATETIME

    for i in range(4):
      custom_metadata = {"key": "value" + str(i)}
      # Save artifacts with "user:" prefix (cross-session artifacts)
      await artifact_service.save_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=user_scoped_filename,
          artifact=versions[i],
          custom_metadata=custom_metadata,
      )

    artifact_versions = await artifact_service.list_artifact_versions(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=user_scoped_filename,
    )

    expected_artifact_versions = []
    for i in range(4):
      metadata = {"key": "value" + str(i)}
      if service_type == ArtifactServiceType.S3:
        uri = f"s3://test_bucket/{app_name}/{user_id}/user/{user_scoped_filename}/{i}"
      else:
        uri = f"memory://apps/{app_name}/users/{user_id}/artifacts/{user_scoped_filename}/versions/{i}"
      expected_artifact_versions.append(
          ArtifactVersion(
              version=i,
              canonical_uri=uri,
              custom_metadata=metadata,
              mime_type="text/plain",
              create_time=FIXED_DATETIME.timestamp(),
          )
      )
    assert artifact_versions == expected_artifact_versions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type",
    [
        ArtifactServiceType.S3,
    ],
)
async def test_get_artifact_version_artifact_does_not_exist(
    service_type, artifact_service_factory
):
  """Tests getting an artifact version when artifact does not exist."""
  artifact_service = artifact_service_factory(service_type)
  assert not await artifact_service.get_artifact_version(
      app_name="test_app",
      user_id="test_user",
      session_id="session_id",
      filename="filename",
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type",
    [
        ArtifactServiceType.S3,
    ],
)
async def test_get_artifact_version_out_of_index(
    service_type, artifact_service_factory
):
  """Tests loading an artifact with an out-of-index version."""
  artifact_service = artifact_service_factory(service_type)
  app_name = "app0"
  user_id = "user0"
  session_id = "123"
  filename = "filename"
  artifact = types.Part.from_bytes(data=b"test_data", mime_type="text/plain")

  await artifact_service.save_artifact(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      artifact=artifact,
  )

  # Attempt to get a version that doesn't exist
  assert not await artifact_service.get_artifact_version(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
      filename=filename,
      version=3,
  )
