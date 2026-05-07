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

"""Artifact service implementation using Amazon S3."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
from typing import Any
from typing import Optional

from google.adk.artifacts.base_artifact_service import ArtifactVersion
from google.adk.artifacts.base_artifact_service import BaseArtifactService
from google.genai import types
from typing_extensions import override

logger = logging.getLogger("google_adk_community." + __name__)


class S3ArtifactService(BaseArtifactService):
  """An S3-backed implementation of the artifact service."""

  def __init__(
      self,
      bucket_name: str,
      aws_configs: Optional[dict[str, Any]] = None,
      save_max_retries: int = -1,
  ):
    """Initializes the S3 artifact service.

    Args:
        bucket_name: The name of the S3 bucket to use.
        aws_configs: Extra kwargs forwarded to the aioboto3 S3 client.
            Use this to pass region_name, endpoint_url (for MinIO), etc.
        save_max_retries: Maximum retries on version conflict. -1 means
            retry indefinitely.
    """
    try:
      import aioboto3  # noqa: F401
    except ImportError as exc:
      raise ImportError(
          "aioboto3 is required to use S3ArtifactService. "
          "Install it with: pip install google-adk-community[s3]"
      ) from exc

    self.bucket_name = bucket_name
    self.aws_configs: dict[str, Any] = aws_configs or {}
    self.save_max_retries = save_max_retries
    self._session = None
    self._session_lock = asyncio.Lock()

  async def _get_session(self):
    import aioboto3

    async with self._session_lock:
      if self._session is None:
        self._session = aioboto3.Session()
    return self._session

  @asynccontextmanager
  async def _client(self):
    session = await self._get_session()
    async with session.client(
        service_name="s3", **self.aws_configs
    ) as s3:
      yield s3

  # S3 user-defined metadata is limited to 2 KB total.  S3 prefixes each
  # key with ``x-amz-meta-`` (11 bytes) in the header, so we include that
  # overhead per key when computing the total size.
  _S3_METADATA_MAX_BYTES = 2048
  _S3_META_PREFIX_LEN = len("x-amz-meta-")

  @staticmethod
  def _flatten_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, str]:
    """JSON-encode metadata values for S3 user-metadata.

    Raises:
        ValueError: If the encoded metadata exceeds the S3 2 KB limit.
    """
    if not metadata:
      return {}
    flat = {str(k): json.dumps(v) for k, v in metadata.items()}
    # Include the x-amz-meta- prefix overhead that S3 adds per key.
    total = sum(
        S3ArtifactService._S3_META_PREFIX_LEN
        + len(k.encode())
        + len(v.encode())
        for k, v in flat.items()
    )
    if total > S3ArtifactService._S3_METADATA_MAX_BYTES:
      raise ValueError(
          f"Custom metadata ({total} bytes including S3 header "
          f"overhead) exceeds the S3 user-metadata limit of "
          f"{S3ArtifactService._S3_METADATA_MAX_BYTES} bytes."
      )
    return flat

  @staticmethod
  def _unflatten_metadata(metadata: Optional[dict[str, str]]) -> dict[str, Any]:
    """Decode JSON metadata back to native Python objects."""
    results: dict[str, Any] = {}
    for k, v in (metadata or {}).items():
      try:
        results[k] = json.loads(v)
      except json.JSONDecodeError:
        logger.warning(
            "Failed to decode metadata value for key %r. Using raw string.", k
        )
        results[k] = v
    return results

  @staticmethod
  def _file_has_user_namespace(filename: str) -> bool:
    return filename.startswith("user:")

  def _get_blob_prefix(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
  ) -> str:
    if self._file_has_user_namespace(filename):
      return f"{app_name}/{user_id}/user/{filename[5:]}"  # strip "user:"
    if session_id is None:
      raise ValueError(
          "session_id is required for session-scoped artifacts."
      )
    return f"{app_name}/{user_id}/{session_id}/{filename}"

  def _get_blob_name(
      self,
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
      version: int,
  ) -> str:
    return (
        f"{self._get_blob_prefix(app_name, user_id, session_id, filename)}"
        f"/{version}"
    )

  @override
  async def save_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      artifact: types.Part,
      session_id: Optional[str] = None,
      custom_metadata: Optional[dict[str, Any]] = None,
  ) -> int:
    """Save an artifact with atomic versioning via IfNoneMatch."""
    from botocore.exceptions import ClientError

    attempt = 0
    while True:
      if self.save_max_retries >= 0 and attempt > self.save_max_retries:
        break

      versions = await self.list_versions(
          app_name=app_name,
          user_id=user_id,
          filename=filename,
          session_id=session_id,
      )
      version = 0 if not versions else max(versions) + 1
      key = self._get_blob_name(
          app_name, user_id, session_id, filename, version
      )

      if artifact.inline_data:
        body = artifact.inline_data.data
        content_type = (
            artifact.inline_data.mime_type or "application/octet-stream"
        )
      elif artifact.text:
        body = artifact.text.encode("utf-8")
        content_type = "text/plain; charset=utf-8"
      else:
        raise ValueError(
            "Artifact must have either inline_data or text content."
        )

      async with self._client() as s3:
        try:
          await s3.put_object(
              Bucket=self.bucket_name,
              Key=key,
              Body=body,
              ContentType=content_type,
              Metadata=self._flatten_metadata(custom_metadata),
              IfNoneMatch="*",
          )
          logger.debug(
              "Saved artifact %s version %d to s3://%s/%s",
              filename,
              version,
              self.bucket_name,
              key,
          )
          return version
        except ClientError as e:
          code = e.response.get("Error", {}).get("Code", "")
          if code in ("PreconditionFailed", "ObjectAlreadyExists"):
            attempt += 1
            backoff = min(0.1 * (2 ** (attempt - 1)), 5.0)
            logger.debug(
                "Version conflict for %s version %d, retrying in "
                "%.2fs (attempt %d)…",
                filename,
                version,
                backoff,
                attempt,
            )
            await asyncio.sleep(backoff)
            continue
          raise

    raise RuntimeError(
        "Failed to save artifact due to version conflicts after "
        f"{self.save_max_retries} retries."
    )

  @override
  async def load_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
      version: Optional[int] = None,
  ) -> Optional[types.Part]:
    """Load a specific version of an artifact, or the latest."""
    from botocore.exceptions import ClientError

    if version is None:
      versions = await self.list_versions(
          app_name=app_name,
          user_id=user_id,
          filename=filename,
          session_id=session_id,
      )
      if not versions:
        return None
      version = max(versions)

    key = self._get_blob_name(
        app_name, user_id, session_id, filename, version
    )
    async with self._client() as s3:
      try:
        response = await s3.get_object(Bucket=self.bucket_name, Key=key)
        async with response["Body"] as stream:
          data = await stream.read()
        content_type = response.get("ContentType", "application/octet-stream")
      except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
          return None
        raise

    logger.debug(
        "Loaded artifact %s version %d from s3://%s/%s",
        filename,
        version,
        self.bucket_name,
        key,
    )
    # Return Part.from_text for text content types so consumers can
    # check ``part.text`` consistently.  Fall back to from_bytes for
    # binary content.
    if content_type.startswith("text/"):
      return types.Part.from_text(text=data.decode("utf-8"))
    return types.Part.from_bytes(data=data, mime_type=content_type)

  @override
  async def list_artifact_keys(
      self, *, app_name: str, user_id: str, session_id: Optional[str] = None
  ) -> list[str]:
    """List all artifact keys for a user, optionally filtered by session.

    Uses S3 ``Delimiter='/'`` with ``CommonPrefixes`` to retrieve only
    unique artifact names without listing every individual version
    object.
    """
    keys: set[str] = set()
    prefixes = [
        f"{app_name}/{user_id}/{session_id}/" if session_id else None,
        f"{app_name}/{user_id}/user/",
    ]
    async with self._client() as s3:
      for prefix in filter(None, prefixes):
        paginator = s3.get_paginator("list_objects_v2")
        async for page in paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=prefix,
            Delimiter="/",
        ):
          for cp in page.get("CommonPrefixes", []):
            # CommonPrefixes entries look like
            # "<prefix><artifact_name>/" — strip the prefix and
            # trailing slash to get the raw filename.
            raw_filename = cp["Prefix"][len(prefix):].rstrip("/")
            if not raw_filename:
              continue
            if prefix.endswith("/user/"):
              keys.add(f"user:{raw_filename}")
            else:
              keys.add(raw_filename)
    return sorted(keys)

  @override
  async def delete_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> None:
    """Delete all versions of an artifact."""
    versions = await self.list_versions(
        app_name=app_name,
        user_id=user_id,
        filename=filename,
        session_id=session_id,
    )
    if not versions:
      return

    keys_to_delete = [
        {
            "Key": self._get_blob_name(
                app_name, user_id, session_id, filename, v
            )
        }
        for v in versions
    ]
    async with self._client() as s3:
      for i in range(0, len(keys_to_delete), 1000):
        batch = keys_to_delete[i : i + 1000]
        await s3.delete_objects(
            Bucket=self.bucket_name, Delete={"Objects": batch}
        )

  @override
  async def list_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[int]:
    """List all available versions of an artifact."""
    prefix = (
        self._get_blob_prefix(app_name, user_id, session_id, filename) + "/"
    )
    versions: list[int] = []
    async with self._client() as s3:
      paginator = s3.get_paginator("list_objects_v2")
      async for page in paginator.paginate(
          Bucket=self.bucket_name, Prefix=prefix
      ):
        for obj in page.get("Contents", []):
          version_str = obj["Key"].split("/")[-1]
          try:
            versions.append(int(version_str))
          except ValueError:
            continue
    return sorted(versions)

  @override
  async def list_artifact_versions(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
  ) -> list[ArtifactVersion]:
    """List all versions with metadata."""
    prefix = (
        self._get_blob_prefix(app_name, user_id, session_id, filename) + "/"
    )
    results: list[ArtifactVersion] = []
    # Limit concurrent head_object calls to avoid S3 rate-limiting.
    sem = asyncio.Semaphore(10)

    async with self._client() as s3:

      async def _head(key: str):
        async with sem:
          return await s3.head_object(Bucket=self.bucket_name, Key=key)

      paginator = s3.get_paginator("list_objects_v2")
      async for page in paginator.paginate(
          Bucket=self.bucket_name, Prefix=prefix
      ):
        page_objects = page.get("Contents", [])
        if not page_objects:
          continue

        head_tasks = [_head(obj["Key"]) for obj in page_objects]
        heads = await asyncio.gather(*head_tasks)

        for obj, head in zip(page_objects, heads):
          version_str = obj["Key"].split("/")[-1]
          try:
            version = int(version_str)
          except ValueError:
            continue
          results.append(
              ArtifactVersion(
                  version=version,
                  canonical_uri=f"s3://{self.bucket_name}/{obj['Key']}",
                  custom_metadata=self._unflatten_metadata(
                      head.get("Metadata", {})
                  ),
                  create_time=obj["LastModified"].timestamp(),
                  mime_type=head["ContentType"],
              )
          )
    return sorted(results, key=lambda a: a.version)

  @override
  async def get_artifact_version(
      self,
      *,
      app_name: str,
      user_id: str,
      filename: str,
      session_id: Optional[str] = None,
      version: Optional[int] = None,
  ) -> Optional[ArtifactVersion]:
    """Retrieve metadata for a specific version, or the latest."""
    from botocore.exceptions import ClientError

    if version is None:
      all_versions = await self.list_versions(
          app_name=app_name,
          user_id=user_id,
          filename=filename,
          session_id=session_id,
      )
      if not all_versions:
        return None
      version = max(all_versions)

    key = self._get_blob_name(
        app_name, user_id, session_id, filename, version
    )
    async with self._client() as s3:
      try:
        head = await s3.head_object(Bucket=self.bucket_name, Key=key)
      except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
          return None
        raise

    return ArtifactVersion(
        version=version,
        canonical_uri=f"s3://{self.bucket_name}/{key}",
        custom_metadata=self._unflatten_metadata(
            head.get("Metadata", {})
        ),
        create_time=head["LastModified"].timestamp(),
        mime_type=head["ContentType"],
    )
