# Community Artifact Services

This module contains community-contributed artifact service implementations for ADK.

## Available Services

### S3ArtifactService

Production-ready artifact storage using Amazon S3 (or any S3-compatible service such as MinIO, DigitalOcean Spaces, etc.).

**Installation:**
```bash
pip install google-adk-community[s3]
```

**Usage:**
```python
from google.adk_community.artifacts import S3ArtifactService

artifact_service = S3ArtifactService(
    bucket_name="my-adk-artifacts",
    aws_configs={"region_name": "us-east-1"},
)
```

**Features:**
- Native async I/O via `aioboto3` (no `asyncio.to_thread` wrappers)
- Atomic versioning using S3 conditional writes (`IfNoneMatch`)
- Session-scoped and user-scoped artifacts
- Automatic version management
- Custom metadata support (JSON-serialised)
- Batch delete for efficient cleanup
- Paginated listing for large artifact collections
- Works with S3-compatible services (MinIO, DigitalOcean Spaces, etc.)

**See Also:**
- [S3ArtifactService Implementation](./s3_artifact_service.py)
- [Tests](../../../tests/unittests/artifacts/test_s3_artifact_service.py)
