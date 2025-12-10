# Community Artifact Services

This module contains community-contributed artifact service implementations for ADK.

## Available Services

### S3ArtifactService

Production-ready artifact storage using Amazon S3.

**Installation:**
```bash
pip install google-adk-community boto3
```

**Usage:**
```python
from google.adk_community.artifacts import S3ArtifactService

artifact_service = S3ArtifactService(
    bucket_name="my-adk-artifacts",
    region_name="us-east-1"
)
```

**Features:**
- Session-scoped and user-scoped artifacts
- Automatic version management
- Custom metadata support
- URL encoding for special characters
- Works with S3-compatible services (MinIO, DigitalOcean Spaces, etc.)

**See Also:**
- [S3ArtifactService Implementation](./s3_artifact_service.py)
- [Example Agent](../../../contributing/samples/s3_artifact_example/)
- [Tests](../../../tests/unittests/artifacts/test_s3_artifact_service.py)

## Contributing

Want to add a new artifact service? See our [contribution guide](../../../CONTRIBUTING.md).

Examples of artifact services to contribute:
- Azure Blob Storage
- Google Drive
- Dropbox
- MinIO (dedicated implementation)
- Any S3-compatible service

