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

"""Example agent demonstrating S3 artifact storage.

This example shows how to configure an ADK agent to use Amazon S3 for
artifact storage using the community S3ArtifactService.

Before running:
1. Install: pip install google-adk-community boto3
2. Set AWS credentials (see README.md)
3. Create S3 bucket
4. Update bucket name below or set ADK_S3_BUCKET environment variable
"""
from __future__ import annotations

import os

from google.adk import Agent
from google.adk.apps import App
from google.adk_community.artifacts import S3ArtifactService

# Get bucket name from environment or use default
BUCKET_NAME = os.getenv("ADK_S3_BUCKET", "my-adk-artifacts")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Initialize S3 artifact service
artifact_service = S3ArtifactService(
    bucket_name=BUCKET_NAME,
    region_name=AWS_REGION,
)

# Define the agent
root_agent = Agent(
    name="s3_artifact_agent",
    model="gemini-2.0-flash",
    instruction="""You are a helpful assistant that can save and retrieve files.
    
When users ask you to save information, use the save_artifact tool to store
it in S3. When they ask for previously saved information, use the load_artifact
tool to retrieve it.

Examples:
- "Save this report to a file called quarterly_report.pdf"
- "Load the file called quarterly_report.pdf"
- "List all my saved files"
""",
    description="An assistant that demonstrates S3 artifact storage",
)

# Create app with S3 artifact service
app = App(
    name="s3_artifact_example",
    root_agent=root_agent,
    artifact_service=artifact_service,
)

