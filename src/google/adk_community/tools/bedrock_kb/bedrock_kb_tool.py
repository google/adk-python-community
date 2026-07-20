# Copyright 2026 Google LLC
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

"""Amazon Bedrock Knowledge Base retrieval tool for Google ADK.

Provides a function tool that queries Amazon Bedrock Managed Knowledge Bases
for retrieval-augmented generation (RAG) in ADK agents.

Environment Variables:
    KNOWLEDGE_BASE_ID: Bedrock Knowledge Base ID (required if not passed as arg)
    AWS_REGION: AWS region (default: us-east-1)
    USE_AGENTIC_RETRIEVAL: Enable agentic multi-hop retrieval (default: true)

Dependencies:
    pip install boto3>=1.43.2

Usage:
    from google.adk_community.tools.bedrock_kb import bedrock_kb_retrieve
    from google.adk.agents import Agent

    agent = Agent(
        model="gemini-2.0-flash",
        tools=[bedrock_kb_retrieve],
        instruction="Use bedrock_kb_retrieve to answer questions from the knowledge base.",
    )
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _get_source_uri(result: dict) -> str:
    """Extract source URI from a retrieval result."""
    location = result.get("location", {})
    if "s3Location" in location:
        return location["s3Location"].get("uri", "")
    if "webLocation" in location:
        return location["webLocation"].get("url", "")
    if "confluenceLocation" in location:
        return location["confluenceLocation"].get("url", "")
    if "salesforceLocation" in location:
        return location["salesforceLocation"].get("url", "")
    if "sharePointLocation" in location:
        return location["sharePointLocation"].get("url", "")
    if "customDocumentLocation" in location:
        return location["customDocumentLocation"].get("id", "")
    return ""


def _get_client():
    """Create boto3 bedrock-agent-runtime client with user-agent."""
    import boto3
    from botocore.config import Config

    region = os.environ.get("AWS_REGION", "us-east-1")
    return boto3.client(
        "bedrock-agent-runtime",
        region_name=region,
        config=Config(user_agent_extra="google-adk/bedrock-kb"),
    )


def _agentic_retrieve(client, knowledge_base_id: str, query: str, max_results: int) -> list[dict]:
    """Use AgenticRetrieveStream for multi-hop reasoning retrieval."""
    response = client.agentic_retrieve_stream(
        messages=[{"content": {"text": query}, "role": "user"}],
        retrievers=[{
            "configuration": {
                "knowledgeBase": {
                    "knowledgeBaseId": knowledge_base_id,
                    "retrievalOverrides": {"maxNumberOfResults": max_results},
                }
            }
        }],
        agenticRetrieveConfiguration={
            "foundationModelType": "MANAGED",
            "rerankingModelType": "MANAGED",
        },
    )
    results = []
    for event in response.get("stream", []):
        if "result" in event and "results" in event["result"]:
            for r in event["result"]["results"]:
                results.append({
                    "content": r.get("content", {}).get("text", ""),
                    "source": _get_source_uri(r),
                    "score": r.get("score", 0),
                })
    return results


def _standard_retrieve(client, knowledge_base_id: str, query: str, max_results: int) -> list[dict]:
    """Use standard Retrieve API."""
    response = client.retrieve(
        knowledgeBaseId=knowledge_base_id,
        retrievalQuery={"text": query},
        retrievalConfiguration={"managedSearchConfiguration": {"numberOfResults": max_results}},
    )
    results = []
    for r in response.get("retrievalResults", []):
        results.append({
            "content": r.get("content", {}).get("text", ""),
            "source": _get_source_uri(r),
            "score": r.get("score", 0),
        })
    return results


def bedrock_kb_retrieve(
    query: str,
    knowledge_base_id: Optional[str] = None,
    max_results: int = 5,
) -> str:
    """Retrieve relevant documents from an Amazon Bedrock Knowledge Base.

    Searches the knowledge base using either agentic retrieval (multi-hop
    reasoning with managed reranking) or standard semantic search, and returns
    formatted results with content, source, and relevance score.

    Args:
        query: The natural language question or search query.
        knowledge_base_id: Bedrock Knowledge Base ID. Defaults to KNOWLEDGE_BASE_ID env var.
        max_results: Maximum number of results to return. Defaults to 5.

    Returns:
        Formatted string with retrieved passages, sources, and scores.
    """
    kb_id = knowledge_base_id or os.environ.get("KNOWLEDGE_BASE_ID", "")
    if not kb_id:
        return "Error: No knowledge_base_id provided. Set KNOWLEDGE_BASE_ID environment variable or pass it as an argument."

    use_agentic = os.environ.get("USE_AGENTIC_RETRIEVAL", "true").lower() != "false"
    client = _get_client()

    results = []
    if use_agentic:
        try:
            results = _agentic_retrieve(client, kb_id, query, max_results)
        except Exception as e:
            logger.debug("Agentic retrieval failed, falling back to standard: %s", e)
            results = _standard_retrieve(client, kb_id, query, max_results)
    else:
        results = _standard_retrieve(client, kb_id, query, max_results)

    if not results:
        return "No relevant documents found in the knowledge base."

    formatted = []
    for i, r in enumerate(results[:max_results], 1):
        formatted.append(
            f"[{i}] (Score: {r['score']:.2f}) Source: {r['source']}\n{r['content']}"
        )
    return "\n\n---\n\n".join(formatted)
