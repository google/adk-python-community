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

"""Tests for Bedrock Knowledge Base tool."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestBedrockKBRetrieve:
    """Tests for bedrock_kb_retrieve function."""

    def test_retrieve_returns_formatted_results(self):
        from google.adk_community.tools.bedrock_kb.bedrock_kb_tool import bedrock_kb_retrieve

        mock_client = MagicMock()
        mock_client.retrieve.return_value = {
            "retrievalResults": [
                {
                    "content": {"text": "Amazon Bedrock is a managed service."},
                    "score": 0.95,
                    "location": {"s3Location": {"uri": "s3://bucket/doc.pdf"}},
                },
                {
                    "content": {"text": "Knowledge bases provide RAG capabilities."},
                    "score": 0.82,
                    "location": {"webLocation": {"url": "https://docs.aws.amazon.com/bedrock"}},
                },
            ]
        }

        with patch("google.adk_community.tools.bedrock_kb.bedrock_kb_tool._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"USE_AGENTIC_RETRIEVAL": "false"}):
                result = bedrock_kb_retrieve("What is Bedrock?", knowledge_base_id="TEST_KB")

        assert "Amazon Bedrock is a managed service" in result
        assert "Knowledge bases provide RAG capabilities" in result
        assert "s3://bucket/doc.pdf" in result
        assert "0.95" in result

    def test_agentic_retrieve_with_fallback(self):
        from google.adk_community.tools.bedrock_kb.bedrock_kb_tool import bedrock_kb_retrieve

        mock_client = MagicMock()
        # Agentic fails
        mock_client.agentic_retrieve_stream.side_effect = Exception("Not supported")
        # Fallback to standard
        mock_client.retrieve.return_value = {
            "retrievalResults": [
                {"content": {"text": "Fallback result."}, "score": 0.7, "location": {}},
            ]
        }

        with patch("google.adk_community.tools.bedrock_kb.bedrock_kb_tool._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"USE_AGENTIC_RETRIEVAL": "true"}):
                result = bedrock_kb_retrieve("test query", knowledge_base_id="TEST_KB")

        assert "Fallback result" in result
        mock_client.agentic_retrieve_stream.assert_called_once()
        mock_client.retrieve.assert_called_once()

    def test_no_kb_id_returns_error(self):
        from google.adk_community.tools.bedrock_kb.bedrock_kb_tool import bedrock_kb_retrieve

        with patch.dict(os.environ, {}, clear=True):
            result = bedrock_kb_retrieve("test", knowledge_base_id="")

        assert "Error" in result
        assert "KNOWLEDGE_BASE_ID" in result

    def test_no_results_returns_message(self):
        from google.adk_community.tools.bedrock_kb.bedrock_kb_tool import bedrock_kb_retrieve

        mock_client = MagicMock()
        mock_client.retrieve.return_value = {"retrievalResults": []}

        with patch("google.adk_community.tools.bedrock_kb.bedrock_kb_tool._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"USE_AGENTIC_RETRIEVAL": "false"}):
                result = bedrock_kb_retrieve("unknown topic", knowledge_base_id="TEST_KB")

        assert "No relevant documents found" in result

    def test_agentic_retrieve_success(self):
        from google.adk_community.tools.bedrock_kb.bedrock_kb_tool import bedrock_kb_retrieve

        mock_client = MagicMock()
        mock_client.agentic_retrieve_stream.return_value = {
            "stream": [
                {
                    "result": {
                        "results": [
                            {"content": {"text": "Agentic result."}, "score": 0.9, "location": {"s3Location": {"uri": "s3://b/k"}}},
                        ]
                    }
                }
            ]
        }

        with patch("google.adk_community.tools.bedrock_kb.bedrock_kb_tool._get_client", return_value=mock_client):
            with patch.dict(os.environ, {"USE_AGENTIC_RETRIEVAL": "true"}):
                result = bedrock_kb_retrieve("test", knowledge_base_id="TEST_KB")

        assert "Agentic result" in result
        mock_client.agentic_retrieve_stream.assert_called_once()
        mock_client.retrieve.assert_not_called()

    def test_user_agent_is_set(self):
        from google.adk_community.tools.bedrock_kb.bedrock_kb_tool import _get_client

        with patch("boto3.client") as mock_boto:
            _get_client()
            call_kwargs = mock_boto.call_args
            config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
            assert "google-adk/bedrock-kb" in config.user_agent_extra
