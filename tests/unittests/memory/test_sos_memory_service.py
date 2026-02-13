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

"""Tests for SOS Memory Service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from google.adk_community.memory.sos_memory_service import (
    SOSMemoryService,
    SOSMemoryServiceConfig,
)


class TestSOSMemoryServiceConfig:
    """Tests for SOSMemoryServiceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SOSMemoryServiceConfig()

        assert config.search_top_k == 10
        assert config.timeout == 30.0
        assert config.user_content_salience == 0.8
        assert config.model_content_salience == 0.7
        assert config.default_salience == 0.6
        assert config.enable_user_filtering is True
        assert config.enable_lineage_tracking is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SOSMemoryServiceConfig(
            search_top_k=20,
            timeout=60.0,
            user_content_salience=0.9,
            enable_lineage_tracking=False,
        )

        assert config.search_top_k == 20
        assert config.timeout == 60.0
        assert config.user_content_salience == 0.9
        assert config.enable_lineage_tracking is False


class TestSOSMemoryService:
    """Tests for SOSMemoryService."""

    def test_init_requires_api_key(self):
        """Test that initialization requires an API key."""
        with pytest.raises(ValueError) as exc_info:
            SOSMemoryService(api_key="")

        assert "api_key is required" in str(exc_info.value)

    def test_init_with_api_key(self):
        """Test successful initialization with API key."""
        service = SOSMemoryService(
            base_url="http://localhost:8844",
            api_key="test-key",
            agent_id="test-agent",
        )

        assert service._base_url == "http://localhost:8844"
        assert service._api_key == "test-key"
        assert service._agent_id == "test-agent"

    def test_init_strips_trailing_slash(self):
        """Test that trailing slashes are stripped from base_url."""
        service = SOSMemoryService(
            base_url="http://localhost:8844/",
            api_key="test-key",
        )

        assert service._base_url == "http://localhost:8844"

    def test_compute_lineage_hash(self):
        """Test lineage hash computation."""
        service = SOSMemoryService(api_key="test-key", agent_id="test-agent")

        hash1 = service._compute_lineage_hash("content1", "context1")
        hash2 = service._compute_lineage_hash("content2", "context2")

        # Hashes should be 16 characters
        assert len(hash1) == 16
        assert len(hash2) == 16

        # Different content should produce different hashes
        assert hash1 != hash2

    def test_lineage_chain_grows(self):
        """Test that lineage chain grows when hashes are appended."""
        service = SOSMemoryService(api_key="test-key", agent_id="test-agent")

        assert len(service._lineage_chain) == 0

        # Compute and manually append (as _prepare_memory_data does)
        hash1 = service._compute_lineage_hash("content1", "")
        service._lineage_chain.append(hash1)
        assert len(service._lineage_chain) == 1

        hash2 = service._compute_lineage_hash("content2", "")
        service._lineage_chain.append(hash2)
        assert len(service._lineage_chain) == 2

        # Verify hashes are different
        assert hash1 != hash2

    def test_determine_salience_user(self):
        """Test salience determination for user content."""
        service = SOSMemoryService(api_key="test-key")

        assert service._determine_salience("user") == 0.8
        assert service._determine_salience("User") == 0.8
        assert service._determine_salience("USER") == 0.8

    def test_determine_salience_model(self):
        """Test salience determination for model content."""
        service = SOSMemoryService(api_key="test-key")

        assert service._determine_salience("model") == 0.7
        assert service._determine_salience("Model") == 0.7

    def test_determine_salience_default(self):
        """Test default salience for unknown authors."""
        service = SOSMemoryService(api_key="test-key")

        assert service._determine_salience(None) == 0.6
        assert service._determine_salience("unknown") == 0.6

    def test_build_search_payload(self):
        """Test search payload construction."""
        service = SOSMemoryService(
            api_key="test-key",
            agent_id="test-agent",
        )

        payload = service._build_search_payload(
            app_name="test-app",
            user_id="user-123",
            query="test query",
        )

        assert payload["query"] == "test query"
        assert payload["agent"] == "test-agent"
        assert payload["limit"] == 10
        assert payload["filters"]["user_id"] == "user-123"
        assert payload["filters"]["app_name"] == "test-app"

    def test_build_search_payload_no_filtering(self):
        """Test search payload without user filtering."""
        config = SOSMemoryServiceConfig(enable_user_filtering=False)
        service = SOSMemoryService(
            api_key="test-key",
            agent_id="test-agent",
            config=config,
        )

        payload = service._build_search_payload(
            app_name="test-app",
            user_id="user-123",
            query="test query",
        )

        assert payload["filters"] == {}

    def test_convert_to_memory_entry_simple(self):
        """Test converting simple result to MemoryEntry."""
        service = SOSMemoryService(api_key="test-key")

        result = {"text": "Hello world", "metadata": {}}
        entry = service._convert_to_memory_entry(result)

        assert entry is not None
        assert entry.content.parts[0].text == "Hello world"

    def test_convert_to_memory_entry_with_prefix(self):
        """Test converting result with metadata prefix to MemoryEntry."""
        service = SOSMemoryService(api_key="test-key")

        result = {
            "text": "[Author: user, Time: 2025-01-15T10:00:00] Hello world",
            "metadata": {},
        }
        entry = service._convert_to_memory_entry(result)

        assert entry is not None
        assert entry.content.parts[0].text == "Hello world"
        assert entry.author == "user"
        assert entry.timestamp == "2025-01-15T10:00:00"

    def test_convert_to_memory_entry_from_metadata(self):
        """Test converting result with metadata dict to MemoryEntry."""
        service = SOSMemoryService(api_key="test-key")

        result = {
            "text": "Hello world",
            "metadata": {
                "author": "model",
                "timestamp": "2025-01-15T11:00:00",
            },
        }
        entry = service._convert_to_memory_entry(result)

        assert entry is not None
        assert entry.author == "model"
        assert entry.timestamp == "2025-01-15T11:00:00"

    @pytest.mark.asyncio
    async def test_get_lineage(self):
        """Test getting lineage information."""
        service = SOSMemoryService(api_key="test-key", agent_id="test-agent")

        # Empty lineage
        lineage = await service.get_lineage()
        assert lineage["agent_id"] == "test-agent"
        assert lineage["chain_length"] == 0
        assert lineage["latest_hash"] is None

        # Add some hashes (simulating what _prepare_memory_data does)
        hash1 = service._compute_lineage_hash("content1", "")
        service._lineage_chain.append(hash1)
        hash2 = service._compute_lineage_hash("content2", "")
        service._lineage_chain.append(hash2)

        lineage = await service.get_lineage()
        assert lineage["chain_length"] == 2
        assert lineage["latest_hash"] == hash2
        assert len(lineage["chain"]) == 2

    @pytest.mark.asyncio
    async def test_close_clears_lineage(self):
        """Test that close() clears the lineage chain."""
        service = SOSMemoryService(api_key="test-key")

        hash1 = service._compute_lineage_hash("content", "")
        service._lineage_chain.append(hash1)
        assert len(service._lineage_chain) == 1

        await service.close()
        assert len(service._lineage_chain) == 0

    @pytest.mark.asyncio
    async def test_search_memory_success(self):
        """Test successful memory search."""
        service = SOSMemoryService(api_key="test-key", agent_id="test-agent")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"text": "Memory 1", "metadata": {}},
                {"text": "Memory 2", "metadata": {}},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            response = await service.search_memory(
                app_name="test-app",
                user_id="user-123",
                query="test query",
            )

            assert len(response.memories) == 2

    @pytest.mark.asyncio
    async def test_search_memory_handles_list_response(self):
        """Test search handles list response format."""
        service = SOSMemoryService(api_key="test-key", agent_id="test-agent")

        mock_response = MagicMock()
        # Some APIs return a list directly instead of {"results": [...]}
        mock_response.json.return_value = [
            {"text": "Memory 1", "metadata": {}},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            response = await service.search_memory(
                app_name="test-app",
                user_id="user-123",
                query="test query",
            )

            assert len(response.memories) == 1

    @pytest.mark.asyncio
    async def test_search_memory_handles_error(self):
        """Test search handles HTTP errors gracefully."""
        service = SOSMemoryService(api_key="test-key", agent_id="test-agent")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            response = await service.search_memory(
                app_name="test-app",
                user_id="user-123",
                query="test query",
            )

            # Should return empty response on error, not raise
            assert len(response.memories) == 0
