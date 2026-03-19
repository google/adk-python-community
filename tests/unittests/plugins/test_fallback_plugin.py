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

import gc
import pytest
from unittest.mock import MagicMock

from google.adk_community.plugins.fallback_plugin import FallbackPlugin

class TestFallbackPlugin:
    """Test cases for FallbackPlugin."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        plugin = FallbackPlugin()
        assert plugin.root_model is None
        assert plugin.fallback_model is None
        assert plugin.error_status == [429, 504]
        assert plugin._error_status_set == {"429", "504"}
        assert plugin._fallback_attempts == {}

    def test_init_custom(self):
        """Test initialization with custom values."""
        plugin = FallbackPlugin(
            root_model="gemini-2.0-flash",
            fallback_model="gemini-1.5-pro",
            error_status=[400, 500],
        )
        assert plugin.root_model == "gemini-2.0-flash"
        assert plugin.fallback_model == "gemini-1.5-pro"
        assert plugin.error_status == [400, 500]
        assert plugin._error_status_set == {"400", "500"}

    @pytest.mark.asyncio
    async def test_before_model_callback_initializes_context(self):
        """Test that before_model_callback initializes context in fallback attempts dict."""
        plugin = FallbackPlugin()
        mock_context = MagicMock()
        mock_request = MagicMock()

        await plugin.before_model_callback(
            callback_context=mock_context, llm_request=mock_request
        )

        assert mock_context in plugin._fallback_attempts
        assert plugin._fallback_attempts[mock_context] == 0

    @pytest.mark.asyncio
    async def test_before_model_callback_resets_model(self):
        """Test that before_model_callback resets model to root_model when attempt is 0."""
        plugin = FallbackPlugin(root_model="root-model")
        mock_context = MagicMock()
        mock_request = MagicMock(model="current-model")

        await plugin.before_model_callback(
            callback_context=mock_context, llm_request=mock_request
        )

        assert mock_request.model == "root-model"

    @pytest.mark.asyncio
    async def test_before_model_callback_no_reset_mid_fallback(self):
        """Test that before_model_callback does not reset model when attempt > 0."""
        plugin = FallbackPlugin(root_model="root-model")
        mock_context = MagicMock()
        mock_request = MagicMock(model="fallback-model")

        plugin._fallback_attempts[mock_context] = 1

        await plugin.before_model_callback(
            callback_context=mock_context, llm_request=mock_request
        )

        assert mock_request.model == "fallback-model"

    @pytest.mark.asyncio
    async def test_after_model_callback_annotates_on_error(self):
        """Test that after_model_callback annotates response on error status."""
        plugin = FallbackPlugin(root_model="root-model", fallback_model="fallback-model")
        mock_context = MagicMock()
        mock_response = MagicMock()
        mock_response.error_code = 429
        mock_response.error_message = "Rate limit"
        mock_response.custom_metadata = {}

        await plugin.after_model_callback(
            callback_context=mock_context, llm_response=mock_response
        )

        assert mock_response.custom_metadata["fallback_triggered"] is True
        assert mock_response.custom_metadata["original_model"] == "root-model"
        assert mock_response.custom_metadata["fallback_model"] == "fallback-model"
        assert mock_response.custom_metadata["fallback_attempt"] == 1
        assert mock_response.custom_metadata["error_code"] == "429"

    @pytest.mark.asyncio
    async def test_after_model_callback_no_annotate_on_non_error(self):
        """Test that after_model_callback does not annotate on success or non-configured error."""
        plugin = FallbackPlugin(root_model="root-model", fallback_model="fallback-model")
        mock_context = MagicMock()
        mock_response = MagicMock()
        mock_response.error_code = None
        mock_response.error_message = None
        mock_response.custom_metadata = {}

        await plugin.after_model_callback(
            callback_context=mock_context, llm_response=mock_response
        )

        assert "fallback_triggered" not in mock_response.custom_metadata

    @pytest.mark.asyncio
    async def test_after_model_callback_no_annotate_no_fallback_model(self):
        """Test that after_model_callback does not annotate when fallback_model is None."""
        plugin = FallbackPlugin(root_model="root-model")
        mock_context = MagicMock()
        mock_response = MagicMock()
        mock_response.error_code = 429
        mock_response.error_message = "Rate limit"
        mock_response.custom_metadata = {}

        await plugin.after_model_callback(
            callback_context=mock_context, llm_response=mock_response
        )

        assert "fallback_triggered" not in mock_response.custom_metadata

    @pytest.mark.asyncio
    async def test_after_model_callback_automatic_pruning(self):
        """Test that after_model_callback entries are automatically pruned when context is GC'd."""
        plugin = FallbackPlugin()
        
        class CustomContext:
            pass
            
        context = CustomContext()
        mock_response = MagicMock()
        mock_response.error_code = 429
        mock_response.error_message = "Rate limit"
        mock_response.custom_metadata = {}

        await plugin.after_model_callback(
            callback_context=context, llm_response=mock_response
        )

        assert context in plugin._fallback_attempts
        
        del context
        gc.collect() # Force GC
        
        assert len(plugin._fallback_attempts) == 0

    
