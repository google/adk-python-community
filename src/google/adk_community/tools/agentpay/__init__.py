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

"""AgentPay x402 payment tools for Google ADK.

Enables ADK agents to make autonomous HTTP payments using the x402 protocol
(HTTP 402 Payment Required) via the agentpay-mcp MCP server.

See: https://www.npmjs.com/package/agentpay-mcp
"""

from .agentpay_tools import check_spend_limit
from .agentpay_tools import fetch_paid_api
from .agentpay_tools import get_wallet_info
from .agentpay_tools import send_payment

__all__ = [
    "fetch_paid_api",
    "get_wallet_info",
    "check_spend_limit",
    "send_payment",
]
