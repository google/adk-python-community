# Copyright 2026 SidClaw
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""SidClaw governance integration for Google ADK agents.

Provides policy evaluation, human-in-the-loop approval, and
tamper-proof audit trails for ADK agent tool calls via the
SidClaw platform.

Usage::

    from google.adk_community.governance import (
        SidClawGovernanceService,
    )
"""

from .sidclaw_governance import SidClawGovernanceConfig
from .sidclaw_governance import SidClawGovernanceService

__all__ = [
    "SidClawGovernanceConfig",
    "SidClawGovernanceService",
]
