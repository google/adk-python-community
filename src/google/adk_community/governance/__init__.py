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

"""Governance plugin for ADK agents.

Provides runtime governance — policy-based tool filtering, delegation scope
enforcement, and structured audit trails — without modifying agent logic.
"""

from .governance_plugin import AuditAction
from .governance_plugin import AuditEvent
from .governance_plugin import AuditHandler
from .governance_plugin import Decision
from .governance_plugin import DelegationScope
from .governance_plugin import GovernancePlugin
from .governance_plugin import LoggingAuditHandler
from .governance_plugin import PolicyDecision
from .governance_plugin import PolicyEvaluator
from .governance_plugin import ToolPolicy

__all__ = [
    "AuditAction",
    "AuditEvent",
    "AuditHandler",
    "Decision",
    "DelegationScope",
    "GovernancePlugin",
    "LoggingAuditHandler",
    "PolicyDecision",
    "PolicyEvaluator",
    "ToolPolicy",
]
