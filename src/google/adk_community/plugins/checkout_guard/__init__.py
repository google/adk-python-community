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

"""Pre-payment checkout checks for shopping agents (ACP and UCP)."""

from ._check import check
from ._check import digest
from ._check import verify_unchanged
from ._models import Checkout
from ._models import Fee
from ._models import Finding
from ._models import Line
from ._models import MandateItem
from ._models import Recurring
from ._models import ShopperMandate
from ._models import TextBlock
from ._models import Verdict
from ._rules import Rule
from ._rules import RULES
from .adapters import from_acp
from .adapters import from_ucp
from .plugin import CheckoutGuardPlugin
from .plugin import DEFAULT_MANDATE_STATE_KEY

__all__ = [
    "Checkout",
    "CheckoutGuardPlugin",
    "DEFAULT_MANDATE_STATE_KEY",
    "Fee",
    "Finding",
    "Line",
    "MandateItem",
    "RULES",
    "Recurring",
    "Rule",
    "ShopperMandate",
    "TextBlock",
    "Verdict",
    "check",
    "digest",
    "from_acp",
    "from_ucp",
    "verify_unchanged",
]
