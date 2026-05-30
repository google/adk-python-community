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

import sys
from types import ModuleType

# Pre-emptively mock/patch google.genai.types.AvatarConfig if it's missing or fails to import
try:
    import google.genai.types as genai_types
    if not hasattr(genai_types, "AvatarConfig"):
        from pydantic import BaseModel
        class AvatarConfig(BaseModel):
            pass
        genai_types.AvatarConfig = AvatarConfig
except Exception:
    try:
        sys.modules["google.genai"] = ModuleType("google.genai")
        
        from pydantic import BaseModel
        class AvatarConfig(BaseModel):
            pass
        genai_types = sys.modules["google.genai.types"] = ModuleType("google.genai.types")
        genai_types.AvatarConfig = AvatarConfig
    except Exception:
        pass
