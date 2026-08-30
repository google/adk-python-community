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

from unittest.mock import MagicMock

import pytest

from google.adk_community.sessions.mongo_session_service import MongoSessionService


def test_constructor_without_pymongo_names_mongo_extra(monkeypatch):
  real_import = __import__

  def _import(name, *args, **kwargs):
    if name == "pymongo":
      raise ModuleNotFoundError("No module named 'pymongo'")
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr("builtins.__import__", _import)
  with pytest.raises(ImportError, match=r"google-adk-community\[mongo\]"):
    MongoSessionService(client=MagicMock())
