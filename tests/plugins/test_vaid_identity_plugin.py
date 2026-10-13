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

"""Tests for VaidIdentityPlugin.

VaidIdentityPlugin is a thin delegate over vaid_adk.VaidAuthPlugin, so these
tests exercise it the same way vaid-adk's own test_plugin.py does: minimal
duck-typed stand-ins for ADK's Agent/Tool/ToolContext, and real VAID
documents minted in-test with an Ed25519 key (cheap, no network). The
ImportError path is tested separately by faking the absence of vaid_adk in
sys.modules.
"""

from __future__ import annotations

import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest
from vaid_adk import ToolAuthorization
from vaid_adk import TrustedIssuers
from vaid_adk import VaidPresentation
from vaid_mint.document import build_unsigned_vaid_document
from vaid_mint.document import canonical_vaid_signing_bytes
from vaid_mint.document import compute_lineage_hash
from vaid_mint.document import prefix_entry
from vaid_mint.issuer_identity import kernel_key_thumbprint
from vaid_mint.revocation import InMemoryRevocationList

from google.adk_community.plugins.vaid_identity_plugin import VaidIdentityPlugin

_KEY = Ed25519PrivateKey.from_private_bytes(bytes([0x33]) * 32)
_PUBLIC = _KEY.public_key().public_bytes_raw()
_THUMBPRINT = kernel_key_thumbprint(_PUBLIC)


class _Agent:

  def __init__(self, name: str) -> None:
    self.name = name


class _Tool:

  def __init__(self, name: str) -> None:
    self.name = name


class _ToolContext:

  def __init__(self, agent_name: str) -> None:
    self.agent_name = agent_name


def _root_vaid(scope: list[str], caps: list[str]) -> dict:
  agent_id = str(uuid.uuid4())
  unsigned = build_unsigned_vaid_document(
      vaid_id=agent_id,
      agent_id=agent_id,
      agent_class="test",
      version="1.0.0",
      tenant_id="acme",
      issued_at="2026-06-01T00:00:00Z",
      expires_at="2999-01-01T00:00:00Z",
      public_key_der=list(range(32)),
      parent_vaid=None,
      scope_boundary=[prefix_entry(v) for v in scope],
      lineage_hash=compute_lineage_hash(None, agent_id),
      capability_set=caps,
      trust_domain="vaid.example",
      kernel_key_thumbprint=_THUMBPRINT,
  )
  signature = _KEY.sign(canonical_vaid_signing_bytes(unsigned))
  return {**unsigned, "kernel_signature": list(signature)}


def _plugin(
    tool_authorizations: dict[str, ToolAuthorization] | None = None,
) -> VaidIdentityPlugin:
  return VaidIdentityPlugin(
      keys=TrustedIssuers.from_raw_keys([("vaid.example", _PUBLIC)]),
      revocation=InMemoryRevocationList.assume_nothing_revoked(),
      tool_authorizations=tool_authorizations or {},
  )


@pytest.mark.asyncio
async def test_plugin_name() -> None:
  plugin = _plugin()
  assert plugin.name == "vaid_identity"


@pytest.mark.asyncio
async def test_before_agent_callback_allows_registered_identity() -> None:
  plugin = _plugin()
  plugin.register(
      "worker", VaidPresentation(leaf=_root_vaid(["data.x"], ["read"]))
  )
  result = await plugin.before_agent_callback(
      agent=_Agent("worker"), callback_context=None
  )
  assert result is None


@pytest.mark.asyncio
async def test_before_agent_callback_denies_unregistered_agent() -> None:
  plugin = _plugin()
  result = await plugin.before_agent_callback(
      agent=_Agent("ghost"), callback_context=None
  )
  assert result is not None
  assert "ghost" in result.parts[0].text


@pytest.mark.asyncio
async def test_before_tool_callback_allows_in_scope_call() -> None:
  plugin = _plugin({
      "lookup_order": ToolAuthorization(
          capability="read", resource="data.orders"
      )
  })
  plugin.register(
      "worker",
      VaidPresentation(leaf=_root_vaid(["data.orders"], ["read"])),
  )
  result = await plugin.before_tool_callback(
      tool=_Tool("lookup_order"),
      tool_args={},
      tool_context=_ToolContext("worker"),
  )
  assert result is None


@pytest.mark.asyncio
async def test_before_tool_callback_denies_out_of_scope_call() -> None:
  # Capability is authorized; the resource is outside the leaf's
  # scope_boundary. Denied before the tool runs — before_tool_callback
  # returns a dict, which ADK uses as the tool's function response directly,
  # so the tool itself is never called.
  plugin = _plugin({
      "issue_refund": ToolAuthorization(
          capability="read", resource="data.payments"
      )
  })
  plugin.register(
      "worker",
      VaidPresentation(leaf=_root_vaid(["data.orders"], ["read"])),
  )
  result = await plugin.before_tool_callback(
      tool=_Tool("issue_refund"),
      tool_args={},
      tool_context=_ToolContext("worker"),
  )
  assert result is not None
  assert result["code"] == "out_of_scope"


@pytest.mark.asyncio
async def test_before_tool_callback_denies_unregistered_agent() -> None:
  # "worker" never registered: no VAID at all, denied rather than exempt.
  plugin = _plugin({
      "lookup_order": ToolAuthorization(
          capability="read", resource="data.orders"
      )
  })
  result = await plugin.before_tool_callback(
      tool=_Tool("lookup_order"),
      tool_args={},
      tool_context=_ToolContext("worker"),
  )
  assert result is not None
  assert result["code"] == "no_vaid"


@pytest.mark.asyncio
async def test_before_tool_callback_denies_unmapped_tool_never_default_allow() -> (
    None
):
  plugin = _plugin(tool_authorizations={})  # nothing mapped
  plugin.register(
      "worker", VaidPresentation(leaf=_root_vaid(["data.x"], ["read"]))
  )
  result = await plugin.before_tool_callback(
      tool=_Tool("anything"), tool_args={}, tool_context=_ToolContext("worker")
  )
  assert result is not None
  assert result["code"] == "unmapped_tool"


def test_raises_import_error_when_vaid_adk_missing(monkeypatch) -> None:
  import builtins
  import importlib
  import sys

  real_import = builtins.__import__

  def _fake_import(name, *args, **kwargs):
    if name == "vaid_adk" or name.startswith("vaid_adk."):
      raise ImportError("no module named vaid_adk")
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", _fake_import)
  for mod_name in list(sys.modules):
    if mod_name == "vaid_adk" or mod_name.startswith("vaid_adk."):
      monkeypatch.delitem(sys.modules, mod_name, raising=False)
  monkeypatch.delitem(
      sys.modules,
      "google.adk_community.plugins.vaid_identity_plugin",
      raising=False,
  )

  with pytest.raises(ImportError):
    importlib.import_module("google.adk_community.plugins.vaid_identity_plugin")
