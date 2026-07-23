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

"""Runnable example for AuthorityRoutingPlugin.

Shows the plugin composing a model router with the deterministic guard without
needing a live model key: the router here is a tiny keyword classifier wrapped
in ``CallableRouter``. Swap it for a Gemini-backed router (sketch at the bottom)
to get real authority classification.

Run::

    python contributing/samples/authority_routing/main.py
"""

from __future__ import annotations

import asyncio

from google.adk_community.plugins import (
    AuthorityRoutingPlugin,
    CallableRouter,
    Posture,
    PostureVerdict,
)


def keyword_router(request: str, context: str) -> PostureVerdict:
    """A trivial stand-in router (no model call).

    A real deployment replaces this with a model that reads the request and the
    standing context. The deterministic guard tightens whatever this returns.
    """
    low = request.lower()
    if any(w in low for w in ("what would", "how would", "review", "suggest")):
        return PostureVerdict(Posture.ADVISE, "", "request reads as advisory")
    return PostureVerdict(Posture.EXECUTE, "the named target", "action requested")


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


class _Ctx:
    def __init__(self, state: dict) -> None:
        self.state = state


class _Session:
    def __init__(self, state: dict) -> None:
        self.state = state


class _Inv:
    def __init__(self, state: dict) -> None:
        self.session = _Session(state)


class _Part:
    def __init__(self, text: str) -> None:
        self.text = text


class _Content:
    def __init__(self, text: str) -> None:
        self.parts = [_Part(text)]


async def main() -> None:
    plugin = AuthorityRoutingPlugin(
        router=CallableRouter(keyword_router),
        read_only_tool_names={"read_file"},
    )

    cases = [
        ("Review the retry settings in config.yaml.", "edit_file"),
        ("Rename foo to bar in utils.py.", "edit_file"),
        ("Deploy the release to production.", "shell"),
    ]

    for request, tool_name in cases:
        state: dict = {}
        await plugin.on_user_message_callback(
            invocation_context=_Inv(state),
            user_message=_Content(request),
        )
        decision = state["authority:posture"]
        result = await plugin.before_tool_callback(
            tool=_Tool(tool_name),
            tool_args={},
            tool_context=_Ctx(state),
        )
        verdict = "ALLOWED" if result is None else "DENIED"
        print(f"[{decision['posture']:7}] {verdict:7} {tool_name:10} <- {request}")


# --- Sketch: a Gemini-backed router (requires google-genai + an API key) ------
#
# from google import genai
# from google.adk_community.plugins import PostureRouter, PostureVerdict, Posture
#
# class GeminiPostureRouter(PostureRouter):
#     def __init__(self, model: str = "gemini-2.5-flash") -> None:
#         self._client = genai.Client()
#         self._model = model
#
#     async def route(self, request: str, context: str) -> PostureVerdict:
#         resp = self._client.models.generate_content(
#             model=self._model,
#             contents=f"STANDING CONTEXT:\n{context}\n\nUSER REQUEST:\n{request}",
#             config={
#                 "system_instruction": (
#                     "Classify the authority of the request as ADVISE, EXECUTE, "
#                     "DEFER, or STOP. When torn, pick the stricter posture."
#                 ),
#                 "response_mime_type": "application/json",
#                 "response_schema": PostureVerdict,
#             },
#         )
#         parsed = resp.parsed
#         if parsed is None:  # fail closed on unparseable output
#             return PostureVerdict(Posture.DEFER, "", "router output unparseable")
#         return parsed


if __name__ == "__main__":
    asyncio.run(main())
