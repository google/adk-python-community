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

"""Runs the shop agent against each demo checkout from the console.

Usage: python main.py [cs_clean|cs_tip|cs_addon] [--yes|--no]

Needs a Gemini API key (GOOGLE_API_KEY). When the guard asks the shopper,
the answer comes from --yes or --no, or from the keyboard.
"""

from __future__ import annotations

import asyncio
import sys

from google.adk.runners import InMemoryRunner
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.genai import types
from shop_agent.agent import app
from shop_agent.agent import ORDERS


async def main(checkout_id: str, answer: str | None) -> None:
  runner = InMemoryRunner(app=app)
  session = await runner.session_service.create_session(
      app_name=app.name, user_id="shopper"
  )
  message = types.Content(
      role="user",
      parts=[
          types.Part(text=f"Please buy the grinder in checkout {checkout_id}.")
      ],
  )
  while message is not None:
    pending = None
    async for event in runner.run_async(
        user_id="shopper", session_id=session.id, new_message=message
    ):
      for call in event.get_function_calls():
        print(f"[tool call] {call.name}({call.args})")
        if call.name == "adk_request_confirmation":
          pending = call
      for response in event.get_function_responses():
        print(f"[tool result] {response.name}: {response.response}")
      if event.content and event.content.parts:
        text = "".join(p.text or "" for p in event.content.parts)
        if text.strip():
          print(f"[{event.author}] {text.strip()}")
    message = None
    if pending is not None:
      print("\n" + pending.args["toolConfirmation"]["hint"] + "\n")
      reply = answer or input("Approve this payment? [y/N] ")
      confirmed = reply.strip().lower() in ("y", "yes", "--yes")
      print(f"[shopper] {'approved' if confirmed else 'rejected'}")
      message = types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id=pending.id,
                      name=pending.name,
                      response=ToolConfirmation(
                          confirmed=confirmed
                      ).model_dump(),
                  )
              )
          ],
      )
  print(f"\nOrders placed: {ORDERS or 'none'}")


if __name__ == "__main__":
  args = sys.argv[1:]
  flag = next((a for a in args if a in ("--yes", "--no")), None)
  ids = [a for a in args if not a.startswith("--")]
  asyncio.run(main(ids[0] if ids else "cs_tip", flag))
