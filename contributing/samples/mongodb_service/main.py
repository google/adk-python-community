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

"""Example of using MongoDB for Session Service."""

import os
import asyncio

from google.adk.runners import Runner
from google.genai import types
from mongo_service_agent import root_agent

from google.adk_community.sessions import MongoSessionService

APP_NAME = "financial_advisor_agent"
USER_ID = "juante_jc_11"
SESSION_ID = "session_07"


async def main():
  """Main function to run the agent asynchronously."""
  
  #   You can create the MongoSessionService in two ways:
  #   1. With an existing AsyncMongoClient instance
  #   2. By providing a connection string directly
  #   from pymongo import AsyncMongoClient
  #   client = AsyncMongoClient(host="localhost", port=27017)
  #   session_service = MongoSessionService(client=client)

  connection_string = os.environ.get("MONGODB_URI")
  if not connection_string:
    raise ValueError("MONGODB_URI environment variable not set. See README.md for setup.")
  session_service = MongoSessionService(connection_string=connection_string)
  await session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
  )

  runner = Runner(
      agent=root_agent, app_name=APP_NAME, session_service=session_service
  )

  query = (
      "What is the status of my university invoice? Also, calculate the tax for"
      " a service amount of 9500 MXN."
  )
  print(f"User Query -> {query}")
  content = types.Content(role="user", parts=[types.Part(text=query)])

  async for event in runner.run_async(
      user_id=USER_ID,
      session_id=SESSION_ID,
      new_message=content,
  ):
    if event.is_final_response():
      print(f"Agent Response -> {event.content.parts[0].text}")


if __name__ == "__main__":
  asyncio.run(main())
