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

"""Manual E2E runner for the Azure OpenAI Responses API community model.

Run from this directory with:
  python main.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time

import agent
from dotenv import load_dotenv
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.cli.utils import logs
from google.adk.models.llm_request import LlmRequest
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types

_ENV_PATH = Path(__file__).parent / '.env'
load_dotenv(_ENV_PATH, override=True)
logs.log_to_tmp_folder(level=logging.INFO)

APP_NAME = 'azure_openai_responses_manual_e2e'
USER_ID = 'manual_e2e_user'


@dataclass
class AgentRunResult:
  text: str
  function_calls: list[str]
  function_responses: list[str]
  interaction_id: str | None


def _require_environment() -> None:
  missing = [
      name
      for name in (
          'AZURE_OPENAI_API_KEY',
          'AZURE_OPENAI_ENDPOINT',
          'AZURE_OPENAI_RESPONSES_DEPLOYMENT',
      )
      if not os.environ.get(name)
  ]
  if missing:
    raise RuntimeError(
        'Missing required environment variables: '
        + ', '.join(missing)
        + '. See README.md for setup instructions.'
    )


async def _run_prompt(
    runner: Runner, session: Session, prompt: str
) -> AgentRunResult:
  content = types.Content(
      role='user', parts=[types.Part.from_text(text=prompt)]
  )
  final_text = ''
  function_calls = []
  function_responses = []
  interaction_id = None

  print(f'\n>> User: {prompt}')
  async for event in runner.run_async(
      user_id=USER_ID,
      session_id=session.id,
      new_message=content,
  ):
    if event.interaction_id:
      interaction_id = event.interaction_id

    for function_call in event.get_function_calls():
      call_log = f'{function_call.name}({function_call.args})'
      function_calls.append(call_log)
      print(f'   [Tool Call] {call_log}')

    for function_response in event.get_function_responses():
      response_log = f'{function_response.name}: {function_response.response}'
      function_responses.append(response_log)
      print(f'   [Tool Result] {response_log}')

    if event.content and event.content.parts and not event.partial:
      for part in event.content.parts:
        if part.text and event.author != 'user':
          final_text += part.text

  print(f'<< Agent: {final_text}')
  if interaction_id:
    print(f'   [Interaction ID: {interaction_id}]')

  return AgentRunResult(
      text=final_text,
      function_calls=function_calls,
      function_responses=function_responses,
      interaction_id=interaction_id,
  )


async def _test_basic_text(runner: Runner, session: Session) -> None:
  print('\n' + '=' * 72)
  print('TEST 1: Runner Basic Text Generation')
  print('=' * 72)

  result = await _run_prompt(
      runner,
      session,
      'Reply with exactly: AZURE_RESPONSES_TEXT_OK',
  )

  assert 'AZURE_RESPONSES_TEXT_OK' in result.text, result.text
  print('PASSED: Runner basic text generation works')


async def _test_function_calling(runner: Runner, session: Session) -> None:
  print('\n' + '=' * 72)
  print('TEST 2: Runner Function Calling')
  print('=' * 72)

  result = await _run_prompt(
      runner,
      session,
      'Use the weather tool for Tokyo and report the result.',
  )

  assert result.function_calls, 'Expected at least one function call.'
  assert any('get_current_weather' in call for call in result.function_calls)
  assert any('get_current_weather' in res for res in result.function_responses)
  assert 'tokyo' in result.text.lower() or '68' in result.text, result.text
  print('PASSED: Runner function calling works')


async def _test_multi_turn_context(runner: Runner, session: Session) -> None:
  print('\n' + '=' * 72)
  print('TEST 3: Runner Multi-Turn Context')
  print('=' * 72)

  first = await _run_prompt(
      runner,
      session,
      'Remember this manual E2E code phrase: cobalt otter. Acknowledge only.',
  )
  second = await _run_prompt(
      runner,
      session,
      'What manual E2E code phrase did I ask you to remember?',
  )

  assert first.text, 'Expected first turn response.'
  assert 'cobalt' in second.text.lower(), second.text
  assert 'otter' in second.text.lower(), second.text
  print('PASSED: Runner multi-turn context works')


async def _test_previous_response_id_direct_model() -> None:
  print('\n' + '=' * 72)
  print('TEST 4: Direct Responses previous_response_id Chaining')
  print('=' * 72)

  model = agent.get_azure_openai_responses_model()
  first_request = LlmRequest(
      contents=[
          types.Content(
              role='user',
              parts=[
                  types.Part.from_text(
                      text=(
                          'Remember this direct Responses code phrase: amber'
                          ' swan. Reply with exactly: DIRECT_CHAIN_READY'
                      )
                  )
              ],
          )
      ]
  )
  first_response = None
  async for response in model.generate_content_async(first_request):
    first_response = response

  assert first_response is not None, 'Expected first direct model response.'
  assert first_response.interaction_id, 'Expected a Responses API response id.'
  print(f'   First response id: {first_response.interaction_id}')

  second_request = LlmRequest(
      previous_interaction_id=first_response.interaction_id,
      contents=[
          types.Content(
              role='user',
              parts=[
                  types.Part.from_text(
                      text=(
                          'Using the previous response context, what direct'
                          ' Responses code phrase did I ask you to remember?'
                      )
                  )
              ],
          )
      ],
  )
  second_text = ''
  second_interaction_id = None
  async for response in model.generate_content_async(second_request):
    second_interaction_id = response.interaction_id
    if response.content and response.content.parts:
      second_text += ''.join(part.text or '' for part in response.content.parts)

  print(f'<< Direct model: {second_text}')
  if second_interaction_id:
    print(f'   Second response id: {second_interaction_id}')

  assert 'amber' in second_text.lower(), second_text
  assert 'swan' in second_text.lower(), second_text
  print('PASSED: Direct previous_response_id chaining works')


async def main() -> None:
  _require_environment()
  session_service = InMemorySessionService()
  artifact_service = InMemoryArtifactService()
  runner = Runner(
      app_name=APP_NAME,
      agent=agent.root_agent,
      artifact_service=artifact_service,
      session_service=session_service,
  )
  session = await session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID
  )

  print('Azure OpenAI Responses manual E2E')
  print(f'Endpoint: {os.environ["AZURE_OPENAI_ENDPOINT"]}')
  print(f'Deployment: {os.environ["AZURE_OPENAI_RESPONSES_DEPLOYMENT"]}')

  start_time = time.time()
  await _test_basic_text(runner, session)
  await _test_function_calling(runner, session)
  await _test_multi_turn_context(runner, session)
  await _test_previous_response_id_direct_model()
  end_time = time.time()

  print('\n' + '=' * 72)
  print('ALL MANUAL E2E TESTS PASSED')
  print(f'Total time: {end_time - start_time:.2f}s')


if __name__ == '__main__':
  asyncio.run(main())
