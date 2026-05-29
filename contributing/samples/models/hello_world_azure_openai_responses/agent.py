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

"""Azure OpenAI Responses API sample agent."""

from __future__ import annotations

import os

from google.adk.agents.llm_agent import Agent
from google.genai import types
from google.genai.types import GenerateContentConfig

from google.adk_community.models.openai_responses import AzureOpenAIResponsesLlm


def get_current_weather(city: str) -> dict:
  """Get deterministic weather for a city.

  Args:
    city: The city to look up.

  Returns:
    A dictionary containing weather information for the requested city.
  """
  weather_data = {
      'london': {'temperature_f': 59, 'condition': 'Cloudy'},
      'paris': {'temperature_f': 64, 'condition': 'Rainy'},
      'san francisco': {'temperature_f': 70, 'condition': 'Sunny'},
      'tokyo': {'temperature_f': 68, 'condition': 'Partly Cloudy'},
  }
  data = weather_data.get(
      city.lower(), {'temperature_f': 72, 'condition': 'Unknown'}
  )
  return {
      'city': city,
      'temperature_f': data['temperature_f'],
      'condition': data['condition'],
  }


def get_azure_openai_responses_model() -> AzureOpenAIResponsesLlm:
  """Builds the Azure OpenAI Responses model from environment variables."""
  return AzureOpenAIResponsesLlm(
      model=os.environ.get('AZURE_OPENAI_RESPONSES_DEPLOYMENT', 'gpt-5.5'),
      azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
      api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
      store=True,
      reasoning={'effort': 'medium', 'summary': 'concise'},
  )


root_agent = Agent(
    model=get_azure_openai_responses_model(),
    name='azure_openai_responses_agent',
    description=(
        'Manual E2E sample agent for the Azure OpenAI Responses API community'
        ' model.'
    ),
    instruction="""
You are a concise test assistant for ADK manual E2E validation.

Rules:
- For exact-response prompts, return only the requested text.
- When the user asks about weather, call get_current_weather.
- When reporting weather, include the city, temperature_f, and condition from
  the tool result.
- If the user asks what they told you earlier, answer from conversation context.
""",
    tools=[get_current_weather],
    generate_content_config=GenerateContentConfig(
        max_output_tokens=11512,
    ),
)
