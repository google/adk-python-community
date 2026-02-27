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

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.function_tool import FunctionTool


load_dotenv()


def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Parameters:
    -----------
    city: str
        The name of the city to get weather for.

    Returns:
    --------
    dict: Contains weather information for the city.
    """
    weather_data = {
        "San Francisco": {"temperature": 18, "condition": "Foggy"},
        "New York": {"temperature": 22, "condition": "Sunny"},
        "London": {"temperature": 15, "condition": "Rainy"},
        "Tokyo": {"temperature": 25, "condition": "Clear"},
    }

    if city in weather_data:
        return weather_data[city]
    else:
        return {"temperature": 20, "condition": "Unknown"}


weather_tool = FunctionTool(func=get_weather)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="weather_assistant",
    description="A simple weather assistant that provides weather information.",
    instruction=(
        "You are a helpful weather assistant. "
        "Use the get_weather tool to provide weather information for cities. "
        "Be friendly and concise in your responses."
    ),
    tools=[weather_tool],
)
