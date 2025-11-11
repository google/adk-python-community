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

from google.adk import Agent

# Example Insight Agent:
# Starting point for analytics-focused users.
# Extend with tools that query data sources (BigQuery, etc.), summarize KPIs,
# and send insights to channels like Slack or email.

root_agent = Agent(
    model="gemini-2.5-flash",
    name="insight_agent",
    description=(
        "An example ADK agent that can be extended to generate KPI summaries, "
        "highlight anomalies, and support decision-making for data teams."
    ),
    instruction=(
        "Explain metrics clearly in natural language, highlight notable changes, "
        "and suggest possible next actions. When tools are configured, use them "
        "to fetch relevant data instead of guessing."
    ),
    # tools=[],  # To be filled by users integrating this example with their data stack.
)
