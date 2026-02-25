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

"""DevOps bot example using Jinja2InstructionProvider.

This example demonstrates how to use Jinja2 templating for dynamic agent
instructions that adapt based on system state.

The agent monitors server status and provides recommendations based on
CPU usage, using Jinja2's powerful features like loops, conditionals,
and filters.

Usage:
    adk run contributing/samples/templating/jinja2_devops_bot
"""

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk_community.templating import Jinja2InstructionProvider


def populate_system_state(callback_context: CallbackContext):
    """Populate the session state with system information.

    In a real application, this would fetch actual server metrics.
    For this demo, we use sample data.
    """
    callback_context.state["environment"] = "PRODUCTION"
    callback_context.state["user_query"] = ""
    callback_context.state["servers"] = [
        {"id": "srv-01", "role": "LoadBalancer", "cpu": 15, "memory": 45},
        {"id": "srv-02", "role": "AppServer", "cpu": 95, "memory": 88},
        {"id": "srv-03", "role": "Database", "cpu": 72, "memory": 65},
        {"id": "srv-04", "role": "Cache", "cpu": 35, "memory": 42},
    ]


# Define the Jinja2 template for instructions
DEVOPS_TEMPLATE = """
You are a System Reliability Engineer (SRE) Assistant.

CURRENT SYSTEM STATUS
=====================
Environment: {{ environment }}

{% if servers|length == 0 %}
⚠️ No servers are currently online.
{% else %}
Active Servers ({{ servers|length }}):
{% for server in servers %}
  - [{{ server.id }}] {{ server.role }}
    CPU: {{ server.cpu }}% | Memory: {{ server.memory }}%
    {% if server.cpu > 80 or server.memory > 80 %}
    🚨 CRITICAL ALERT: Resource usage is dangerously high!
    {% elif server.cpu > 60 or server.memory > 60 %}
    ⚠️  WARNING: Resource usage is elevated.
    {% else %}
    ✅ Status: Normal
    {% endif %}
{% endfor %}
{% endif %}

INSTRUCTIONS
============
Based on the status above:
1. Analyze the current system health
2. Identify any critical issues or warnings
3. Provide specific, actionable recommendations
4. If user asks a question, answer it in the context of the current system state

{% if user_query %}
User Question: {{ user_query }}
{% endif %}

Be concise, technical, and prioritize critical issues.
""".strip()

# Create the provider with the template
instruction_provider = Jinja2InstructionProvider(DEVOPS_TEMPLATE)

# Create the agent
root_agent = Agent(
    name="devops_sre_agent",
    model="gemini-2.0-flash",
    instruction=instruction_provider,
    before_agent_callback=[populate_system_state],
)
