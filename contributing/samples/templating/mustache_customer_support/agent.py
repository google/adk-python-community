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

"""Customer support bot example using MustacheInstructionProvider.

This example demonstrates how to use Mustache (logic-less) templating
for dynamic agent instructions that are simple and declarative.

The agent provides personalized customer support based on user tier
and ticket history, showcasing Mustache's clean, minimal syntax.

Usage:
    adk run contributing/samples/templating/mustache_customer_support
"""

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk_community.templating import MustacheInstructionProvider


def populate_customer_data(callback_context: CallbackContext):
  """Populate the session state with customer support data.

  In a real application, this would fetch data from a CRM system.
  For this demo, we use sample data.
  """
  callback_context.state['customer'] = {
      'name': 'Sarah Johnson',
      'id': 'CUST-12345',
      'tier': 'Premium',
      'is_premium': True,
      'account_age_days': 847,
  }

  callback_context.state['open_tickets'] = [
      {
          'id': 'TKT-001',
          'subject': 'Billing discrepancy',
          'priority': 'High',
          'is_high_priority': True,
      },
      {
          'id': 'TKT-002',
          'subject': 'Feature request',
          'priority': 'Low',
          'is_high_priority': False,
      },
  ]

  callback_context.state['has_open_tickets'] = True
  callback_context.state['recent_purchases'] = [
      {'product': 'Enterprise Plan', 'date': '2024-10-15'},
      {'product': 'Add-on Storage', 'date': '2024-11-01'},
  ]


# Define the Mustache template for instructions
# Mustache uses {{}} for variables, {{#}} for sections, {{^}} for inverted sections
CUSTOMER_SUPPORT_TEMPLATE = """
You are a Customer Support Agent.

CUSTOMER PROFILE
================
Name: {{customer.name}}
Customer ID: {{customer.id}}
Account Tier: {{customer.tier}}
Account Age: {{customer.account_age_days}} days

{{#customer.is_premium}}
⭐ PREMIUM CUSTOMER - Provide priority white-glove service!
{{/customer.is_premium}}

{{^customer.is_premium}}
Standard customer - Provide professional, helpful service.
{{/customer.is_premium}}

OPEN TICKETS
============
{{#has_open_tickets}}
Active Support Tickets:
{{#open_tickets}}
  - [{{id}}] {{subject}}
    Priority: {{priority}}
    {{#is_high_priority}}🔴 HIGH PRIORITY - Address immediately!{{/is_high_priority}}
{{/open_tickets}}
{{/has_open_tickets}}

{{^has_open_tickets}}
No open tickets - customer is all caught up!
{{/has_open_tickets}}

RECENT PURCHASES
================
{{#recent_purchases}}
  - {{product}} (purchased {{date}})
{{/recent_purchases}}

INSTRUCTIONS
============
1. Greet the customer warmly by name
2. Acknowledge their account tier and value
3. {{#has_open_tickets}}Prioritize resolving their open tickets{{/has_open_tickets}}
4. {{^has_open_tickets}}Ask how you can help them today{{/has_open_tickets}}
5. Be empathetic, professional, and solution-oriented
6. {{#customer.is_premium}}Offer additional premium perks or expedited solutions{{/customer.is_premium}}

Your goal is to ensure customer satisfaction and resolve issues efficiently.
""".strip()

# Create the provider with the template
instruction_provider = MustacheInstructionProvider(CUSTOMER_SUPPORT_TEMPLATE)

# Create the agent
root_agent = Agent(
    name='customer_support_agent',
    model='gemini-2.0-flash',
    instruction=instruction_provider,
    before_agent_callback=[populate_customer_data],
)
