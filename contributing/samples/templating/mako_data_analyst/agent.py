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

"""Data analyst bot example using MakoInstructionProvider.

This example demonstrates how to use Mako templating for dynamic agent
instructions that include Python expressions and calculations.

The agent analyzes sales data and provides insights, showcasing Mako's
Python-centric approach with inline expressions.

Usage:
    adk run contributing/samples/templating/mako_data_analyst
"""

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk_community.templating import MakoInstructionProvider


def populate_sales_data(callback_context: CallbackContext):
  """Populate the session state with sales data.

  In a real application, this would fetch data from a database.
  For this demo, we use sample data.
  """
  callback_context.state['company_name'] = 'TechCorp Inc.'
  callback_context.state['quarter'] = 'Q4 2024'
  callback_context.state['sales_data'] = [
      {'product': 'Widget A', 'revenue': 125000, 'units': 450, 'region': 'North'},
      {'product': 'Widget B', 'revenue': 89000, 'units': 320, 'region': 'South'},
      {'product': 'Widget C', 'revenue': 156000, 'units': 580, 'region': 'East'},
      {'product': 'Widget D', 'revenue': 72000, 'units': 210, 'region': 'West'},
  ]
  callback_context.state['target_revenue'] = 500000


# Define the Mako template for instructions
# Mako allows Python expressions directly in templates using ${}
DATA_ANALYST_TEMPLATE = """
You are a Data Analyst Assistant for ${company_name}.

SALES REPORT - ${quarter}
${'=' * 50}

% if sales_data:
## Calculate totals using Python expressions
<%
  total_revenue = sum(item['revenue'] for item in sales_data)
  total_units = sum(item['units'] for item in sales_data)
  avg_price = total_revenue / total_units if total_units > 0 else 0
%>

Summary:
  Total Revenue: $$${"{:,}".format(total_revenue)}
  Total Units Sold: ${"{:,}".format(total_units)}
  Average Price per Unit: $$${"{:.2f}".format(avg_price)}
  Target Revenue: $$${"{:,}".format(target_revenue)}
  Performance: ${"{:.1f}%".format((total_revenue / target_revenue * 100) if target_revenue > 0 else 0)} of target

Product Breakdown:
% for item in sales_data:
  - ${item['product']} (${item['region']}):
    Revenue: $$${"{:,}".format(item['revenue'])} | Units: ${item['units']}
    Avg Price: $$${"{:.2f}".format(item['revenue'] / item['units'] if item['units'] > 0 else 0)}
    % if item['revenue'] > 100000:
    ⭐ Top Performer!
    % elif item['revenue'] < 80000:
    ⚠️  Needs attention
    % endif
% endfor

% else:
No sales data available for this period.
% endif

INSTRUCTIONS
============
Based on the data above:
1. Analyze sales trends and patterns
2. Identify top and underperforming products
3. Provide actionable recommendations for improvement
4. Answer any questions about the sales data

Be data-driven, specific, and highlight key insights.
""".strip()

# Create the provider with the template
instruction_provider = MakoInstructionProvider(DATA_ANALYST_TEMPLATE)

# Create the agent
root_agent = Agent(
    name='data_analyst_agent',
    model='gemini-2.0-flash',
    instruction=instruction_provider,
    before_agent_callback=[populate_sales_data],
)
