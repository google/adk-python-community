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
from google.adk.tools import FunctionTool


_TAX_RATE = 0.16
load_dotenv()

# Tool 1
def get_invoice_status(service: str) -> dict:
  """Return the invoice status details for the requested service.

  Args:
    service: Business category to query (for example 'gas' or 'restaurant').

  Returns:
    dict: Contains a ``status`` entry with the invoice state and a ``report``
      entry that gives the relevant context for that status.
  """
  if service == "university":
    return {"status": "success", "report": "All TEC invoices are paid."}
  elif service == "gas":
    return {"status": "pending", "report": "Gas invoice due in 5 days."}
  elif service == "restaurant":
    return {
        "status": "overdue",
        "report": "Restaurant invoice is overdue by 10 days.",
    }
  else:
    return {"status": "error", "report": "Service not recognized."}


invoice_tool = FunctionTool(func=get_invoice_status)


# Tool 2
def calculate_service_tax(amount: float) -> dict:
  """Calculate tax for a service amount using a fixed rate.

  Args:
    amount: Untaxed amount that needs a tax calculation.

  Returns:
    dict: Keys ``amount``, ``tax_amount``, and ``total_amount`` capturing the
      original value, the computed tax, and the amount plus tax respectively.
  """
  tax_amount = amount * _TAX_RATE
  total_amount = amount + tax_amount
  return {
      "amount": amount,
      "tax_amount": tax_amount,
      "total_amount": total_amount,
  }


tax_tool = FunctionTool(func=calculate_service_tax)

# Agent
root_agent = Agent(
    model="gemini-2.0-flash",
    name="financial_advisor_agent",
    description="Financial advisor agent for managing invoices and calculating service taxes.",
    instruction=(
        "You are an AI agent designed to assist users with financial"
        " inquiries related to invoices and service tax calculations.\n"
        "**Available Tools:**\n"
        "1. get_invoice_status(service): Retrieves the status of invoices\n"
        "2. calculate_service_tax(amount): Calculates the tax for a given amount\n"
        "Use these tools to assist users with their financial inquiries.\n"
        "If the user asks about other financial topics, respond politely"
        " that you can only assist with invoice status and tax calculations.\n"
    ),
    tools=[invoice_tool, tax_tool],
)
