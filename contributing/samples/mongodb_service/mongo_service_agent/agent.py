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

MODEL_ID = "gemini-2.0-flash"

load_dotenv()


# Tool 1
def get_invoice_status(service: str) -> dict:
  """Retrieves the status of invoices for a given service.

  Returns:
  dict: A dictionary with invoice status details with a "status" key indicating the invoice status and a "report" key providing additional information.
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
  """Calculates the tax for a given service amount.

  Returns:
  dict: A dictionary with the original amount, calculated tax amount, and total amount including tax.
  """
  tax_rate = 0.16
  tax_amount = amount * tax_rate
  total_amount = amount + tax_amount
  return {
      "amount": amount,
      "tax_amount": tax_amount,
      "total_amount": total_amount,
  }


tax_tool = FunctionTool(func=calculate_service_tax)

# Agent
root_agent = Agent(
    model=MODEL_ID,
    name="financial_advisor_agent",
    description=(
        "Financial advisor agent for managing invoices and calculating service taxes."
    ),
    instruction=(
        "You are an AI agent designed to assist users with financial",
        " inquiries related to invoices and service tax calculations.\n",
        "**Available Tools:**\n",
        "1. get_invoice_status(service): Retrieves the status of invoices\n"
        "2. calculate_service_tax(amount): Calculates the tax for a given amount\n"
        "Use these tools to assist users with their financial inquiries.\n",
        "If the user asks about other financial topics, respond politely",
        " that you can only assist with invoice status and tax calculations.\n",
    ),
    tools=[invoice_tool, tax_tool],
)
