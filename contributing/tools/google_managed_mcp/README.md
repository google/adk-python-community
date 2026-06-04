# GoogleManagedMcpToolset

A `McpToolset` subclass that simplifies connecting to Google-managed MCP servers
by automating Application Default Credentials (ADC) authentication and endpoint
URL resolution.

## Features

- **Automatic endpoint resolution**: Just specify the product name (e.g.,
  `"bigquery"`) and the correct MCP server URL is resolved automatically.
- **ADC-based authentication**: Uses Application Default Credentials with
  automatic OAuth2 token refresh — no manual token management required.
- **Custom scopes & project ID**: Supports optional OAuth2 scope overrides and
  project ID for quota attribution.
- **Full McpToolset compatibility**: All standard `McpToolset` parameters
  (`tool_filter`, `tool_name_prefix`, etc.) are supported.

## Supported Products

| Product    | Endpoint URL                              |
|------------|-------------------------------------------|
| `bigquery` | `https://bigquery.googleapis.com/mcp`     |

## Installation

```bash
pip install google-adk-community[google-managed-mcp]
```

## Usage

```python
from google.adk.agents import Agent
from google.adk_community.tools.google_managed_mcp import (
    GoogleManagedMcpToolset,
)

# Minimal — uses ADC and default BigQuery scopes
toolset = GoogleManagedMcpToolset(product="bigquery")

# With custom project and scopes
toolset = GoogleManagedMcpToolset(
    product="bigquery",
    project_id="my-project",
    scopes=["https://www.googleapis.com/auth/bigquery.readonly"],
    tool_filter=["list_datasets", "execute_sql"],
)

agent = Agent(
    model="gemini-2.5-flash",
    name="bq_agent",
    instruction="Help users query BigQuery.",
    tools=[toolset],
)
```
