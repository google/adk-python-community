# SDC Agents Demo

A minimal example composing SDC Agents toolsets with an ADK `LlmAgent`.

## Prerequisites

- Python 3.11+
- An SDCStudio API key (set as `SDC_API_KEY` environment variable)

## Setup

```bash
pip install google-adk-community[sdc-agents]

export SDC_API_KEY="your-sdcstudio-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

## Usage

```bash
# Run with the ADK CLI
adk run .

# Or use the ADK web UI
adk web .
```

## What This Demo Does

The agent composes two SDC Agents toolsets:

- **CatalogToolset**: Search published SDC4 schemas, download artifacts
  (XSD, RDF, JSON-LD), and check wallet balance.
- **IntrospectToolset**: Analyze a datasource to infer column types,
  constraints, and statistics.

## Sample Queries

```
> Search the catalog for schemas related to lab results
> Introspect the sample datasource
> What published schemas match the columns in my datasource?
```

## Structure

```
sdc_agents_demo/
├── agent.py    # Agent definition with SDC toolsets
└── README.md   # This file
```

## Resources

- [SDC Agents Documentation](https://github.com/SemanticDataCharter/SDC_Agents)
- [SDC Agents on PyPI](https://pypi.org/project/sdc-agents/)
- [ADK Integration Guide](https://github.com/SemanticDataCharter/SDC_Agents/blob/main/docs/integrations/ADK_INTEGRATION.md)
