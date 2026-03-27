# SDC Agents -- Semantic Data Governance for ADK

Thin re-export wrapper over the
[`sdc-agents`](https://pypi.org/project/sdc-agents/) PyPI package. The
canonical source lives at
[SemanticDataCharter/SDC_Agents](https://github.com/SemanticDataCharter/SDC_Agents);
this module provides importability through the `google.adk_community`
namespace.

## Installation

```bash
pip install google-adk-community[sdc-agents]
```

## Usage

```python
from google.adk.agents import LlmAgent
from google.adk_community.sdc_agents import (
    load_config,
    CatalogToolset,
    IntrospectToolset,
    MappingToolset,
)

config = load_config("sdc-agents.yaml")

agent = LlmAgent(
    name="data_governance_agent",
    model="gemini-2.0-flash",
    description="Introspects data sources and maps them to SDC4 schemas.",
    instruction=(
        "You help data engineers govern their data. When given a datasource:\n"
        "1. Introspect the structure to discover columns and types\n"
        "2. Search the SDC4 catalog for matching published schemas\n"
        "3. Map columns to schema components by type and name similarity\n"
        "4. Report the mapping with confidence scores"
    ),
    tools=[
        IntrospectToolset(config=config),
        CatalogToolset(config=config),
        MappingToolset(config=config),
    ],
)
```

## Exported Toolsets

| Toolset | Description |
|---------|-------------|
| **CatalogToolset** | Discover published SDC4 schemas, download artifacts (XSD, RDF, JSON-LD) |
| **IntrospectToolset** | Analyze datasource structure -- infer column types and constraints from SQL, CSV, JSON, MongoDB with sidecar metadata support |
| **MappingToolset** | Match datasource columns to schema components by type compatibility and name similarity, persist mapping configs with schema and datasource context |
| **AssemblyToolset** | Compose data models from catalog components -- reuse existing or mint new, with catalog-first discovery and structured unmatched column reporting |
| **GeneratorToolset** | Generate validated XML instances, batch processing, and preview |
| **ValidationToolset** | Validate XML instances against schemas, digitally sign via VaaS API |
| **DistributionToolset** | Deliver RDF triples to Fuseki, Neo4j, GraphDB, or REST endpoints |
| **KnowledgeToolset** | Index domain documentation (JSON, CSV, TTL, Markdown, PDF, DOCX) for semantic search |

## Configuration

SDC Agents uses a YAML config file with environment variable substitution:

```yaml
sdcstudio:
  base_url: "https://sdcstudio.com"
  api_key: "${SDC_API_KEY}"

datasources:
  warehouse:
    type: csv
    path: "./data/sample.csv"

cache:
  root: ".sdc-cache"

audit:
  path: ".sdc-cache/audit.jsonl"
```

## Resources

- [SDC Agents on PyPI](https://pypi.org/project/sdc-agents/)
- [SDC Agents GitHub](https://github.com/SemanticDataCharter/SDC_Agents)
- [ADK Integration Guide](https://github.com/SemanticDataCharter/SDC_Agents/blob/main/docs/integrations/ADK_INTEGRATION.md)
- [SDCStudio](https://sdcstudio.com)
