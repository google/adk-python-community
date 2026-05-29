# OpenAI Responses Models

This package provides ADK model adapters backed by the OpenAI Responses API.

Install the optional OpenAI dependency before using these models:

```bash
pip install "google-adk-community[openai]"
```

## OpenAI

```python
from google.adk_community.models.openai_responses import OpenAIResponsesLlm

model = OpenAIResponsesLlm(model="gpt-5")
```

## Azure OpenAI

Azure OpenAI exposes the Responses API through an OpenAI-compatible
`/openai/v1/responses` endpoint. Use the Azure deployment name as the model.

```python
from google.adk_community.models.openai_responses import AzureOpenAIResponsesLlm

model = AzureOpenAIResponsesLlm(
    model="my-gpt-5-deployment",
    azure_endpoint="https://my-resource.openai.azure.com/",
)
```

Set `AZURE_OPENAI_API_KEY`, or pass `api_key` directly.
