# `google.adk_community.models`

Community-contributed model integrations for [Google ADK](https://google.github.io/adk-docs/).

## Available models

| Class | Provider | Install extra |
|---|---|---|
| `BedrockModel` | Amazon Bedrock (Converse API) | `bedrock` |

---

## `BedrockModel`

Native Amazon Bedrock integration via the
[Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html).
Supports all Bedrock-hosted models and cross-region inference profiles.

### Supported models

Any model available via the Bedrock Converse API is supported, including cross-region
inference profiles (`us.*`, `eu.*`, `ap.*`). See the
[Amazon Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html)
for the full list of supported models.

### Installation

```bash
pip install "google-adk-community[bedrock]"
```

### Quick start

```python
from google.adk.agents import Agent
from google.adk_community.models import BedrockModel

agent = Agent(
    model=BedrockModel(model="us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    name="my_agent",
    instruction="You are a helpful assistant.",
    tools=[...],
)
```

Because `BedrockModel` is registered with `LLMRegistry` on import, you can
also pass the model ID as a plain string after importing the module:

```python
import google.adk_community.models  # triggers LLMRegistry registration

agent = Agent(model="us.anthropic.claude-haiku-4-5-20251001-v1:0", ...)
```

### AWS authentication

`BedrockModel` resolves credentials through the standard
[boto3 credential chain](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html):

1. **Environment variables** — `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
2. **AWS credentials file** — `~/.aws/credentials`
3. **IAM role** — EC2 instance profile, ECS task role, Lambda execution role

The AWS region is resolved in this order:

1. `region_name` constructor argument
2. `region_name` of the supplied `boto_session`
3. `AWS_REGION` environment variable
4. `AWS_DEFAULT_REGION` environment variable
5. Fallback: `us-east-1`

#### Custom boto3 session (assumed role, named profile, …)

```python
import boto3
from google.adk_community.models import BedrockModel

session = boto3.Session(profile_name="my-prod-profile")
# or an assumed-role session:
# sts = boto3.client("sts")
# creds = sts.assume_role(RoleArn="arn:aws:iam::123456789:role/MyRole",
#                         RoleSessionName="adk-session")["Credentials"]
# session = boto3.Session(
#     aws_access_key_id=creds["AccessKeyId"],
#     aws_secret_access_key=creds["SecretAccessKey"],
#     aws_session_token=creds["SessionToken"],
# )

model = BedrockModel(
    model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    boto_session=session,
)
```

> **Note:** `boto_session` and `region_name` are mutually exclusive.
> Pass the region when constructing the `boto3.Session` instead.

### Configuration reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock model ID or inference profile |
| `region_name` | `str \| None` | `None` | AWS region (see resolution order above) |
| `max_tokens` | `int` | `4096` | Maximum tokens to generate |
| `guardrail_id` | `str \| None` | `None` | Bedrock Guardrail identifier |
| `guardrail_version` | `str \| None` | `None` | Bedrock Guardrail version (`"1"`, `"DRAFT"`, …) |
| `boto_session` | `boto3.Session \| None` | `None` | Pre-configured boto3 session |

### Guardrails

```python
model = BedrockModel(
    model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    guardrail_id="abc123def456",
    guardrail_version="1",
)
```

Responses blocked by a guardrail are returned with
`finish_reason = FinishReason.SAFETY`.

### Streaming

Streaming is enabled by default when the ADK runner requests it. Each text
delta is yielded as a partial `LlmResponse` (`partial=True`), followed by a
single aggregated final response (`partial=False`).

### Error handling

`BedrockModel` propagates `botocore.exceptions.ClientError` unchanged so
callers can inspect `error.response["Error"]["Code"]`. Common codes:

| Code | Meaning |
|---|---|
| `ThrottlingException` | Rate limit exceeded — add retry/back-off logic |
| `ValidationException` | Invalid request — often a context-window overflow |
| `AccessDeniedException` | IAM principal lacks model access — check the [Bedrock console](https://console.aws.amazon.com/bedrock/home#/modelaccess) |
