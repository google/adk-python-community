# Community Planner Content Blocks

This module contains community-contributed helpers for working with ADK planner
output.

## What it is

ADK planners such as
[`PlanReActPlanner`](https://github.com/google/adk-python/blob/main/src/google/adk/planners/plan_re_act_planner.py)
and
[`BuiltInPlanner`](https://github.com/google/adk-python/blob/main/src/google/adk/planners/built_in_planner.py)
emit their output as a list of `google.genai.types.Part` objects. Reasoning is
signalled on those parts with `thought=True` and, for `PlanReActPlanner`, the
text is additionally annotated with inline `/*PLANNING*/` style tags that
callers have to parse themselves.

`parts_to_content_blocks` is a small, **read-only and additive** helper that
converts that `Part` list into a provider-agnostic, structured representation
modelled after
[LangChain v1's standard content blocks](https://docs.langchain.com/oss/python/langchain/messages#standard-content-blocks),
so consumers can branch on a typed `type` discriminator instead of re-parsing
raw text. It never mutates the parts it is given and it does not change how
planners build instructions or post-process responses, so existing consumers of
the `Part` output are unaffected.

## Installation

```bash
pip install google-adk-community
```

No extra dependency group is required: the converter only depends on
`google-genai` (already pulled in by `google-adk`).

## Usage

```python
from google.adk_community.planners import parts_to_content_blocks

# `response` is an LLM response whose parts a planner produced.
blocks = parts_to_content_blocks(response.content.parts)

for block in blocks:
    if block["type"] == "reasoning":
        print("reasoning:", block["reasoning"], "(", block["reasoning_kind"], ")")
    elif block["type"] == "tool_call":
        print("tool call:", block["name"], block["args"])
    elif block["type"] == "text":
        print("text:", block["text"])
```

You can also convert a single part with `part_to_content_block(part)`, which
returns `None` for parts that carry no mappable content (e.g. an empty part, or
a redacted thought that only carries a signature).

## Block schema

`parts_to_content_blocks` returns a list of typed dicts, each carrying a `type`
discriminator. New `type`s may be added in the future, so consumers should
treat unknown types defensively.

| `type` | Fields | Produced from |
|--------|--------|---------------|
| `reasoning` | `reasoning: str`, `reasoning_kind: "planning" \| "replanning" \| "reasoning" \| "action" \| None` | A `thought=True` text part. `reasoning_kind` is derived from a `PlanReActPlanner` leading tag when present, otherwise `None` (e.g. `BuiltInPlanner` thinking). |
| `text` | `text: str` | A non-thought text part (e.g. a planner's final answer). |
| `tool_call` | `name: str`, `args: dict`, `id: str \| None` | A part carrying a `function_call`. Takes precedence over any incidental text on the part. |

The public API also exports the `ReasoningContentBlock`, `TextContentBlock` and
`ToolCallContentBlock` `TypedDict`s, the `ReasoningKind` literal, and the
`ContentBlock` union type for typing your own code.

## See Also

- [planner_content_blocks Implementation](./planner_content_blocks.py)
- [Tests](../../../../tests/unittests/planners/test_planner_content_blocks.py)
