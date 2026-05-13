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

"""Wikipedia research agent powered by Amazon Bedrock.

Demonstrates BedrockModel integration with Google ADK by building a simple
research assistant that answers questions using Wikipedia.

Usage::

    # Default question
    python agent.py

    # Custom question
    python agent.py "Who invented the World Wide Web?"

    # Streaming mode
    python agent.py --stream "What is quantum computing?"

    # Use a different Bedrock model
    python agent.py --model amazon.nova-pro-v1:0 "What is AWS Lambda?"

Prerequisites::

    pip install google-adk-community[bedrock] wikipedia-api

AWS credentials must be configured via one of:
  - Environment variables (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
  - AWS credentials file (~/.aws/credentials)
  - IAM role (EC2 instance profile, ECS task role, Lambda execution role)
"""

import argparse
import asyncio
from functools import lru_cache
import os

import wikipediaapi
from google.adk.agents import Agent
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from google.adk_community.models.bedrock_model import BedrockModel

_APP_NAME = "bedrock_wikipedia_agent"


# ---------------------------------------------------------------------------
# Wikipedia tools
# ---------------------------------------------------------------------------


@lru_cache
def _get_wiki_client(language: str) -> wikipediaapi.Wikipedia:
  return wikipediaapi.Wikipedia(
    user_agent="google-adk-community-example/1.0", language=language
  )


def wikipedia_search(query: str, language: str = "en") -> dict:
  """Search Wikipedia and return a summary for the best-matching article.

  Args:
    query: The topic or question to search for on Wikipedia.
    language: Wikipedia language code (default: ``"en"``).

  Returns:
    A dict containing ``title``, ``snippet``, ``url``, and optionally
    ``related`` articles. Returns a ``"no_results"`` status dict when no
    matching article is found.
  """
  wiki = _get_wiki_client(language)
  page = wiki.page(query)
  if not page.exists():
    return {
      "status": "no_results",
      "query": query,
      "message": f"No Wikipedia article found for: {query}",
    }

  summary = page.summary
  snippet = summary[:500] + "..." if len(summary) > 500 else summary

  related = []
  for _, link_page in list(page.links.items())[:3]:
    if link_page.exists():
      s = link_page.summary
      related.append({
        "title": link_page.title,
        "snippet": s[:150] + "..." if len(s) > 150 else s,
      })

  return {
    "status": "success",
    "title": page.title,
    "snippet": snippet,
    "url": page.fullurl,
    "related": related,
  }


def wikipedia_get_article(
  title: str,
  summary_only: bool = True,
  max_length: int = 3000,
  language: str = "en",
) -> dict:
  """Retrieve content from a Wikipedia article by its exact title.

  Args:
    title: Exact Wikipedia article title (e.g. ``"Python (programming
      language)"``).
    summary_only: When ``True`` (default), return only the introductory
      summary. Set to ``False`` for the full article text.
    max_length: Maximum character length of full-text content (default 3000).
    language: Wikipedia language code (default: ``"en"``).

  Returns:
    A dict containing ``title``, ``content``, ``url``, and ``categories``.
    Returns a ``"not_found"`` status dict when the article does not exist.
  """
  wiki = _get_wiki_client(language)
  page = wiki.page(title)
  if not page.exists():
    return {
      "status": "not_found",
      "title": title,
      "message": f"Wikipedia article not found: {title}",
    }

  if summary_only:
    content = page.summary
  else:
    content = page.text[:max_length]
    if len(page.text) > max_length:
      content += "\n\n[... content truncated]"

  return {
    "status": "success",
    "title": page.title,
    "content": content,
    "url": page.fullurl,
    "categories": list(page.categories.keys())[:5],
  }


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def build_agent(model_id: str, region: str) -> Agent:
  """Create a Wikipedia research Agent backed by Bedrock.

  Args:
    model_id: Bedrock model ID or cross-region inference profile.
    region: AWS region for the Bedrock API endpoint.

  Returns:
    A configured ADK :class:`~google.adk.agents.Agent`.
  """
  return Agent(
    model=BedrockModel(model=model_id, region_name=region, max_tokens=2048),
    name="wikipedia_research_agent",
    description="Answers questions using Wikipedia via Amazon Bedrock.",
    instruction=(
      "You are a concise research assistant. "
      "Use wikipedia_search to find relevant articles and "
      "wikipedia_get_article to retrieve detail when needed. "
      "Always cite the Wikipedia URL in your final answer."
    ),
    tools=[wikipedia_search, wikipedia_get_article],
  )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def ask(question: str, model_id: str, region: str, stream: bool) -> None:
  """Send a single question to the agent and print the response.

  Args:
    question: The user's question.
    model_id: Bedrock model ID to use.
    region: AWS region.
    stream: When ``True``, stream partial text deltas to stdout.
  """
  agent = build_agent(model_id, region)
  session_service = InMemorySessionService()
  runner = Runner(
    agent=agent,
    app_name=_APP_NAME,
    session_service=session_service,
  )
  session = await session_service.create_session(
    app_name=_APP_NAME, user_id="user"
  )

  print(f"\n{'='*60}")
  print(f"Model : {model_id}  |  Region : {region}")
  print(f"Q: {question}")
  print(f"{'='*60}\n")

  async for event in runner.run_async(
    user_id="user",
    session_id=session.id,
    new_message=types.Content(
      role="user", parts=[types.Part.from_text(text=question)]
    ),
  ):
    if not event.content or not event.content.parts:
      continue
    for part in event.content.parts:
      if part.function_call:
        print(
          f"  [tool] {part.function_call.name}({part.function_call.args})"
        )
      elif part.function_response:
        status = (part.function_response.response or {}).get("status", "?")
        print(f"  [result] status={status}")
      elif part.text:
        if stream and not event.is_final_response():
          print(part.text, end="", flush=True)
        elif event.is_final_response():
          if stream:
            print()  # newline after streamed output
          print(f"\nA:\n{part.text}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Wikipedia research agent powered by Amazon Bedrock + ADK"
  )
  parser.add_argument(
    "question",
    nargs="?",
    default="What is Amazon Bedrock?",
    help="Question to answer (default: 'What is Amazon Bedrock?')",
  )
  parser.add_argument(
    "--model",
    default="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    help="Bedrock model ID (default: us.anthropic.claude-haiku-4-5-20251001-v1:0)",
  )
  parser.add_argument(
    "--region",
    default=os.environ.get("AWS_REGION", "us-east-1"),
    help="AWS region (default: AWS_REGION env var or us-east-1)",
  )
  parser.add_argument(
    "--stream",
    action="store_true",
    help="Stream text output to stdout",
  )
  args = parser.parse_args()

  asyncio.run(ask(args.question, args.model, args.region, args.stream))


if __name__ == "__main__":
  main()
