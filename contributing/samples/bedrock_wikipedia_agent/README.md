# Bedrock Wikipedia Agent

A minimal research agent built with **Google ADK** and **Amazon Bedrock** that
answers questions by searching Wikipedia.

## What it demonstrates

- Instantiating `BedrockModel` and passing it to an ADK `Agent`
- Defining Python functions as ADK tools (`wikipedia_search`, `wikipedia_get_article`)
- Running an agent turn with `Runner.run_async`
- Streaming and non-streaming response modes

## Prerequisites

```bash
pip install "google-adk-community[bedrock]" wikipedia-api
```

AWS credentials must be available via the standard boto3 credential chain:

| Method | How |
|---|---|
| Environment variables | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_REGION` |
| Credentials file | `~/.aws/credentials` |
| IAM role | EC2 instance profile, ECS task role, Lambda execution role |

## Usage

```bash
# Default question
python agent.py

# Custom question
python agent.py "Who invented the World Wide Web?"

# Streaming output
python agent.py --stream "What is quantum computing?"

# Different model and region
python agent.py --model amazon.nova-pro-v1:0 --region us-west-2 "What is AWS Lambda?"
```

## Sample output

```
============================================================
Model : us.anthropic.claude-haiku-4-5-20251001-v1:0  |  Region : us-east-1
Q: What is Python?
============================================================

  [tool] wikipedia_search({'query': 'Python programming language'})
  [result] status=success

A:
**Python** is a high-level, general-purpose programming language known for its emphasis on code readability through significant indentation. Here are the key characteristics:

**Main Features:**
- **Design Philosophy**: Emphasizes code readability and clean syntax
- **Type System**: Dynamically type-checked
- **Memory Management**: Garbage-collected (automatic memory management)
- **Programming Paradigms**: Supports multiple styles including:
  - Structured programming (particularly procedural)
  - Object-oriented programming
  - Functional programming

**History:**
- Created by Guido van Rossum in the late 1980s as a successor to the ABC programming language
- Python 3.0, released in 2008, was a major revision

Python has become one of the most popular programming languages due to its readability, versatility, and ease of learning, making it suitable for web development, data science, artificial intelligence, automation, and many other applications.

**Source:** https://en.wikipedia.org/wiki/Python_(programming_language)
```
