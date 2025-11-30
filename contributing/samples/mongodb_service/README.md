# MongoDB Session Service Sample

This sample shows how to persist ADK sessions and state in MongoDB using the
community `MongoSessionService`.

## Prerequisites

- Python 3.9+ (Python 3.11+ recommended)
- A running MongoDB instance (local or Atlas) and a connection string with
  create/read/write permissions
- ADK and ADK Community installed
- Google API key for the sample agent (Gemini), set as `GOOGLE_API_KEY`

## Setup

### 1. Install dependencies

```bash
pip install google-adk google-adk-community python-dotenv
```

### 2. Configure environment variables

Create a `.env` in this directory:

```bash
# Required: Google API key for the agent
GOOGLE_API_KEY=your-google-api-key

# Recommended: Mongo connection string (Atlas or local)
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster-url>/
```

**Note:** Keep your Mongo credentials out of source control. The sample loads the connection string from the `MONGODB_URI` environment variable, which is loaded from the `.env` file at runtime.

### 3. Pick a database name

By default the sample uses `adk_sessions_db`. Collections are created
automatically if they do not exist.

## Usage

### Option 1: Run the included sample

```bash
python main.py
```

`main.py`:
- Creates a `MongoSessionService` with a connection string
- Creates a session for the demo user
- Runs the `financial_advisor_agent` with `Runner.run_async`
- Prints the agent's final response

### Option 2: Use `MongoSessionService` with your own runner

```python
import os
from google.adk.runners import Runner
from google.genai import types
from google.adk_community.sessions import MongoSessionService

session_service = MongoSessionService(
    connection_string=os.environ.get("MONGODB_URI")
)

await session_service.create_session(
    app_name="my_app", user_id="user1", session_id="demo"
)

runner = Runner(app_name="my_app", agent=root_agent, session_service=session_service)
query = "Hello, can you help me with my account?"
content = types.Content(role="user", parts=[types.Part(text=query)])

async for event in runner.run_async(
    user_id="user1",
    session_id="demo",
    new_message=content,
):
  if event.is_final_response():
    print(event.content.parts[0].text)
```

If you already have an `AsyncMongoClient`, pass it instead of a connection
string:

```python
from pymongo import AsyncMongoClient

client = AsyncMongoClient(host="localhost", port=27017)
session_service = MongoSessionService(client=client)
```

## Collections and indexing

`MongoSessionService` writes to two collections (configurable):
- `sessions`: conversation history and session-level state
- `session_state`: shared app/user state across sessions

Indexes are created on first use:
- Unique session identity: `(app_name, user_id, id)`
- Last update for recency queries: `(app_name, user_id, last_update_time)`

## Sample structure

```
mongodb_service/
├── main.py                    # Runs the sample with Mongo-backed sessions
├── mongo_service_agent/
│   ├── __init__.py            # Agent package init
│   └── agent.py               # Financial advisor agent with two tools
└── README.md                  # This file
```

## Sample agent

The agent (`mongo_service_agent/agent.py`) includes:
- `get_invoice_status(service)` tool for simple invoice lookups
- `calculate_service_tax(amount)` tool for tax calculations
- Gemini model (`gemini-2.0-flash`) with instructions to route to the tools

## Sample query

```
What is the status of my university invoice? Also, calculate the tax for a service amount of 9500 MXN.
```

## Configuration options (`MongoSessionService`)

- `database_name` (str, optional): Mongo database to store session data (defaults to `adk_sessions_db`)
- `connection_string` (str, optional): Mongo URI (mutually exclusive with `client`)
- `client` (AsyncMongoClient, optional): Provide your own client/connection pool
- `session_collection` (str, default `sessions`): Collection for session docs
- `state_collection` (str, default `session_state`): Collection for shared state
- `default_app_name` (str, optional): Fallback app name when not provided per call (defaults to `adk-cosmos-session-service`)

## Tips

- Use `runner.run_async` (as in `main.py`) to keep the Mongo client on the same
  event loop and avoid loop-bound client errors.
- For production, prefer environment variables or secrets managers for the
  connection string and database credentials.
