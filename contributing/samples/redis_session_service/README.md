# Redis Session Service Sample

This sample demonstrates how to use Redis as a session storage backend for ADK agents using the community package.

## Prerequisites

- Python 3.9+ (Python 3.11+ recommended)
- Redis server running locally or remotely
- ADK and ADK Community installed

## Setup

### 1. Install Dependencies

```bash
pip install google-adk-community[redis]
```

### 2. Set Up Redis Server

#### Option A: Using Homebrew (macOS)

```bash
brew install redis
brew services start redis
```

#### Option B: Using Docker

```bash
docker run -d -p 6379:6379 redis:latest
```

#### Option C: Using apt (Ubuntu/Debian)

```bash
sudo apt-get install redis-server
sudo systemctl start redis-server
```

Once Redis is running, verify the connection:

```bash
redis-cli ping
# Should return: PONG
```

### 3. Configure Environment Variables

Create a `.env` file in this directory:

```bash
# Required: Google API key for the agent
GOOGLE_API_KEY=your-google-api-key

# Required: Redis connection URI
REDIS_URI=redis://localhost:6379
```

**Note:** The default Redis URI assumes Redis is running locally on port 6379. Adjust if your Redis instance is running elsewhere.

## Usage

### Running the Sample

The simplest way to use this sample is to run the included `main.py`:

```bash
python main.py
```

This will start a FastAPI web server on `http://localhost:8080` with:
- A web interface for interacting with the weather assistant agent
- Redis-backed session persistence
- Session TTL (Time To Live) of 1 hour
- Health check endpoint at `/health`

The `main.py` file demonstrates how to:
- Register the Redis session service with the service registry
- Use `get_fast_api_app` to create a FastAPI application with Redis session support
- Configure session expiration and CORS settings
- Serve the web interface for agent interaction

### Using get_fast_api_app with Redis

```python
import os
import uvicorn
from google.adk.cli.fast_api import get_fast_api_app

# Configure application
AGENTS_DIR = "./redis_service_agent"
SESSION_SERVICE_URI = "redis://localhost:6379/0"
ALLOWED_ORIGINS = ["http://localhost", "http://localhost:8080", "*"]

# Create FastAPI app with Redis session service
app = get_fast_api_app(
    agents_dir=AGENTS_DIR,
    session_service_uri=SESSION_SERVICE_URI,
    session_db_kwargs=dict(expire=60 * 60 * 1),  # 1 hour TTL
    artifact_service_uri="",
    allow_origins=ALLOWED_ORIGINS,
    web=True,
)

# Add custom endpoints if needed
@app.get("/health")
async def get_health():
    return {"status": "ok"}

# Run the server
uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
```

### Direct RedisSessionService Usage

You can also use `RedisSessionService` directly without FastAPI:

```python
from google.adk_community.sessions.redis_session_service import RedisSessionService
from google.adk.runners import Runner
from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.agents import Agent
from google.genai import types

# Define your agent (example)
my_agent = Agent(
    model="gemini-2.5-flash",
    name="my_assistant",
    description="A helpful assistant",
    instruction="You are a helpful assistant.",
)

# Create Redis session service
redis_uri = "redis://localhost:6379"
session_service = RedisSessionService(uri=redis_uri)

# Create a session
APP_NAME = "weather_assistant"
USER_ID = "user_123"
SESSION_ID = "session_01"

try:
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
except AlreadyExistsError:
    # Session already exists, which is fine
    pass

# Use with runner
runner = Runner(
    agent=my_agent,  # Your agent instance
    app_name=APP_NAME,
    session_service=session_service
)

# Prepare user message
query = "What's the weather like today?"
user_message = types.Content(
    role="user",
    parts=[types.Part(text=query)]
)

# Run agent queries
async for event in runner.run_async(
    user_id=USER_ID,
    session_id=SESSION_ID,
    new_message=user_message,
):
    if event.is_final_response():
        print(event.content.parts[0].text)
```

## Sample Structure

```
redis_session_service/
├── main.py                        # FastAPI app with Redis session service
├── redis_service_agent/
│   ├── __init__.py                # Agent package initialization
│   ├── agent.py                   # Simple weather assistant agent
│   └── services.py                # Redis service registry setup
├── .env                           # Environment variables (create this)
└── README.md                      # This file
```

## Sample Agent

The sample agent (`redis_service_agent/agent.py`) is a simple weather assistant that includes:
- A `get_weather` tool that returns weather information for cities
- Basic agent configuration using Gemini 2.5 Flash

This is a minimal example to demonstrate Redis session persistence without complexity.

## Service Registration

The `services.py` file registers the Redis session service with the ADK service registry:

```python
from google.adk.cli.service_registry import get_service_registry
from google.adk_community.sessions.redis_session_service import RedisSessionService

def redis_session_factory(uri: str, **kwargs):
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop("agents_dir", None)
    return RedisSessionService(uri=uri, **kwargs_copy)

get_service_registry().register_session_service("redis", redis_session_factory)
```

This registration allows `get_fast_api_app` to automatically create a Redis session service when a `redis://` URI is provided.

## Interacting with the Sample

Once the server is running, you can:

1. **Use the Web Interface**: Navigate to `http://localhost:8080` in your browser to interact with the weather assistant through the web UI.

2. **Make API Calls**: Send requests to the API endpoints programmatically.

3. **Check Health**: Visit `http://localhost:8080/health` to verify the server is running.

Example queries to try:
- "What's the weather like in San Francisco and Tokyo?"
- "How's the weather in New York?"
- "Tell me about London's weather conditions"

All conversation history is stored in Redis with a 1-hour TTL, so you can have multi-turn conversations and the agent will remember the context.

## Configuration Options

### Redis URI Format

The `REDIS_URI` environment variable supports various Redis connection formats:

- `redis://localhost:6379` - Local Redis instance (default)
- `redis://username:password@host:port` - Redis with authentication
- `rediss://host:port` - Redis with TLS/SSL
- `redis://host:port/db_number` - Specify Redis database number

### FastAPI Configuration Options

When using `get_fast_api_app`, you can customize behavior:

```python
app = get_fast_api_app(
    agents_dir="./redis_service_agent",
    session_service_uri="redis://localhost:6379/0",
    session_db_kwargs=dict(
        expire=60 * 60 * 24,  # Session TTL in seconds (24 hours)
    ),
    artifact_service_uri="",  # Optional artifact storage URI
    allow_origins=["*"],  # CORS allowed origins
    web=True,  # Enable web interface
)
```

### RedisSessionService Options

When creating the `RedisSessionService` directly, you can customize behavior:

```python
session_service = RedisSessionService(
    uri="redis://localhost:6379",
    expire=3600,  # Session TTL in seconds (1 hour)
)
```

## Features

Redis Session Service provides:

- **Persistent session storage**: Conversation history survives application restarts
- **Multi-session support**: Handle multiple users and sessions simultaneously
- **Fast performance**: In-memory data structure store for quick access
- **Scalability**: Redis can be clustered for high-availability deployments
- **Simple setup**: Works with existing Redis infrastructure

## Troubleshooting

### Connection Issues

If you see connection errors:
1. Verify Redis is running: `redis-cli ping`
2. Check your `REDIS_URI` is correct
3. Ensure no firewall is blocking port 6379

### Session Already Exists

The sample handles `AlreadyExistsError` gracefully. If you want to start fresh:
```bash
redis-cli FLUSHDB
```

## Learn More

- [Redis Documentation](https://redis.io/docs/)
- [ADK Session Documentation](https://google.github.io/adk-docs)
- [ADK Community Repository](https://github.com/google/adk-python-community)
