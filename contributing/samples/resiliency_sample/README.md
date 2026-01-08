# AgentTool Resilience: Timeout, Retry, and Redirect Patterns

This sample demonstrates how to handle failures, timeouts, and partial results from downstream agents in multi-agent workflows using ADK.


## Prerequisites

- Python 3.9+ (Python 3.11+ recommended)
- Google API key for the agent

## Setup

### 1. Clone the Repository

First, clone the `adk-python-community` repository to get the sample code:

```bash
git clone https://github.com/google/adk-python-community.git
cd adk-python-community
```

### 2. Install Dependencies

Navigate to the sample directory and install the required package:

```bash
cd contributing/samples/resiliency_sample
pip install google-adk
```

**Optional:** If you want to use a `.env` file for environment variables, also install:

```bash
pip install python-dotenv
```

**Note:** This sample only requires `google-adk` (core ADK package). You don't need to install `google-adk-community` for this sample.

### 3. Configure Environment Variables

Create a `.env` file in the `contributing/samples/resiliency_sample` directory:

```bash
# Required: Google API key for the agent
GOOGLE_API_KEY=your-google-api-key
```

## Usage

### Running the Test Example

This sample includes a test script that demonstrates various resilience patterns using the `test_helpers.py` utilities. Run the test example:

```bash
python test_example.py
```

This will run three test scenarios:
1. **Normal Operation**: Tests the coordinator agent with a simple query
2. **Timeout Scenario**: Demonstrates timeout handling using `timeout_test_agent`
3. **Failure Scenario**: Demonstrates error handling using `failure_test_agent`

### Using the Sample Programmatically

You can also use the sample agents directly in your own code:

```python
import asyncio
from google.adk.runners import Runner
from agent import coordinator_agent

async def main():
    runner = Runner(
        app_name="my_app",
        agent=coordinator_agent,
    )
    
    response = await runner.run_async("What is quantum computing?")
    print(response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

### Using Test Helpers

The `test_helpers.py` module provides utilities for testing resilience patterns:

- `timeout_test_agent`: An agent that uses a tool that simulates timeouts
- `failure_test_agent`: An agent that uses a tool that always fails
- `TimeoutSimulatorTool`: A tool that sleeps to trigger timeout scenarios
- `FailureSimulatorTool`: A tool that raises exceptions to test error handling

You can use these in your own tests to verify resilience patterns work correctly.

## Sample Structure

```
resiliency_sample/
├── agent.py               # Agent definitions with resilience patterns
├── test_helpers.py        # Test utilities for simulating timeouts and failures
├── test_example.py        # Example script demonstrating how to test the sample
├── __Init__.py            # Package initialization
└── README.md              # This file
```

## Sample Agent

The sample agent (`agent.py`) includes:
- `coordinator_agent` - Routes requests and handles errors
- `research_agent_primary` - Primary agent with timeout protection (default: 30s)
- `research_agent_fallback` - Fallback agent with longer timeout (60s)
- `error_recovery_agent` - Analyzes failures and provides recommendations

## Sample Queries

### Simple Query
- "What is quantum computing?"

### Complex Query
- "Research quantum computing applications in healthcare, finance, cryptography, logistics, weather prediction, drug discovery, and machine learning. For each domain, provide: historical context, current state-of-the-art, technical challenges, recent breakthroughs in 2024, comparison with classical approaches, economic impact, and future roadmap for the next 10 years."

### Testing Custom Scenarios

You can modify `test_example.py` to test custom scenarios:

1. **Test with different timeout values**: Modify the `timeout` parameter in `TimeoutAgentTool`
2. **Test with different failure modes**: Use different agents from `test_helpers.py`
3. **Test with your own agents**: Create custom test agents and wrap them with `TimeoutAgentTool`

Example: Testing with a custom timeout

```python
from test_helpers import timeout_test_agent, TimeoutAgentTool
from agent import Agent

# Create coordinator with custom timeout
test_coordinator = Agent(
    name='test_coordinator',
    model='gemini-2.5-flash-lite',
    tools=[
        TimeoutAgentTool(
            agent=timeout_test_agent,
            timeout=3.0,  # Custom timeout value
        ),
    ],
)
```

## Features Demonstrated

- **Timeout Protection**: Custom `TimeoutAgentTool` wrapper adds timeout handling to sub-agents
- **Automatic Retry**: `ReflectAndRetryToolPlugin` handles retries with structured guidance
- **Dynamic Fallback**: Coordinator agent routes to alternative agents when primary fails
- **Error Recovery**: Specialized agent provides user-friendly error analysis

## Expected Behavior

1. **Normal Operation**: Primary agent handles the query successfully
2. **Timeout Scenario**: Primary times out → Fallback agent is automatically tried
3. **Failure Scenario**: Primary fails → Retry → Fallback → Error recovery agent provides guidance