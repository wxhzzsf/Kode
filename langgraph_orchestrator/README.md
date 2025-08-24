# LangGraph Orchestrator

This folder contains a runnable Python implementation of the orchestration loop found in `src/query.ts`, adapted to Python and LangGraph. It includes:

- A main orchestration loop that:
  - Builds a system prompt with context
  - Calls the LLM
  - Detects tool calls
  - Executes tools serially or concurrently based on read-only status
  - Recursively continues until no more tools are requested
- A `Task` tool that launches a sub-agent with its own prompt and toolset
- Example tools: `Think` and `Echo`

## Requirements

- Python 3.10+
- An OpenAI API key set as `OPENAI_API_KEY` in your environment

Install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
export OPENAI_API_KEY=...  # required for ChatOpenAI
python main.py
```

You should see assistant messages, any tool_result messages, and progress lines printed to the console.

## Notes

- The LLM tool-calling is enabled via LangChain `bind_tools` using structured schemas derived from the tool input models. Tool execution is handled by our orchestrator (not by LangChain ToolNode), to mirror the original `query()` pipeline's validation, permission checks, and concurrency gating.
- `Task` uses a subgraph with its own model. Configure the model by editing `default_task_model` in `main.py` or by passing `model_name` via a `Task` call in the conversation.
- This is a minimal but faithful reproduction of the main orchestration logic and tool interfaces so it’s easy to extend with additional tools (file read/write, bash, grep, etc.).