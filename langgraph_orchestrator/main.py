from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, Field, ValidationError

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

# ----------------------------
# Types similar to src/query.ts
# ----------------------------

MessageType = Literal["user", "assistant", "progress"]

@dataclass
class AssistantContentBlock:
  type: Literal["text", "tool_use"]
  text: Optional[str] = None
  name: Optional[str] = None
  id: Optional[str] = None
  input: Optional[Dict[str, Any]] = None

@dataclass
class AssistantMessagePy:
  type: Literal["assistant"] = "assistant"
  content: List[AssistantContentBlock] = field(default_factory=list)

@dataclass
class UserMessagePy:
  type: Literal["user"] = "user"
  content: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ProgressMessagePy:
  type: Literal["progress"] = "progress"
  content: AssistantMessagePy = field(default_factory=AssistantMessagePy)

MessagePy = AssistantMessagePy | UserMessagePy | ProgressMessagePy

# ----------------------------
# Tool base and context
# ----------------------------

class ToolContext(BaseModel):
  safe_mode: bool = False
  abort_event: asyncio.Event | None = None
  agent_id: Optional[str] = None
  tools: Dict[str, "BaseTool"] = Field(default_factory=dict)

class ToolResult(BaseModel):
  type: Literal["result", "progress"]
  result_for_assistant: Optional[str] = None
  data: Optional[Any] = None
  progress_message: Optional[AssistantMessagePy] = None

class BaseTool(BaseModel):
  name: str
  read_only: bool = True
  concurrency_safe: bool = True

  class InputModel(BaseModel):
    pass

  class Config:
    arbitrary_types_allowed = True

  async def validate_input(self, input_data: dict, ctx: ToolContext) -> Tuple[bool, str]:
    try:
      self.InputModel.model_validate(input_data)
      return True, ""
    except ValidationError as ve:
      return False, str(ve)

  async def needs_permissions(self, input_data: dict) -> bool:
    return False

  async def call(self, input_data: dict, ctx: ToolContext) -> AsyncGenerator[ToolResult, None]:
    yield ToolResult(type="result", result_for_assistant="")

# ----------------------------
# Example tools
# ----------------------------

class ThinkToolPy(BaseTool):
  name: str = "Think"
  read_only: bool = True
  concurrency_safe: bool = True

  class InputModel(BaseModel):
    thought: str

  async def call(self, input_data: dict, ctx: ToolContext):
    i = self.InputModel.model_validate(input_data)
    yield ToolResult(type="result", result_for_assistant="Your thought has been logged.", data={"thought": i.thought})

class EchoToolPy(BaseTool):
  name: str = "Echo"
  read_only: bool = True
  concurrency_safe: bool = True

  class InputModel(BaseModel):
    text: str

  async def call(self, input_data: dict, ctx: ToolContext):
    i = self.InputModel.model_validate(input_data)
    yield ToolResult(type="result", result_for_assistant=i.text, data={"echo": i.text})

# ----------------------------
# Task prompt
# ----------------------------

def get_task_prompt(available_tool_names: List[str], current_task_model: str) -> str:
  return (
    f"Launch a new agent that has access to the following tools: {', '.join(available_tool_names)}.\n"
    "When you are searching for a keyword or file and are not confident you will find the right match in the first few tries, use the Task tool to perform the search for you.\n\n"
    f"Default task model: {current_task_model}\n"
    "Usage notes:\n"
    "1. Launch multiple agents concurrently whenever possible by issuing multiple tool uses in a single message.\n"
    "2. The agent returns a single final message. Summarize results concisely for the user.\n"
    "3. Each agent invocation is stateless. Provide a detailed prompt including exactly what to return.\n"
    "4. The agent's outputs should generally be trusted.\n"
    "5. Specify whether to write code or only research.\n"
  )

# ----------------------------
# LLM helpers
# ----------------------------

def build_system_prompt_with_context(base_prompt: List[str], context: Dict[str, str]) -> str:
  parts = list(base_prompt)
  if context:
    parts.append("\n---\n# Project Context\n")
    for k, v in context.items():
      parts.append(f"<context name=\"{k}\">{v}</context>")
    parts.append("\n---\n")
  return "\n".join(parts)

def make_langchain_tools_for_llm(tools: Dict[str, BaseTool]) -> List[StructuredTool]:
  wrappers: List[StructuredTool] = []
  for t in tools.values():
    def _dummy(**kwargs):
      return "ok"  # Only for tool selection; real execution is handled by our pipeline
    wrappers.append(
      StructuredTool.from_function(
        name=t.name,
        description=f"Orchestrator tool: {t.name}",
        func=_dummy,
        args_schema=t.InputModel,
      )
    )
  return wrappers

def extract_tool_uses(ai_message: AIMessage) -> List[Dict[str, Any]]:
  calls = []
  tool_calls = ai_message.tool_calls or []
  for i, tc in enumerate(tool_calls):
    calls.append({
      "id": f"tc_{i}",
      "name": tc.get("name"),
      "input": tc.get("args", {}) or {},
    })
  return calls

# ----------------------------
# Permissions
# ----------------------------

PermissionFn = Callable[[BaseTool, dict, ToolContext, AssistantMessagePy], Awaitable[bool]]

async def default_can_use_tool(tool: BaseTool, input_data: dict, ctx: ToolContext, last_assistant: AssistantMessagePy) -> bool:
  if await tool.needs_permissions(input_data):
    return True
  return True

# ----------------------------
# Tool execution
# ----------------------------

async def run_tool_use(
  tool_use: Dict[str, Any],
  tools: Dict[str, BaseTool],
  ctx: ToolContext,
  can_use_tool: PermissionFn,
  assistant_msg: AssistantMessagePy,
) -> List[MessagePy]:
  tool_name = tool_use["name"]
  tool = tools.get(tool_name)
  if not tool:
    return [UserMessagePy(content=[{"type": "tool_result", "tool_use_id": tool_use["id"], "content": f"Error: No such tool {tool_name}", "is_error": True}])]

  ok, err = await tool.validate_input(tool_use.get("input", {}), ctx)
  if not ok:
    return [UserMessagePy(content=[{"type": "tool_result", "tool_use_id": tool_use["id"], "content": f"InputValidationError: {err}", "is_error": True}])]

  allowed = await can_use_tool(tool, tool_use.get("input", {}), ctx, assistant_msg)
  if not allowed:
    return [UserMessagePy(content=[{"type": "tool_result", "tool_use_id": tool_use["id"], "content": "Permission denied", "is_error": True}])]

  out_messages: List[MessagePy] = []
  async for res in tool.call(tool_use.get("input", {}), ctx):
    if res.type == "progress" and res.progress_message:
      out_messages.append(ProgressMessagePy(content=res.progress_message))
    elif res.type == "result":
      out_messages.append(UserMessagePy(content=[{"type": "tool_result", "tool_use_id": tool_use["id"], "content": res.result_for_assistant or "", "is_error": False}]))

  if not out_messages:
    out_messages.append(UserMessagePy(content=[{"type": "tool_result", "tool_use_id": tool_use["id"], "content": "", "is_error": False}]))

  return out_messages

async def run_tools(
  tool_uses: List[Dict[str, Any]],
  tools: Dict[str, BaseTool],
  ctx: ToolContext,
  can_use_tool: PermissionFn,
  assistant_msg: AssistantMessagePy,
) -> List[MessagePy]:
  can_run_concurrent = all(tools.get(tu["name"]) and tools[tu["name"]].read_only for tu in tool_uses)
  if can_run_concurrent:
    results = await asyncio.gather(*[
      run_tool_use(tu, tools, ctx, can_use_tool, assistant_msg) for tu in tool_uses
    ])
    return [m for batch in results for m in batch]
  else:
    out: List[MessagePy] = []
    for tu in tool_uses:
      out.extend(await run_tool_use(tu, tools, ctx, can_use_tool, assistant_msg))
    return out

# ----------------------------
# LangGraph state and nodes
# ----------------------------

@dataclass
class QueryState:
  messages: List[MessagePy]
  system_prompt_blocks: List[str]
  context: Dict[str, str]
  tools: Dict[str, BaseTool]
  can_use_tool: PermissionFn
  tool_ctx: ToolContext
  last_tool_uses: List[Dict[str, Any]] = field(default_factory=list)

def state_to_langchain_messages(state: QueryState) -> List[Any]:
  lc_messages: List[Any] = []
  system = build_system_prompt_with_context(state.system_prompt_blocks, state.context)
  lc_messages.append(SystemMessage(content=system))
  for m in state.messages:
    if m.type == "user":
      text = ""
      for c in m.content:
        if c.get("type") == "tool_result":
          text += f"\n[tool_result id={c.get('tool_use_id')}] {c.get('content')}"
        elif c.get("type") == "text":
          text += c.get("text", "")
        else:
          text += str(c)
      lc_messages.append(HumanMessage(content=text or ""))
    elif m.type == "assistant":
      parts = []
      for b in m.content:
        if b.type == "text" and b.text:
          parts.append(b.text)
      lc_messages.append(AIMessage(content="\n".join(parts)))
  return lc_messages

async def llm_node(state: QueryState, llm: ChatOpenAI) -> QueryState:
  lc_messages = state_to_langchain_messages(state)
  bound = llm.bind_tools(make_langchain_tools_for_llm(state.tools))
  ai = await bound.ainvoke(lc_messages)
  tool_uses = extract_tool_uses(ai)
  blocks: List[AssistantContentBlock] = []
  if ai.content:
    blocks.append(AssistantContentBlock(type="text", text=str(ai.content)))
  for tc in tool_uses:
    blocks.append(AssistantContentBlock(type="tool_use", name=tc["name"], id=tc["id"], input=tc["input"]))
  assistant = AssistantMessagePy(content=blocks)
  return QueryState(
    messages=state.messages + [assistant],
    system_prompt_blocks=state.system_prompt_blocks,
    context=state.context,
    tools=state.tools,
    can_use_tool=state.can_use_tool,
    tool_ctx=state.tool_ctx,
    last_tool_uses=tool_uses,
  )

async def tools_node(state: QueryState) -> QueryState:
  # Get the last assistant message
  assistant_msgs = [m for m in state.messages if isinstance(m, AssistantMessagePy)]
  assistant_msg = assistant_msgs[-1] if assistant_msgs else AssistantMessagePy()
  results = await run_tools(state.last_tool_uses, state.tools, state.tool_ctx, state.can_use_tool, assistant_msg)
  return QueryState(
    messages=state.messages + results,
    system_prompt_blocks=state.system_prompt_blocks,
    context=state.context,
    tools=state.tools,
    can_use_tool=state.can_use_tool,
    tool_ctx=state.tool_ctx,
    last_tool_uses=[],
  )

def router_decision(state: QueryState) -> Literal["tools", "end"]:
  last_ai: Optional[AssistantMessagePy] = None
  for m in reversed(state.messages):
    if isinstance(m, AssistantMessagePy):
      last_ai = m
      break
  if not last_ai:
    return "end"
  has_tool = any(b.type == "tool_use" for b in last_ai.content)
  return "tools" if has_tool else "end"

# ----------------------------
# Task tool (sub-agent)
# ----------------------------

class TaskToolPy(BaseTool):
  name: str = "Task"
  read_only: bool = True
  concurrency_safe: bool = True

  class InputModel(BaseModel):
    description: str
    prompt: str
    model_name: Optional[str] = None

  def __init__(self, available_tools: Dict[str, BaseTool], default_task_model: str = "gpt-4o-mini"):
    super().__init__()
    self.available_tools = {k: v for k, v in available_tools.items() if k != self.name}
    self.default_task_model = default_task_model

  async def call(self, input_data: dict, ctx: ToolContext):
    args = self.InputModel.model_validate(input_data)
    # Initial progress
    yield ToolResult(type="progress", progress_message=AssistantMessagePy(content=[AssistantContentBlock(type="text", text="Initializing…")]))

    system_blocks = [get_task_prompt(list(self.available_tools.keys()), args.model_name or self.default_task_model)]

    sub_state = QueryState(
      messages=[UserMessagePy(content=[{"type": "text", "text": args.prompt}])],
      system_prompt_blocks=system_blocks,
      context={},
      tools=self.available_tools,
      can_use_tool=default_can_use_tool,
      tool_ctx=ctx,
    )
    sub_llm = ChatOpenAI(model=args.model_name or self.default_task_model, temperature=0)

    # Build subgraph
    g = StateGraph(QueryState)
    g.add_node("llm", lambda s: llm_node(s, sub_llm))
    g.add_node("tools", tools_node)
    g.set_entry_point("llm")
    g.add_conditional_edges("llm", router_decision, {"tools": "tools", "end": "__end__"})
    g.add_edge("tools", "llm")
    app = g.compile()

    final_texts: List[str] = []
    async for out in app.astream(sub_state):
      # Optionally, stream progress up from sub-agent
      yield ToolResult(type="progress", progress_message=AssistantMessagePy(content=[AssistantContentBlock(type="text", text="[Task] step")]))
      # Capture last assistant text
      outputs: QueryState = out
      if outputs.messages:
        last = outputs.messages[-1]
        if isinstance(last, AssistantMessagePy):
          for b in last.content:
            if b.type == "text" and b.text:
              final_texts.append(b.text)

    yield ToolResult(type="result", result_for_assistant="\n".join(final_texts), data=final_texts)

# ----------------------------
# Main
# ----------------------------

async def main():
  # Tool registry
  tools: Dict[str, BaseTool] = {
    "Think": ThinkToolPy(),
    "Echo": EchoToolPy(),
  }
  tools["Task"] = TaskToolPy(available_tools=tools, default_task_model="gpt-4o-mini")

  ctx = ToolContext(safe_mode=True, tools=tools)
  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

  # Initial conversation
  initial_state = QueryState(
    messages=[UserMessagePy(content=[{"type": "text", "text": "Use Task to research: outline how to configure logging; then echo 'done'."}])],
    system_prompt_blocks=["You are a helpful assistant."],
    context={"gitStatus": "branch: main", "readme": "Project readme..."},
    tools=tools,
    can_use_tool=default_can_use_tool,
    tool_ctx=ctx,
  )

  # Build graph
  g = StateGraph(QueryState)
  g.add_node("llm", lambda s: llm_node(s, llm))
  g.add_node("tools", tools_node)
  g.set_entry_point("llm")
  g.add_conditional_edges("llm", router_decision, {"tools": "tools", "end": "__end__"})
  g.add_edge("tools", "llm")
  app = g.compile()

  async for state in app.astream(initial_state):
    out: QueryState = state
    last = out.messages[-1]
    if isinstance(last, AssistantMessagePy):
      texts = [b.text for b in last.content if b.type == "text" and b.text]
      print("[assistant]", " ".join(texts))
    elif isinstance(last, UserMessagePy):
      print("[tool_result]", last.content)
    else:
      print("[progress]")

if __name__ == "__main__":
  asyncio.run(main())