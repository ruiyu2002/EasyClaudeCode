"""
基于 LangGraph 的多工具 Agent 图。

图结构：
    START -> call_model -> (有 tool_use?) -> execute_tools -> call_model -> ...
                                          -> END
"""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from config import client, MODEL, SYSTEM
from normalizer import normalize_messages
from subagent import run_subagent
from tools import TOOLS, TOOL_HANDLERS
from tools.todo import TODO


# ---------------------------------------------------------------------------
# LangGraph 状态
# ---------------------------------------------------------------------------

class State(TypedDict):
    # operator.add 作为 reducer：每个节点返回的新消息自动追加到列表末尾
    messages: Annotated[list, operator.add]
    turn_count: int


# ---------------------------------------------------------------------------
# 图节点
# ---------------------------------------------------------------------------

def call_model(state: State) -> dict:
    """调用模型，将回复追加到消息列表。"""
    response = client.messages.create(
        model=MODEL,
        system=SYSTEM,
        messages=normalize_messages(state["messages"]),
        tools=TOOLS,
        max_tokens=8000,
    )
    return {
        "messages": [{"role": "assistant", "content": response.content}],
        "turn_count": state["turn_count"] + 1,
    }


def execute_tools(state: State) -> dict:
    """执行模型请求的所有工具调用，将结果追加为 tool_result 消息。

    额外逻辑：
    - 若本轮调用了 todo 工具，重置计划轮次计数
    - 否则累加计数，超过阈值时在结果前插入提醒文本
    """
    last = state["messages"][-1]
    results = []
    used_todo = False

    for block in last["content"]:
        if getattr(block, "type", None) != "tool_use":
            continue
        if block.name == "task":
            # 子代理：打印描述并在独立上下文中运行
            desc = block.input.get("description", "subtask")
            prompt = block.input.get("prompt", "")
            print(f"\033[35m> task ({desc}): {prompt[:80]}\033[0m")
            try:
                output = run_subagent(prompt)
            except Exception as e:
                output = f"Error: {e}"
        else:
            handler = TOOL_HANDLERS.get(block.name)
            try:
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
            except Exception as e:
                output = f"Error: {e}"
        print(f"\033[33m> {block.name}: {str(output)[:200]}\033[0m")
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": str(output),
        })
        if block.name == "todo":
            used_todo = True

    if not used_todo:
        TODO.note_round_without_update()
        reminder = TODO.reminder()
        if reminder:
            results.insert(0, {"type": "text", "text": reminder})

    return {
        "messages": [{"role": "user", "content": results}],
    }


# ---------------------------------------------------------------------------
# 路由函数
# ---------------------------------------------------------------------------

def should_continue(state: State) -> str:
    """判断模型是否还有工具调用需要执行，决定下一步走向。"""
    last = state["messages"][-1]
    content = last.get("content", [])
    if any(getattr(b, "type", None) == "tool_use" for b in content):
        return "execute_tools"
    return END


# ---------------------------------------------------------------------------
# 构建图
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(State)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tools", execute_tools)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue)
    builder.add_edge("execute_tools", "call_model")
    return builder.compile()


graph = build_graph()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def extract_text(messages: list) -> str:
    """从最后一条 assistant 消息中提取纯文本内容。"""
    if not messages:
        return ""
    content = messages[-1].get("content", [])
    if not isinstance(content, list):
        return ""
    return "\n".join(
        getattr(b, "text", "") for b in content if getattr(b, "text", "")
    ).strip()
