"""
基于 LangGraph 的多工具 Agent。

图结构：
    START -> call_model -> (有 tool_use?) -> execute_tools -> call_model -> ...
                                          -> END
"""
import os
import operator
import subprocess
from pathlib import Path
from typing import Annotated, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv(override=True)

# 使用自定义 base_url 时，移除可能冲突的 auth token
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool for multi-step work.
Keep exactly one step in_progress when a task has multiple steps.
Refresh the plan as work advances. Prefer tools over prose.
Use the task tool to delegate exploration or subtasks to a subagent with fresh context."""

SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."

# 多少轮不更新 todo 后触发提醒
PLAN_REMINDER_INTERVAL = 3

# 禁止执行的危险命令关键词
DANGEROUS = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]


# ---------------------------------------------------------------------------
# 待办计划
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class PlanItem:
    """单条待办事项。"""
    content: str                # 任务描述
    status: str = "pending"     # pending | in_progress | completed
    active_form: str = ""       # 进行中时的动词短语，如 "Reading the failing test"


@dataclass
class PlanningState:
    """当前会话的整体计划状态。"""
    items: list[PlanItem] = field(default_factory=list)
    rounds_since_update: int = 0  # 距上次调用 todo 工具的轮次数


class TodoManager:
    """管理会话计划，供 todo 工具调用。"""

    def __init__(self):
        self.state = PlanningState()

    def update(self, items: list) -> str:
        """用模型提供的新列表完整替换当前计划，返回渲染结果。"""
        if len(items) > 12:
            raise ValueError("Keep the session plan short (max 12 items)")

        normalized = []
        in_progress_count = 0
        for index, raw in enumerate(items):
            content = str(raw.get("content", "")).strip()
            status = str(raw.get("status", "pending")).lower()
            active_form = str(raw.get("activeForm", "")).strip()

            if not content:
                raise ValueError(f"Item {index}: content required")
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(f"Item {index}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1

            normalized.append(PlanItem(content=content, status=status, active_form=active_form))

        if in_progress_count > 1:
            raise ValueError("Only one plan item can be in_progress")

        self.state.items = normalized
        self.state.rounds_since_update = 0
        return self.render()

    def note_round_without_update(self) -> None:
        """每轮没有调用 todo 工具时调用，累计计数。"""
        self.state.rounds_since_update += 1

    def reminder(self) -> str | None:
        """超过提醒间隔时返回提醒文本，否则返回 None。"""
        if not self.state.items:
            return None
        if self.state.rounds_since_update < PLAN_REMINDER_INTERVAL:
            return None
        return "<reminder>Refresh your current plan before continuing.</reminder>"

    def render(self) -> str:
        """将计划渲染为可读文本。"""
        if not self.state.items:
            return "No session plan yet."
        lines = []
        for item in self.state.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item.status]
            line = f"{marker} {item.content}"
            if item.status == "in_progress" and item.active_form:
                line += f" ({item.active_form})"
            lines.append(line)
        completed = sum(1 for i in self.state.items if i.status == "completed")
        lines.append(f"\n({completed}/{len(self.state.items)} completed)")
        return "\n".join(lines)


# 全局单例，整个会话共享一个计划
TODO = TodoManager()


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------

def safe_path(p: str) -> Path:
    """将相对路径解析为绝对路径，并确保不会逃出工作目录。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"路径超出工作目录: {p}")
    return path


def run_bash(command: str) -> str:
    """执行 shell 命令，返回 stdout + stderr。"""
    if any(d in command for d in DANGEROUS):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def run_read(path: str, limit: int = None) -> str:
    """读取文件内容，limit 指定最多返回多少行。"""
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入文件，父目录不存在时自动创建。"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """在文件中精确替换第一处匹配的文本。"""
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# 子代理
# ---------------------------------------------------------------------------

# 子代理可用工具（不含 todo 和 task，防止递归生成计划或嵌套子代理）
CHILD_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]

# 子代理工具分发表（与父代理共用同名处理函数，但不含 todo/task）
_CHILD_TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}


def run_subagent(prompt: str) -> str:
    """以全新上下文运行子代理，完成后仅返回最终文本摘要给父代理。

    子代理与父代理共享文件系统，但拥有独立的空白消息列表，
    完成后父代理上下文保持整洁——子代理的中间过程全部丢弃。
    """
    sub_messages = [{"role": "user", "content": prompt}]  # 全新上下文
    response = None
    for _ in range(30):  # 安全上限
        response = client.messages.create(
            model=MODEL,
            system=SUBAGENT_SYSTEM,
            messages=sub_messages,
            tools=CHILD_TOOLS,
            max_tokens=8000,
        )
        sub_messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = _CHILD_TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output)[:50000],
                })
        sub_messages.append({"role": "user", "content": results})

    # 仅将最终文本返回给父代理——子代理上下文随即丢弃
    if response is None:
        return "(no response)"
    return "".join(
        b.text for b in response.content if hasattr(b, "text")
    ) or "(no summary)"


# ---------------------------------------------------------------------------
# 工具名 -> 处理函数的分发表，新增工具只需在此处添加一行
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),
    # task 由 execute_tools 单独处理，此处仅占位保持 handler 存在
    "task":       lambda **kw: run_subagent(kw["prompt"]),
}

# 传给模型的工具定义（JSON Schema 格式）
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "task", "description": "Spawn a subagent with fresh context to handle exploration or subtasks. The subagent shares the filesystem but not conversation history, and returns only a summary.",
     "input_schema": {"type": "object", "properties": {
         "prompt": {"type": "string", "description": "Full instructions for the subagent."},
         "description": {"type": "string", "description": "Short description of the subtask (shown in logs)."},
     }, "required": ["prompt"]}},
    {"name": "todo", "description": "Rewrite the current session plan for multi-step work.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {
         "type": "object",
         "properties": {
             "content":    {"type": "string"},
             "status":     {"type": "string", "enum": ["pending", "in_progress", "completed"]},
             "activeForm": {"type": "string", "description": "Optional present-continuous label."},
         },
         "required": ["content", "status"],
     }}}, "required": ["items"]}},
]


# ---------------------------------------------------------------------------
# 消息归一化
# ---------------------------------------------------------------------------

def _block_to_dict(block) -> dict | None:
    """将 SDK 内容块对象转换为普通 dict，方便后续处理。"""
    if isinstance(block, dict):
        # 过滤掉以 _ 开头的内部字段
        return {k: v for k, v in block.items() if not k.startswith("_")}
    if hasattr(block, "type"):
        if block.type == "text":
            return {"type": "text", "text": block.text}
        if block.type == "tool_use":
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return None


def normalize_messages(messages: list) -> list:
    """在每次调用 API 前清理消息列表，做三件事：

    1. 将 SDK 对象统一转为普通 dict
    2. 为没有对应 tool_result 的孤儿 tool_use 块插入 (cancelled) 占位
    3. 合并连续的同角色消息（API 要求 user/assistant 严格交替）
    """
    # 第一步：统一转为 dict
    cleaned = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            cleaned.append({"role": msg["role"], "content": content})
        elif isinstance(content, list):
            blocks = [b for b in (_block_to_dict(x) for x in content) if b is not None]
            cleaned.append({"role": msg["role"], "content": blocks})
        else:
            cleaned.append({"role": msg["role"], "content": str(content)})

    # 第二步：找出所有已有的 tool_result id
    existing_results = {
        block.get("tool_use_id")
        for msg in cleaned
        for block in (msg["content"] if isinstance(msg.get("content"), list) else [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    }
    # 为孤儿 tool_use 插入占位结果，避免 API 报错
    for msg in list(cleaned):
        if msg["role"] != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if block.get("id") not in existing_results:
                    cleaned.append({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": block["id"], "content": "(cancelled)"}
                    ]})

    # 第三步：合并连续同角色消息
    if not cleaned:
        return cleaned
    merged = [cleaned[0]]
    for msg in cleaned[1:]:
        if msg["role"] == merged[-1]["role"]:
            prev = merged[-1]
            prev_c = prev["content"] if isinstance(prev["content"], list) \
                else [{"type": "text", "text": str(prev["content"])}]
            curr_c = msg["content"] if isinstance(msg["content"], list) \
                else [{"type": "text", "text": str(msg["content"])}]
            prev["content"] = prev_c + curr_c
        else:
            merged.append(msg)
    return merged


# ---------------------------------------------------------------------------
# LangGraph 状态
# ---------------------------------------------------------------------------

class State(TypedDict):
    # operator.add 作为 reducer：每个节点返回的新消息自动追加到列表末尾
    messages: Annotated[list, operator.add]
    turn_count: int  # 记录已执行的轮次


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
        # 打印工具调用摘要，方便调试
        print(f"\033[33m> {block.name}: {str(output)[:200]}\033[0m")
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": str(output),
        })
        if block.name == "todo":
            used_todo = True

    if used_todo:
        # todo 工具已在 update() 中重置了计数，此处无需重复操作
        pass
    else:
        TODO.note_round_without_update()
        reminder = TODO.reminder()
        if reminder:
            # 将提醒作为纯文本插到结果列表最前面，让模型优先看到
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
    builder.add_conditional_edges("call_model", should_continue)  # 有工具调用则循环，否则结束
    builder.add_edge("execute_tools", "call_model")               # 工具执行完毕后回到模型
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
