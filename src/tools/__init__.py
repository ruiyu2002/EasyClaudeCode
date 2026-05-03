from tools.implementations import run_bash, run_read, run_write, run_edit
from tools.todo import TODO
from skills import SKILL_REGISTRY


def _run_subagent(**kw):
    # 延迟导入，避免与 subagent.py → tools 之间的循环依赖
    from subagent import run_subagent
    return run_subagent(kw["prompt"])

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
    {"name": "load_skill", "description": "Load the full body of a named skill into the current context.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
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

# 工具名 -> 处理函数的分发表，新增工具只需在此处添加一行
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "load_skill": lambda **kw: SKILL_REGISTRY.load_full_text(kw["name"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),
    "task":       _run_subagent,
}
