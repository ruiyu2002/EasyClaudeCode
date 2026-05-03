from config import client, MODEL, SUBAGENT_SYSTEM
from tools.implementations import run_bash, run_read, run_write, run_edit

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

_HANDLERS = {
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
    sub_messages = [{"role": "user", "content": prompt}]
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
                handler = _HANDLERS.get(block.name)
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
