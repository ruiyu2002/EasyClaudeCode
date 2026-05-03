def _block_to_dict(block) -> dict | None:
    """将 SDK 内容块对象转换为普通 dict，方便后续处理。"""
    if isinstance(block, dict):
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
