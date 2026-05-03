from dataclasses import dataclass, field

from config import PLAN_REMINDER_INTERVAL


@dataclass
class PlanItem:
    """单条待办事项。"""
    content: str
    status: str = "pending"   # pending | in_progress | completed
    active_form: str = ""     # 进行中时的动词短语，如 "Reading the failing test"


@dataclass
class PlanningState:
    """当前会话的整体计划状态。"""
    items: list[PlanItem] = field(default_factory=list)
    rounds_since_update: int = 0


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
