import re
from dataclasses import dataclass
from pathlib import Path

from config import SKILLS_DIR


@dataclass
class SkillManifest:
    name: str
    description: str
    path: Path


@dataclass
class SkillDocument:
    manifest: SkillManifest
    body: str


class SkillRegistry:
    """扫描 skills/ 目录，按需加载技能全文。

    两层模型：
    1. 启动时只读取各技能的 name/description（目录），注入系统提示。
    2. 模型调用 load_skill 时，才将对应技能的完整正文加载进上下文。
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.documents: dict[str, SkillDocument] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.skills_dir.exists():
            return
        for path in sorted(self.skills_dir.rglob("SKILL.md")):
            meta, body = self._parse_frontmatter(path.read_text())
            name = meta.get("name", path.parent.name)
            description = meta.get("description", "No description")
            manifest = SkillManifest(name=name, description=description, path=path)
            self.documents[name] = SkillDocument(manifest=manifest, body=body.strip())

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        meta = {}
        for line in match.group(1).strip().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
        return meta, match.group(2)

    def describe_available(self) -> str:
        """返回供系统提示使用的技能目录（单行格式）。"""
        if not self.documents:
            return "(no skills available)"
        return "\n".join(
            f"- {doc.manifest.name}: {doc.manifest.description}"
            for doc in self.documents.values()
        )

    def load_full_text(self, name: str) -> str:
        """按名称加载技能完整正文，返回给模型。"""
        document = self.documents.get(name)
        if not document:
            known = ", ".join(sorted(self.documents)) or "(none)"
            return f"Error: Unknown skill '{name}'. Available: {known}"
        return (
            f'<skill name="{document.manifest.name}">\n'
            f"{document.body}\n"
            "</skill>"
        )


# 全局单例，启动时扫描一次
SKILL_REGISTRY = SkillRegistry(SKILLS_DIR)
