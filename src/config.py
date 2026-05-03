import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

# 使用自定义 base_url 时，移除可能冲突的 auth token
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
SKILLS_DIR = WORKDIR / "skills"

# SYSTEM 在 graph.py 中组装（需要先加载 SkillRegistry 的技能目录）

SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."

# 多少轮不更新 todo 后触发提醒
PLAN_REMINDER_INTERVAL = 3

# 禁止执行的危险命令关键词
DANGEROUS = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
