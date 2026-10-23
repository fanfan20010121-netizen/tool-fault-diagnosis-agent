"""诊断 Agent：手写 function-calling 循环。

流程：用户输入 → LLM 决策 → 执行工具 → 结果回填 → 再决策 → 最终回答。
每一步都可观测，便于调试与面试讲解。
"""
import json
from typing import Callable

from .llm_client import chat
from .prompts import SYSTEM_PROMPT, TOOL_DEFINITIONS
from .tools import execute_tool

StepCallback = Callable[[str, dict, object], None]


class DiagnosisAgent:
    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def run(self, user_input: str, on_step: StepCallback | None = None) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(self.max_steps):
            message = chat(self.messages, tools=TOOL_DEFINITIONS)
            # 只回填协议要求的字段，避免把 reasoning_content 等私有字段发回接口
            assistant_message = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ]
            self.messages.append(assistant_message)

            if not message.tool_calls:
                return message.content or "（模型未返回内容）"

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                result = execute_tool(name, args)

                if on_step:
                    on_step(name, args, result)

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

        return "已达到最大推理步数，请把问题拆小一些再试。"
