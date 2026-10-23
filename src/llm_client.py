"""DeepSeek API 封装（OpenAI 兼容协议）。"""
import httpx
from openai import OpenAI

from . import config

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY，请先在 .env 中配置")
        _client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            # DeepSeek 是国内直连服务，不走系统代理，避免被代理拦截导致连接重置
            http_client=httpx.Client(trust_env=False, timeout=60.0),
        )
    return _client


def chat(messages: list, tools: list | None = None, temperature: float = 0.0):
    """调用 LLM，返回 assistant message（可能带 tool_calls）。"""
    kwargs = {
        "model": config.MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
    resp = get_client().chat.completions.create(**kwargs)
    return resp.choices[0].message
