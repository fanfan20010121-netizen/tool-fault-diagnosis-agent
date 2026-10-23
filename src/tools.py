"""工具路由：把 LLM 的 tool_call 分发到具体实现。"""
import pandas as pd

from . import config
from .ml_engine import ToolFaultEngine
from .rag_engine import ManualRetriever

_ml_engine = None
_retriever = None


def _get_ml_engine() -> ToolFaultEngine:
    global _ml_engine
    if _ml_engine is None:
        _ml_engine = ToolFaultEngine()
    return _ml_engine


def _get_retriever() -> ManualRetriever:
    global _retriever
    if _retriever is None:
        _retriever = ManualRetriever()
    return _retriever


def analyze_tool_image(image_path: str) -> dict:
    return _get_ml_engine().predict(image_path)


def search_manuals(query: str, top_k: int = 3) -> list:
    return _get_retriever().query(query, top_k=top_k)


def get_history_stats(days: int = 7) -> dict:
    csv_path = config.DATA_DIR / "sensor_data.csv"
    if not csv_path.exists():
        return {
            "note": "暂无历史数据文件 data/sensor_data.csv",
            "demo": True,
            "days": days,
        }
    df = pd.read_csv(csv_path)
    column = "prediction" if "prediction" in df.columns else "label"
    counts = df[column].value_counts().to_dict()
    total = int(sum(counts.values())) or 1
    return {
        "days": days,
        "total": total,
        "distribution": {k: round(v / total, 3) for k, v in counts.items()},
    }


def execute_tool(name: str, args: dict) -> dict | list:
    """按工具名分发；参数由 LLM 以 JSON 形式给出。"""
    if name == "analyze_tool_image":
        return analyze_tool_image(args["image_path"])
    if name == "search_manuals":
        return search_manuals(args["query"], int(args.get("top_k", 3)))
    if name == "get_history_stats":
        return get_history_stats(int(args.get("days", 7)))
    return {"error": f"未知工具：{name}"}
