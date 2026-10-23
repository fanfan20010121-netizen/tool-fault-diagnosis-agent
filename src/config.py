"""全局配置：路径与环境变量。"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Embedding 模型已缓存到本地，离线加载，避免 HuggingFace 连接重试拖慢启动
# （新机器首次下载模型时，把 .env 里加一行 HF_HUB_OFFLINE=0 即可）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

DATA_DIR = ROOT / "data"
MANUAL_DIR = DATA_DIR / "manuals"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
CHECKPOINT_DIR = ROOT / "checkpoints"

for _d in (MANUAL_DIR, UPLOAD_DIR, INDEX_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
