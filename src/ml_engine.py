"""刀具磨损分类引擎。

默认 DEMO 模式：没有权重时返回模拟结果，保证 Agent 全链路可运行。
接入真实模型：
  1. 把 EfficientNet-B2 网络定义放到 src/efficientnet_b2.py，暴露 build_model(num_classes)；
  2. 权重放到 checkpoints/efficientnet_b2.pt。
"""
from pathlib import Path

from . import config

CLASSES = ["初期磨损", "正常磨损", "急剧磨损"]


class ToolFaultEngine:
    def __init__(self, checkpoint: str | None = None):
        self.checkpoint = Path(checkpoint) if checkpoint else config.CHECKPOINT_DIR / "efficientnet_b2.pt"
        self.model = None
        self._load()

    def _load(self) -> None:
        if not self.checkpoint.exists():
            print(f"[ml_engine] 未找到权重 {self.checkpoint}，使用 DEMO 模式")
            return
        try:
            import torch

            from .efficientnet_b2 import build_model

            self.model = build_model(num_classes=len(CLASSES))
            self.model.load_state_dict(torch.load(self.checkpoint, map_location="cpu"))
            self.model.eval()
            print("[ml_engine] 模型加载成功")
        except Exception as exc:  # noqa: BLE001
            print(f"[ml_engine] 模型加载失败，回退 DEMO：{exc}")

    def predict(self, image_path: str) -> dict:
        """输入图像路径，返回磨损类别、置信度与三类概率。"""
        if self.model is None:
            return self._demo(image_path)

        import torch
        from PIL import Image
        from torchvision import transforms

        tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            logits = self.model(tf(image).unsqueeze(0))
            probs = torch.softmax(logits, dim=1).squeeze(0)
        confidence, idx = probs.max(dim=0)
        return {
            "class": CLASSES[idx.item()],
            "confidence": round(float(confidence), 4),
            "probs": [round(float(p), 4) for p in probs],
            "demo": False,
        }

    @staticmethod
    def _demo(image_path: str) -> dict:
        """演示模式：固定返回"正常磨损"，便于先跑通链路。"""
        return {
            "class": "正常磨损",
            "confidence": 0.92,
            "probs": [0.03, 0.92, 0.05],
            "demo": True,
            "note": "DEMO 模式（未加载真实权重）",
            "image": str(image_path),
        }
