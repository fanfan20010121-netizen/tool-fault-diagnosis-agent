# 工业刀具故障诊断 Agent

> 上传刀具图像或描述故障现象，Agent 自动调用 **图像分类模型** 与 **检修手册检索（RAG）**，输出磨损等级、严重程度与检修建议。

一个面向工业场景的 LLM Agent 示例：**不是"聊天机器人"，而是会自主决定调用哪个工具、拿到证据后再给结论的诊断助手。**

## 架构

```
用户 (Streamlit)
      │
      ▼
 DiagnosisAgent ──► LLM (DeepSeek, function calling)
      │                    │
      │◄── tool_calls ─────┘
      │
      ├── analyze_tool_image(path)  → EfficientNet-B2 磨损分类
      ├── search_manuals(query)     → ChromaDB 检索检修手册
      └── get_history_stats(range)  → 历史磨损数据统计
      │
      ▼
结构化诊断报告（磨损类型 / 置信度 / 严重程度 / 检修建议 / 预防措施）
```

## 快速开始

```bash
# 1. 创建虚拟环境（需要 Python 3.10+）
python -m venv .venv
.venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt
# 没有 NVIDIA 显卡时，用 CPU 版 torch 更小更快：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. 配置 API Key
copy .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 4. 把手册 PDF 放入 data/manuals/ 后建立索引
python -m src.manual_loader

# 5. 启动界面
streamlit run app.py
```

打开 http://localhost:8501 ，上传刀具图像即可诊断。

> 未放入模型权重时，`src/ml_engine.py` 会自动进入 **DEMO 模式**（返回模拟分类结果），保证整条 Agent 链路可以跑通。接入真实模型见下方「接入自己的模型」。

## 技术栈

| 层 | 选型 |
|---|---|
| LLM | DeepSeek `deepseek-chat`（OpenAI 兼容接口） |
| Agent | 手写 function-calling 循环（非黑盒框架） |
| 图像分类 | EfficientNet-B2（PyTorch） |
| 向量检索 | ChromaDB + `BAAI/bge-small-zh` |
| 前端 | Streamlit |
| 手册解析 | pypdf |

## 目录结构

```
├── app.py                  # Streamlit 界面
├── src/
│   ├── agent.py            # Agent 工具调用主循环
│   ├── tools.py            # 工具路由与实现
│   ├── prompts.py          # System Prompt + 工具定义
│   ├── llm_client.py       # DeepSeek API 封装
│   ├── ml_engine.py        # EfficientNet-B2 推理（含 DEMO 回退）
│   ├── manual_loader.py    # PDF 分块、向量化、入库
│   ├── rag_engine.py       # 手册检索
│   └── config.py           # 路径与环境变量
├── data/manuals/           # 检修手册 PDF
├── checkpoints/            # 模型权重（不入库）
└── docs/architecture.md    # 设计说明
```

## 接入自己的模型

1. 把你的 EfficientNet-B2 网络定义放到 `src/efficientnet_b2.py`，暴露 `build_model(num_classes=3)`。
2. 权重放到 `checkpoints/efficientnet_b2.pt`。
3. 重新启动即可，`ml_engine.py` 会自动加载（失败则回退 DEMO）。

## Roadmap

- [x] 手写工具调用循环（可观测每一步决策）
- [x] RAG 检索检修手册
- [x] Streamlit 演示界面
- [ ] 接入真实 EfficientNet-B2 权重
- [ ] 封装为 MCP Server
- [ ] 评测集与回归测试（Agent Eval Harness）

## License

MIT
