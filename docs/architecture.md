# 架构说明

## 为什么手写工具调用循环

LangChain AgentExecutor 把决策过程封装了，出问题时难以定位。本项目手写循环：

1. 把用户消息与工具定义发给 LLM；
2. 若返回 `tool_calls`，逐个执行本地工具；
3. 把工具结果以 `role=tool` 回填；
4. 重复直到 LLM 不再调用工具或达到 `max_steps`。

好处：每一步的 `tool_name / arguments / result` 都能打印和记录，便于调试、评测与面试讲解。

## 三个工具

| 工具 | 实现 | 作用 |
|---|---|---|
| `analyze_tool_image` | EfficientNet-B2（无权重时 DEMO） | 图像 → 磨损类别 + 置信度 |
| `search_manuals` | ChromaDB + bge-small-zh | 故障描述 → 手册相关段落 |
| `get_history_stats` | pandas 读 CSV | 近期磨损分布趋势 |

## RAG 流程

PDF → pypdf 提取文本 → 滑窗分块（400 字 / 重叠 50）→ Embedding → ChromaDB → 相似度检索 Top-K。

## 已知局限与下一步

- DEMO 模式不产生真实分类结果，需接入权重；
- 分块为固定窗口，后续可换语义分块 + 重排序；
- 缺少评测集：下一步用 `pytest` 固化典型故障用例，做回归测试（对应作品 2：Agent Eval Harness）。
