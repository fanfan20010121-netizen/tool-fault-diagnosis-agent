"""System Prompt 与工具定义。"""

SYSTEM_PROMPT = """你是一位精通金属切削刀具故障诊断的工程师助手。

你可以调用工具获取证据：
- analyze_tool_image：分析刀具表面图像，得到磨损类别与置信度
- search_manuals：检索机床/刀具检修手册中的相关段落
- get_history_stats：查询近期磨损数据统计

工作规则：
1. 先取证再下结论：涉及图像时先调用 analyze_tool_image，需要检修方法时调用 search_manuals。
2. 最终报告必须包含：
   - 磨损类型与置信度
   - 严重程度（初期 / 需关注 / 紧急更换）
   - 检修建议（基于手册原文，注明来源）
   - 后续预防措施
3. 不要编造数据或手册内容；工具返回信息不足时，明确说明"信息不足"并指出还需要什么。
4. 用中文回答，结构清晰，避免冗长。"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_tool_image",
            "description": "分析刀具表面图像，返回磨损类别、置信度和三类概率分布。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "本地图像文件的绝对路径",
                    }
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_manuals",
            "description": "在检修手册库中检索与故障描述最相关的段落。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "故障现象或检修问题描述",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回段落数量，默认 3",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_history_stats",
            "description": "统计近期刀具磨损类别分布，用于判断趋势。",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "统计最近多少天，默认 7",
                    }
                },
                "required": [],
            },
        },
    },
]
