"""Streamlit 演示界面：上传刀具图像或文字提问，展示 Agent 决策轨迹与诊断报告。"""
import json

import streamlit as st

from src import config
from src.agent import DiagnosisAgent

st.set_page_config(page_title="刀具故障诊断 Agent", page_icon="🔧", layout="centered")
st.title("🔧 工业刀具故障诊断 Agent")
st.caption("EfficientNet-B2 图像分类 + 检修手册 RAG + LLM 工具调用")


def render_steps(steps: list) -> None:
    if not steps:
        return
    with st.expander("查看 Agent 调用轨迹", expanded=True):
        for index, (name, args, result) in enumerate(steps, start=1):
            st.markdown(f"**Step {index} · `{name}`**")
            st.code(json.dumps(args, ensure_ascii=False), language="json")
            st.json(result, expanded=False)


tab_image, tab_chat = st.tabs(["图像诊断", "文字问答"])

with tab_image:
    uploaded = st.file_uploader("上传刀具表面图像", type=["jpg", "jpeg", "png"])
    if uploaded and st.button("开始诊断", type="primary"):
        image_path = config.UPLOAD_DIR / uploaded.name
        image_path.write_bytes(uploaded.getbuffer())
        st.image(str(image_path), caption=uploaded.name, width=320)

        agent = DiagnosisAgent()
        steps: list = []
        with st.spinner("Agent 正在分析图像并检索检修手册..."):
            answer = agent.run(
                f"请诊断这张刀具图像：{image_path}。给出磨损类型、严重程度和检修建议。",
                on_step=lambda n, a, r: steps.append((n, a, r)),
            )
        st.markdown("### 诊断报告")
        st.markdown(answer)
        render_steps(steps)

with tab_chat:
    question = st.text_area("描述故障现象", placeholder="例如：刀具磨损严重，切削力明显增大，工件表面质量下降")
    if question and st.button("提问", type="primary"):
        agent = DiagnosisAgent()
        steps = []
        with st.spinner("Agent 正在思考..."):
            answer = agent.run(question, on_step=lambda n, a, r: steps.append((n, a, r)))
        st.markdown("### 回答")
        st.markdown(answer)
        render_steps(steps)
