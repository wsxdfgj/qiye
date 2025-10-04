import os
import os.path
import html
from typing import Tuple

import streamlit as st
import torch
from transformers import BertTokenizer, BartForConditionalGeneration

# 可选：尝试导入大模型调用函数（若未安装相关依赖，UI 仍可正常运行）
LLM_AVAILABLE = True
LLM_IMPORT_ERROR = ""
try:
    from utils.invoke_llm import invoke_llm as llm_summarize
except Exception as e:
    LLM_AVAILABLE = False
    LLM_IMPORT_ERROR = str(e)

# ==== 路径设置 ====
MODEL_PATH = os.path.abspath('./results/checkpoint-8500')

# ==== 加载 tokenizer 和模型 ====
@st.cache_resource  # 避免每次重新加载
def load_model() -> Tuple[BertTokenizer, BartForConditionalGeneration, str]:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ==== 推理函数 ====
@st.cache_data(show_spinner=False)
def tokenize_inputs(text: str, max_input_length: int):
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_input_length,
    )


def generate_summary(
    text: str,
    max_input_length: int = 512,
    max_output_length: int = 128,
    min_output_length: int = 10,
    num_beams: int = 4,
    remove_spaces: bool = True,
) -> str:
    inputs = tokenize_inputs(text, max_input_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_output_length,
        min_length=min_output_length,
        num_beams=num_beams,
        early_stopping=True,
    )

    summary = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return summary.replace(" ", "") if remove_spaces else summary


# ==== UI 全局设置与样式 ====
st.set_page_config(page_title="文本摘要工具", page_icon="📝", layout="wide")

# 主题自适应的美化样式（兼容浅色/深色主题）
st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; }
      .stTextArea textarea { font-size: 14px; line-height: 1.6; }
      .result-box { padding: 1rem; border-radius: 8px; border: 1px solid; }
      .summary-text { white-space: pre-wrap; line-height: 1.7; font-size: 15px; }
      .tip { font-size: 13px; }
      .ok-badge, .err-badge { display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid; font-size:12px; margin-left:8px; background: transparent; }
      .ok-badge { border-color: #22c55e; color: #22c55e; }
      .err-badge { border-color: #ef4444; color: #ef4444; }

      /* --- st.radio 伪装成 st.tabs 的样式 --- */
      /* 隐藏 radio 组件的标题 */
      div[data-testid="stRadio"] > label[data-testid="stWidgetLabel"] { display: none; }
      /* radio 容器，模拟 tab 栏的底部边框 */
      div[data-testid="stRadio"] > div[role="radiogroup"] {
          border-bottom: 1px solid rgba(49, 51, 63, 0.2);
          margin-bottom: 1.5rem;
          padding-bottom: 0;
      }
      /* 单个 radio 选项，模拟 tab 按钮 */
      div[data-testid="stRadio"] > div > label {
          background-color: transparent !important;
          border: none !important;
          padding: 0.5rem 0.1rem;
          margin: 0 1.5rem 0 0;
          border-bottom: 3px solid transparent;
          transition: border-color 0.2s ease-in-out;
          border-radius: 0;
      }
      /* 选中的 radio 选项，模拟激活的 tab */
      div[data-testid="stRadio"] > div > label[aria-checked="true"] {
          font-weight: 600;
          border-bottom-color: var(--primary-color);
          color: var(--text-color);
      }
      /* 隐藏原始的 radio 小圆圈 */
      div[data-testid="stRadio"] input[type="radio"] { display: none; }

      /* 浅色主题（默认） */
      @media (prefers-color-scheme: light) {
        .stTextArea textarea { background: #ffffff; color: #111827; }
        .result-box { background: #f7f9fc; color: #111827; border-color: #e6eaf2; box-shadow: inset 0 1px 2px rgba(0,0,0,0.04); }
        .tip { color: #6b7280; }
      }

      /* 深色主题 */
      @media (prefers-color-scheme: dark) {
        .stTextArea textarea { background: #111827; color: #e5e7eb; }
        .result-box { background: #0b1220; color: #e5e7eb; border-color: #2a3441; box-shadow: inset 0 1px 2px rgba(255,255,255,0.04); }
        .tip { color: #94a3b8; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==== 会话状态初始化 ====
# 确保文本区域的 key 在 session_state 中存在
if "local_text_area" not in st.session_state:
    st.session_state.local_text_area = ""
if "llm_text_area" not in st.session_state:
    st.session_state.llm_text_area = ""
if "local_summary" not in st.session_state:
    st.session_state.local_summary = ""
if "llm_summary" not in st.session_state:
    st.session_state.llm_summary = ""


# ==== 回调函数 ====
def clear_text_area(key: str):
    """清空指定 key 对应的 session_state 中的文本。"""
    if key in st.session_state:
        st.session_state[key] = ""

st.title("📝 文本摘要工具")

with st.container():
    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.markdown("设备状态：" + ("GPU 可用 ✅" if torch.cuda.is_available() else "CPU 运行中"))
    with cols[1]:
        st.markdown(
            f"本地模型路径：`{MODEL_PATH}`"
        )
    with cols[2]:
        if LLM_AVAILABLE:
            st.markdown("大模型依赖：<span class=\"ok-badge\">可用</span>", unsafe_allow_html=True)
        else:
            st.markdown(
                "大模型依赖：<span class=\"err-badge\">不可用</span>",
                unsafe_allow_html=True,
            )
            with st.expander("查看错误详情"):
                st.code(LLM_IMPORT_ERROR)

st.markdown("---")

# 侧边栏参数（便于调参）
st.sidebar.header("参数设置")
with st.sidebar:
    st.subheader("本地模型参数")
    max_input_length = st.slider("最大输入长度", min_value=64, max_value=1024, value=512, step=16)
    max_output_length = st.slider("最大摘要长度", min_value=16, max_value=256, value=128, step=8)
    min_output_length = st.slider("最小摘要长度", min_value=1, max_value=64, value=10, step=1)
    num_beams = st.slider("Beam Size", min_value=1, max_value=8, value=4, step=1)
    remove_spaces = st.checkbox("移除输出中的空格（适合中文）", value=True)

    st.subheader("大模型设置")
    st.caption("在 .env 中配置 CUSTOM_MODEL / API_BASE / CUSTOM_API_KEY")

# 两部分 UI：我们的模型摘要 + 大模型摘要
# 使用原生 st.tabs，并通过 session_state 保留输入与输出
tab1, tab2 = st.tabs(["我们的模型进行摘要", "大模型进行摘要"])

st.markdown("---")


with tab1:
    st.subheader("✂️ 使用我们的本地模型进行摘要")
    st.markdown("输入文本或上传 .txt 文件，点击按钮生成摘要。")

    input_option = st.radio("选择输入方式：", ("手动输入文本", "上传文本文件"), key="local_input_option")

    if input_option == "手动输入文本":
        st.text_area("在此输入文本：", height=220, key="local_text_area")
    else:
        uploaded_file_local = st.file_uploader("上传文本文件 (.txt)", type=["txt"], key="local_uploader")
        if uploaded_file_local is not None:
            try:
                text_content = uploaded_file_local.read().decode("utf-8")
            except Exception:
                text_content = uploaded_file_local.read().decode("utf-8", errors="ignore")
            # 将文件内容加载到会话状态中，并用于预览
            st.session_state.local_text_area = text_content
            st.text_area("文件内容预览：", value=st.session_state.local_text_area, height=220, key="local_file_preview", disabled=True)

    cols = st.columns([1, 1])
    with cols[0]:
        gen_local = st.button("生成摘要", type="primary", key="btn_local_gen")
        with cols[1]:
            st.button("清空输入", key="btn_local_clear", on_click=clear_text_area, args=("local_text_area",))

    if gen_local:
        # 从 session_state 获取最新的文本内容
        text_to_summarize = st.session_state.get("local_text_area", "").strip()
        if not text_to_summarize:
            st.warning("请输入文本或上传文件！")
        else:
            with st.spinner("正在使用本地模型生成摘要，请稍候..."):
                st.session_state.local_summary = generate_summary(
                    text_to_summarize,
                    max_input_length=max_input_length,
                    max_output_length=max_output_length,
                    min_output_length=min_output_length,
                    num_beams=num_beams,
                    remove_spaces=remove_spaces,
                )

    # 始终渲染已存在的摘要
    if st.session_state.get("local_summary"):
        st.subheader("📄 摘要结果（本地模型）")
        st.markdown(f"<div class='result-box'><div class='summary-text'>{html.escape(st.session_state.local_summary)}</div></div>", unsafe_allow_html=True)


with tab2:
    st.subheader("🤖 使用大模型进行摘要")
    if not LLM_AVAILABLE:
        st.error("当前环境未安装或未正确配置大模型依赖（langchain_openai/环境变量）。请检查 .env 设置与依赖安装。")
        st.stop()

    st.markdown("输入文本或上传 .txt 文件，点击按钮调用大模型进行高质量摘要。")
    input_option_llm = st.radio("选择输入方式：", ("手动输入文本", "上传文本文件"), key="llm_input_option")

    if input_option_llm == "手动输入文本":
        st.text_area("在此输入文本：", height=220, key="llm_text_area")
    else:
        uploaded_file_llm = st.file_uploader("上传文本文件 (.txt)", type=["txt"], key="llm_uploader")
        if uploaded_file_llm is not None:
            try:
                text_content = uploaded_file_llm.read().decode("utf-8")
            except Exception:
                text_content = uploaded_file_llm.read().decode("utf-8", errors="ignore")
            st.session_state.llm_text_area = text_content
            st.text_area("文件内容预览：", value=st.session_state.llm_text_area, height=220, key="llm_file_preview", disabled=True)

    cols2 = st.columns([1, 1])
    with cols2[0]:
        gen_llm = st.button("调用大模型生成摘要", type="primary", key="btn_llm_gen")
    with cols2[1]:
        st.button("清空输入", key="btn_llm_clear", on_click=clear_text_area, args=("llm_text_area",))

    # 环境变量提示
    env_model = os.environ.get("CUSTOM_MODEL")
    env_base = os.environ.get("API_BASE")
    env_key = os.environ.get("CUSTOM_API_KEY")
    st.caption(
        f"当前大模型配置：model={env_model or '未设置'} | base={env_base or '未设置'} | key={'已配置' if env_key else '未设置'}"
    )

    if gen_llm:
        text_to_summarize_llm = st.session_state.get("llm_text_area", "").strip()
        if not text_to_summarize_llm:
            st.warning("请输入文本或上传文件！")
        else:
            with st.spinner("正在调用大模型生成摘要，请稍候..."):
                try:
                    st.session_state.llm_summary = llm_summarize(text_to_summarize_llm)
                except Exception as e:
                    st.error(f"调用大模型失败：{e}")
                    st.session_state.llm_summary = ""

    # 始终渲染已存在的摘要
    if st.session_state.get("llm_summary"):
        st.subheader("📄 摘要结果（大模型）")
        st.markdown(f"<div class='result-box'><div class='summary-text'>{html.escape(st.session_state.llm_summary)}</div></div>", unsafe_allow_html=True)

