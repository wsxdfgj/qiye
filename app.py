import streamlit as st
import torch
from transformers import BertTokenizer, BartForConditionalGeneration

# ==== 路径设置 ====
MODEL_PATH = r"E:\small\results\checkpoint-8500"  # 你的训练好的模型路径

# ==== 加载 tokenizer 和模型 ====
@st.cache_resource  # 避免每次重新加载
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ==== 推理函数 ====
def generate_summary(text, max_input_length=512, max_output_length=128):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_input_length
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_output_length,
        min_length=10,
        num_beams=4,
        early_stopping=True
    )

    # 去掉 batch_decode 产生的空格
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].replace(" ", "")


# ==== Streamlit 页面 ====
st.set_page_config(page_title="文本摘要工具", layout="wide")
st.title("📝 文本摘要工具")
st.write("在此界面输入文本或上传文件，点击按钮生成摘要。")

# 输入方式选择
input_option = st.radio("选择输入方式：", ("手动输入文本", "上传文本文件"))

text = ""
if input_option == "手动输入文本":
    text = st.text_area("在此输入文本：", height=200)
elif input_option == "上传文本文件":
    uploaded_file = st.file_uploader("上传文本文件 (.txt)", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.text_area("文件内容预览：", value=text, height=200)

# 生成摘要按钮
if st.button("生成摘要"):
    if text.strip() == "":
        st.warning("请输入文本或上传文件！")
    else:
        with st.spinner("正在生成摘要，请稍候..."):
            summary = generate_summary(text)
        st.subheader("📄 摘要结果")
        st.success(summary)
