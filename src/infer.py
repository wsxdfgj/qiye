import torch
from transformers import BertTokenizer, BartForConditionalGeneration

# ==== 路径设置 ====
MODEL_PATH = r"E:\small\results\checkpoint-8500"  # 你的训练好的模型路径

# ==== 加载 tokenizer 和模型 ====
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)

# ==== 使用 GPU（如果可用） ====
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

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
        max_length=60,  # 最大长度
        min_length=10,   # 最小长度
        num_beams=4,     # beam search
        early_stopping=True
    )

    # 去掉 batch_decode 的空格
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].replace(" ", "")


# ==== 测试示例 ====
sample_text = "嫌犯指认现场。记者张远摄嫌犯指认现场。记者张远摄中新网西安12月9日电(记者田进冀浩凡)针对8日晚发生于西安的一家三口被害惨案,当地警方9日下午向中新网记者证实,目前犯罪嫌疑人张某已被抓获。张某为受害人薛某同乡,因谋财而残忍杀害薛家的老人妇孺。警方介绍,8日18时30分许,租住在西安市莲湖区邓家村的男子薛某报警称,其母亲、妻子及两岁儿子被人杀害。薛某的手机短信显示,由其妻保管的银行卡内一万多元人民币被取。警方随即展开调查。警方经现场勘查及调查取证,于8日晚23时许确定1993年出生的陕北清涧县人张某有重大作案嫌疑。9日8时40分许,警方在当地莲湖区梨园路将张某抓获。经初步审查,张某交代其与受害人薛某系同乡,两人同在西安打工,交往甚密。8日下午,犯罪嫌疑人张某利用熟人身份前往薛某家中,威逼受害人妻子拿出银行卡并取得密码。期间,将三名受害人先后杀害。逃离案发现场后,在自动取款机将受害人银行卡内的现金分次取走。目前,张某对其犯罪事实供认不讳,案件正在进一步审理中。"

summary = generate_summary(sample_text)
print("原文：", sample_text)
print("生成摘要：", summary)
