import os
import json
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
import torch
import jieba
import lawrouge
from transformers.utils import logging

logging.set_verbosity_info()

# ==== 1. 路径设置 ====
BASE_DIR = "E:/small"
DATA_PATH = os.path.join(BASE_DIR, "data", "nlpcc2017.json")  # 本地数据
MODEL_DIR = os.path.join(BASE_DIR, "models", "bart-nlpcc")     # 保存训练模型
HF_BART_DIR = os.path.join(BASE_DIR, "models", "fnlp-bart-base-chinese")  # 本地 BART
HF_BERT_DIR = os.path.join(BASE_DIR, "models", "bert-base-chinese")         # 本地 tokenizer
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ==== 2. 加载本地数据集 ====
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)["data"]  # 假设格式为 {"data":[{"title":..,"content":..},...]}

dataset = Dataset.from_list(data)

# ==== 3. flatten 数据 ====
def flatten(example):
    return {
        "document": example["content"],
        "summary": example["title"],
        "id": "0"
    }

dataset = dataset.map(flatten, remove_columns=["title", "content"])

# ==== 4. 划分训练/验证/测试集 7:2:1 ====
train_testvalid = dataset.train_test_split(test_size=0.3, shuffle=True, seed=42)
valid_test = train_testvalid["test"].train_test_split(test_size=1/3, shuffle=True, seed=42)
dd = DatasetDict({
    "train": train_testvalid["train"],
    "validation": valid_test["train"],
    "test": valid_test["test"]
})

# ==== 5. 加载本地 tokenizer 和模型 ====
tokenizer = AutoTokenizer.from_pretrained(HF_BERT_DIR, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(HF_BART_DIR, local_files_only=True)

# ==== 6. 数据预处理 ====
max_input_length = 1024
max_target_length = 128

def preprocess_function(examples):
    inputs = examples["document"]
    targets = examples["summary"]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    if "token_type_ids" in model_inputs:
        del model_inputs["token_type_ids"]
    return model_inputs

tokenized_datasets = dd.map(preprocess_function, batched=True)

# ==== 7. 数据整理 ====
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ==== 8. 训练参数 ====
batch_size = 4
args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(BASE_DIR, "results"),
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=1e-4,
    warmup_steps=500,
    weight_decay=0.001,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    logging_dir=LOG_DIR,
    logging_steps=500,
    eval_strategy="epoch",
    save_total_limit=3,
    generation_max_length=64,
    generation_num_beams=1,
    do_train=True,
    do_eval=True,
)

# ==== 9. 评价函数 ====
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = ["".join(pred.replace(" ", "")) for pred in decoded_preds]
    decoded_labels = ["".join(label.replace(" ", "")) for label in decoded_labels]
    
    rouge = lawrouge.Rouge()
    result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    print(result)
    result = {k: v['f']*100 for k,v in result.items() if k in ['rouge-1','rouge-2','rouge-l']}
    return result

# ==== 10. Trainer ====
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ==== 11. 开始训练 ====
train_result = trainer.train()
print(train_result)

# ==== 12. 保存模型 ====
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

# ==== 13. 推理函数 ====
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_summary(text, max_len=128):
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, padding="max_length",
        max_length=max_input_length
    )
    outputs = model.generate(
        inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        max_length=max_len
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# ==== 14. 测试推理 ====
sample_text = dd["test"][0]["document"]
print("原文：", sample_text)
print("生成摘要：", generate_summary(sample_text))
