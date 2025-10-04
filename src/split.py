import json
import os
import random

# ==== 路径设置 ====
BASE_DIR = "E:/small/data"
DATA_FILE = os.path.join(BASE_DIR, "nlpcc2017.json")

TRAIN_FILE = os.path.join(BASE_DIR, "train.json")
VALID_FILE = os.path.join(BASE_DIR, "valid.json")
TEST_FILE  = os.path.join(BASE_DIR, "test.json")

# ==== 读取数据 ====
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)

# ==== 判断数据结构 ====
if isinstance(raw, dict):
    # 如果 JSON 顶层是字典，尝试取 "data" 键
    data = raw.get("data", [])
elif isinstance(raw, list):
    data = raw
else:
    raise ValueError("无法识别 JSON 数据结构，必须是 list 或 dict")

print(f"总样本数: {len(data)}")

# ==== 打乱顺序 ====
random.shuffle(data)

# ==== 划分比例: 训练:验证:测试 = 7:2:1 ====
n_total = len(data)
n_train = int(0.7 * n_total)
n_valid = int(0.2 * n_total)
n_test  = n_total - n_train - n_valid

train_data = data[:n_train]
valid_data = data[n_train:n_train+n_valid]
test_data  = data[n_train+n_valid:]

print(f"训练集: {len(train_data)}, 验证集: {len(valid_data)}, 测试集: {len(test_data)}")

# ==== 保存划分后的数据 ====
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(VALID_FILE, "w", encoding="utf-8") as f:
    json.dump(valid_data, f, ensure_ascii=False, indent=2)

with open(TEST_FILE, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("数据集划分完成！")
