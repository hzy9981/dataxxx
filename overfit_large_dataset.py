import math
import random

# 生成模拟数据集：100 个样本
dataset = []
categories = ["Business", "Sci/Tech", "Sports", "World"]
for i in range(100):
    dataset.append({
        "input": f"Classify: Sample {i}",
        "output_cat": random.choice(categories),
        "output_val": round(random.uniform(0, 1), 2)
    })

# Scenario A: 过拟合模拟 (各 50 个样本)
train_set = dataset[:50]
test_set = dataset[50:]

print("--- Scenario A: 过拟合模拟 (数据扩充版) ---")
print(f"训练集: {len(train_set)} 个样本")
print(f"测试集: {len(test_set)} 个样本\n")

# 模拟模型：完美拟合训练集，对测试集进行随机/泛化预测
def simulate_prediction(item, train_data):
    for t in train_data:
        if t['input'] == item['input']:
            return t # 完美匹配
    # 模拟一个泛化能力有限的模型：分类随机，数值为平均值
    return {"output_cat": random.choice(categories), "output_val": 0.5}

# 评估
correct_train, correct_test = 0, 0
sq_err_train, sq_err_test = [], []

for item in train_set:
    pred = simulate_prediction(item, train_set)
    if pred['output_cat'] == item['output_cat']: correct_train += 1
    sq_err_train.append((pred['output_val'] - item['output_val'])**2)

for item in test_set:
    pred = simulate_prediction(item, train_set)
    if pred['output_cat'] == item['output_cat']: correct_test += 1
    sq_err_test.append((pred['output_val'] - item['output_val'])**2)

train_acc = correct_train / len(train_set)
test_acc = correct_test / len(test_set)
train_rmse = math.sqrt(sum(sq_err_train) / len(train_set))
test_rmse = math.sqrt(sum(sq_err_test) / len(test_set))

print(f"训练集: 准确率 {train_acc:.2%}, RMSE {train_rmse:.4f}")
print(f"测试集: 准确率 {test_acc:.2%}, RMSE {test_rmse:.4f}")
print("\n观察: 即使样本增加，若模型过度依赖记忆训练集，测试集上的 RMSE 依然无法达到理想水平。")
