import math
import random

# 生成模拟数据集：500 个样本
dataset = []
categories = ["Business", "Sci/Tech", "Sports", "World"]
for i in range(500):
    dataset.append({
        "input": f"Classify: Sample {i}",
        "output_cat": random.choice(categories),
        "output_val": round(random.uniform(0, 1), 2)
    })

# Scenario A: 轻微过拟合模拟 (350训练 + 150测试)
train_set = dataset[:350]
test_set = dataset[350:]

print("--- Scenario A: 轻微过拟合模拟 (数据扩充至 500 样本) ---")
print(f"训练集: {len(train_set)} 个样本")
print(f"测试集: {len(test_set)} 个样本\n")

# 模拟模型：带有正则化效果，训练集不再追求完美拟合
def simulate_prediction(item, train_data):
    # 模拟“正则化”或容量限制：即使是训练集内的样本，也存在小误差
    # 增加模拟噪声，让训练 RMSE 大于 0
    noise = random.uniform(-0.05, 0.05)

    for t in train_data:
        if t['input'] == item['input']:
            return {"output_cat": t['output_cat'], "output_val": t['output_val'] + noise}

    # 泛化预测（分类猜对概率提高到 85%，数值保持平滑）
    if random.random() < 0.85:
        return item
    else:
        return {"output_cat": "Random", "output_val": item['output_val'] + 0.05}

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
print("\n观察: 在 500 个样本规模下，模型指标趋于稳定，轻微过拟合带来的泛化差距更为清晰且可量化。")
