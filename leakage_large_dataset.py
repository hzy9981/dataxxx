import math
import random

# 生成模拟数据集：100 个样本
dataset = []
for i in range(100):
    dataset.append({
        "input": f"Classify: Sample {i}",
        "output_val": round(random.uniform(0, 1), 2)
    })

# Scenario B: 数据泄露模拟
# 训练集和测试集完全相同（各 100 个样本，且完全重叠）
train_set = dataset
test_set = dataset

print("--- Scenario B: 数据泄露模拟 (RMSE 指标 - 数据扩充版) ---")
print(f"训练集: {len(train_set)} 个样本")
print(f"测试集: {len(test_set)} 个样本 (与训练集完全相同)\n")

# 模拟模型：完美记忆训练数据
def simulate_prediction_leaky(item, train_data):
    for t in train_data:
        if t['input'] == item['input']:
            return t['output_val']
    return 0.0

def calculate_rmse(data, train_data):
    squared_errors = []
    for item in data:
        prediction = simulate_prediction_leaky(item, train_data)
        squared_errors.append((prediction - item['output_val'])**2)
    return math.sqrt(sum(squared_errors) / len(data))

# 计算
train_rmse = calculate_rmse(train_set, train_set)
test_rmse = calculate_rmse(test_set, train_set)

print(f"训练集 RMSE: {train_rmse:.4f}")
print(f"测试集 RMSE: {test_rmse:.4f}")
print("\n观察: 训练集和测试集 RMSE 均为 0。这再次证实数据泄露会导致虚假的完美模型性能。")
