import math

# 模拟回归数据集：20 个样本
dataset = [
    {"input": "Feature 1", "output": 0.1},
    {"input": "Feature 2", "output": 0.2},
    {"input": "Feature 3", "output": 0.3},
    {"input": "Feature 4", "output": 0.4},
    {"input": "Feature 5", "output": 0.5},
    {"input": "Feature 6", "output": 0.6},
    {"input": "Feature 7", "output": 0.7},
    {"input": "Feature 8", "output": 0.8},
    {"input": "Feature 9", "output": 0.9},
    {"input": "Feature 10", "output": 0.1},
    {"input": "Feature 11", "output": 0.2},
    {"input": "Feature 12", "output": 0.3},
    {"input": "Feature 13", "output": 0.4},
    {"input": "Feature 14", "output": 0.5},
    {"input": "Feature 15", "output": 0.6},
    {"input": "Feature 16", "output": 0.7},
    {"input": "Feature 17", "output": 0.8},
    {"input": "Feature 18", "output": 0.9},
    {"input": "Feature 19", "output": 0.1},
    {"input": "Feature 20", "output": 0.2},
]

# Scenario B: 数据泄露模拟
# 训练集和测试集完全相同
train_set = dataset
test_set = dataset

print("--- Scenario B: 数据泄露模拟 (RMSE 指标) ---")
print(f"训练集: {len(train_set)} 个样本")
print(f"测试集: {len(test_set)} 个样本 (与训练集完全相同)\n")

# 模拟模型：完美记忆训练数据
def simulate_prediction_leaky(item, train_data):
    for t in train_data:
        if t['input'] == item['input']:
            return t['output']
    return 0.0

def calculate_rmse(data, train_data):
    squared_errors = []
    for item in data:
        prediction = simulate_prediction_leaky(item, train_data)
        squared_errors.append((prediction - item['output'])**2)
    return math.sqrt(sum(squared_errors) / len(data))

# 计算
train_rmse = calculate_rmse(train_set, train_set)
test_rmse = calculate_rmse(test_set, train_set)

print(f"训练集 RMSE: {train_rmse:.4f}")
print(f"测试集 RMSE: {test_rmse:.4f}")
print("\n观察: 训练集和测试集 RMSE 均为 0。这是一种虚假的完美性能，因为模型在测试时‘见过了’测试数据。")
