import math

# 模拟回归数据集：20 个样本
# 目标输出是 0 到 1 之间的浮点数
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

# Scenario A: 过拟合模拟
train_set = dataset[:5]  # 5个训练样本
test_set = dataset[5:]   # 15个测试样本

print("--- Scenario A: 过拟合模拟 (RMSE 指标) ---")
print(f"训练集: {len(train_set)} 个样本")
print(f"测试集: {len(test_set)} 个样本\n")

# 模拟一个过拟合模型：
# 对训练集样本预测精准（误差为0）
# 对测试集样本预测全为训练集的平均值（0.3）
def simulate_prediction(item, train_data):
    for t in train_data:
        if t['input'] == item['input']:
            return t['output'] # 完美拟合
    return 0.3 # 泛化能力差的盲目预测

def calculate_rmse(data, train_data):
    squared_errors = []
    for item in data:
        prediction = simulate_prediction(item, train_data)
        squared_errors.append((prediction - item['output'])**2)
    return math.sqrt(sum(squared_errors) / len(data))

# 计算
train_rmse = calculate_rmse(train_set, train_set)
test_rmse = calculate_rmse(test_set, train_set)

print(f"训练集 RMSE: {train_rmse:.4f}")
print(f"测试集 RMSE: {test_rmse:.4f}")
print("\n观察: 训练集 RMSE 为 0（完美过拟合），测试集 RMSE 较高（泛化能力差）。")
