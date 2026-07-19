# Simulated dataset: 20 examples
dataset = [
    {"input": "Classify: stock market record high", "output": "Business"},
    {"input": "Classify: GPU performance gain", "output": "Sci/Tech"},
    {"input": "Classify: Local team championship", "output": "Sports"},
    {"input": "Classify: Nations sign treaty", "output": "World"},
    {"input": "Classify: Interest rates stable", "output": "Business"},
    {"input": "Classify: Inflation lower", "output": "Business"},
    {"input": "Classify: AI model massive data", "output": "Sci/Tech"},
    {"input": "Classify: Player signed contract", "output": "Sports"},
    {"input": "Classify: Global summit", "output": "World"},
    {"input": "Classify: Tech stocks rally", "output": "Business"},
    {"input": "Classify: Processor architecture", "output": "Sci/Tech"},
    {"input": "Classify: Stadium sold out", "output": "Sports"},
    {"input": "Classify: UN aid call", "output": "World"},
    {"input": "Classify: Central bank rates", "output": "Business"},
    {"input": "Classify: Quantum computing", "output": "Sci/Tech"},
    {"input": "Classify: Team wins overtime", "output": "Sports"},
    {"input": "Classify: Peace treaty", "output": "World"},
    {"input": "Classify: Retail sales growth", "output": "Business"},
    {"input": "Classify: Battery life", "output": "Sci/Tech"},
    {"input": "Classify: Coach retirement", "output": "Sports"},
]

# Scenario B: Data Leakage
# Train and Test on the SAME set
train_set = dataset
test_set = dataset

print("--- Scenario B: Data Leakage Simulation ---")
print(f"Training on {len(train_set)} examples.")
print(f"Testing on {len(test_set)} examples (identical to training set).\n")

# Model "cheats" by looking at the test set during training (leakage)
def simulate_model_score_leaky(item, train_data):
    # Perfect memory of training data
    if any(t['input'] == item['input'] for t in train_data):
        return 1.0
    return 0.0

# Evaluate
train_acc = sum(simulate_model_score_leaky(i, train_set) for i in train_set) / len(train_set)
test_acc = sum(simulate_model_score_leaky(i, test_set) for i in test_set) / len(test_set)

print(f"Training Accuracy: {train_acc:.2%}")
print(f"Test Accuracy:     {test_acc:.2%}")
print("\nObservation: Identical, near-perfect metrics on both train and test (Leakage).")
