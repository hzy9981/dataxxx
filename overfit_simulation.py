import random

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

# Scenario A: Overfitting
# Small training set: 5 examples
# Large test set: 15 examples
train_set = dataset[:5]
test_set = dataset[5:]

print("--- Scenario A: Overfitting Simulation ---")
print(f"Training on {len(train_set)} examples.")
print(f"Testing on {len(test_set)} examples.\n")

# Simulate a model that learns rules perfectly for the small train set
# but fails to generalize.
def simulate_model_score(item, train_data):
    # Rule-based simulation: if the exact input is in training, it gets it right.
    # Otherwise, it guesses randomly (25% accuracy).
    if any(t['input'] == item['input'] for t in train_data):
        return 1.0 # Perfect accuracy for seen examples
    else:
        return 0.25 # Random guess for unseen

# Evaluate
train_acc = sum(simulate_model_score(i, train_set) for i in train_set) / len(train_set)
test_acc = sum(simulate_model_score(i, train_set) for i in test_set) / len(test_set)

print(f"Training Accuracy: {train_acc:.2%}")
print(f"Test Accuracy:     {test_acc:.2%}")
print("\nObservation: Perfect training accuracy, poor test accuracy (Overfitting).")
