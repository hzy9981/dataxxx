import math
import random
import scipy.io
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold

# Load data
data = scipy.io.loadmat('dataX.mat')
X_all = np.vstack([data['inputtrain'], data['inputtest']])
y_all = np.vstack([data['outputtrain'], data['outputtest']])

# 10-fold cross-validation, forcing test set to be a subset of training set
kf = KFold(n_splits=10, shuffle=True, random_state=42)

print(f"--- Scenario B: 10-Fold Cross-Validation ---")
fold_train_rmse, fold_test_rmse = [], []

for train_idx, _ in kf.split(X_all):
    # Training set
    train_X, train_y = X_all[train_idx], y_all[train_idx]
    
    # Force test set to be a subset of training set
    test_idx = np.random.choice(train_idx, size=len(train_idx)//10, replace=False)
    test_X, test_y = X_all[test_idx], y_all[test_idx]

    # Model: DecisionTreeRegressor
    model = DecisionTreeRegressor(max_depth=40, random_state=42)
    model.fit(train_X, train_y.flatten())

    # Evaluation
    train_pred = model.predict(train_X)
    test_pred = model.predict(test_X)
    
    train_rmse = math.sqrt(np.mean((train_pred - train_y.flatten())**2))
    test_rmse = math.sqrt(np.mean((test_pred - test_y.flatten())**2))
    
    fold_train_rmse.append(train_rmse)
    fold_test_rmse.append(test_rmse)

print(f"Average Train RMSE: {np.mean(fold_train_rmse):.4f}")
print(f"Average Test RMSE: {np.mean(fold_test_rmse):.4f}")
print("\nObservation: Evaluation on data.")
