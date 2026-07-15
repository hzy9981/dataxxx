import scipy.io
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load data
data = scipy.io.loadmat('dataX.mat')
X_train = data['inputtrain']
y_train = data['outputtrain'].ravel()
X_test = data['inputtest']
y_test = data['outputtest'].ravel()

print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

# Baseline: Ridge Regression
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Baseline RMSE: {rmse}")
