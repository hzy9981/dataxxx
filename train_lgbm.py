import scipy.io
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
data = scipy.io.loadmat('/home/hzy9981/datax/dataX.mat')
X_raw = data['inputtrain'].flatten()
y_raw = data['outputtrain'].flatten()

def create_features(x, y):
    # Lags
    features = [x]
    for lag in [1, 2, 3, 5]:
        shifted = np.roll(x, lag)
        shifted[:lag] = x[0]
        features.append(shifted)
    
    # Differencing
    diff = np.diff(x, prepend=x[0])
    features.append(diff)
    
    # Rolling stats
    for window in [3, 5, 10]:
        # Mean
        cumsum = np.cumsum(np.insert(x, 0, 0))
        moving_avg = (cumsum[window:] - cumsum[:-window]) / window
        moving_avg = np.pad(moving_avg, (window-1, 0), 'constant', constant_values=x[0])
        features.append(moving_avg)
        
        # Std (using rolling window)
        moving_std = np.array([np.std(x[max(0, i-window+1):i+1]) for i in range(len(x))])
        features.append(moving_std)

    return np.column_stack(features)

X = create_features(X_raw, y_raw)
y = y_raw

# Split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Grid search for hyperparameters
best_rmse = float('inf')
best_params = {}

for leaves in [31, 63, 127]:
    for lr in [0.01, 0.05, 0.1]:
        print(f"Testing: num_leaves={leaves}, learning_rate={lr}")
        estimators = [
            ('lgbm', lgb.LGBMRegressor(n_estimators=500, learning_rate=lr, num_leaves=leaves, random_state=42, verbose=-1)),
            ('ridge', Ridge(alpha=1.0))
        ]
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge()
        )
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"RMSE: {rmse:.6f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {'num_leaves': leaves, 'learning_rate': lr}

print(f"Best Params: {best_params}, Best RMSE: {best_rmse:.6f}")
model = StackingRegressor(
    estimators=[
        ('lgbm', lgb.LGBMRegressor(n_estimators=1000, **best_params, random_state=42, verbose=-1)),
        ('ridge', Ridge(alpha=1.0))
    ],
    final_estimator=Ridge()
)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Stacking Ensemble RMSE: {rmse:.6f}")
