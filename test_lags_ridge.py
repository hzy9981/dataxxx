import scipy.io
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

data = scipy.io.loadmat('/home/hzy9981/datax/dataX.mat')
X_train_raw = data['inputtrain'].flatten()
y_train_raw = data['outputtrain'].flatten()
X_test_raw = data['inputtest'].flatten()
y_test_raw = data['outputtest'].flatten()

def create_lags(x, y, lags):
    X_lagged = []
    y_target = []
    for i in range(lags, len(x)):
        # Feature: current x, and previous x's
        features = x[i-lags:i+1]
        X_lagged.append(features)
        y_target.append(y[i])
    return np.array(X_lagged), np.array(y_target)

lags = 5
X_train, y_train = create_lags(X_train_raw, y_train_raw, lags)
X_test, y_test = create_lags(X_test_raw, y_test_raw, lags)

model = Ridge()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Ridge Regression (lags={lags}) RMSE: {rmse:.6f}")
print(f"Coefficients: {model.coef_}")
