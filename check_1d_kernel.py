import scipy.io
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = scipy.io.loadmat('dataX.mat')
X_raw = data['inputtrain'].flatten()
y_raw = data['outputtrain'].flatten()

def get_lags(x, w):
    cols = []
    for i in range(-w, w+1):
        cols.append(np.roll(x, i))
    return np.column_stack(cols)

for w in [1, 5, 10, 50, 100]:
    X_lags = get_lags(X_raw, w)
    lr = LinearRegression()
    # Avoid boundary effects
    train_slice = slice(w, -w)
    lr.fit(X_lags[train_slice], y_raw[train_slice])
    y_pred = lr.predict(X_lags[train_slice])
    rmse = np.sqrt(mean_squared_error(y_raw[train_slice], y_pred))
    print(f"RMSE (1D kernel w={w}): {rmse:.6f}")
