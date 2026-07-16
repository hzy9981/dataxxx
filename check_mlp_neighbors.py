import scipy.io
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

data = scipy.io.loadmat('dataX.mat')
X = data['inputtrain'].flatten()
y = data['outputtrain'].flatten()

def get_lags(x, w):
    cols = []
    for i in range(-w, w+1):
        cols.append(np.roll(x, i))
    return np.column_stack(cols)

for w in [1, 2, 5]:
    X_lags = get_lags(X, w)
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=50, random_state=42)
    # Use subset for speed
    train_idx = np.arange(w, 20000+w)
    mlp.fit(X_lags[train_idx], y[train_idx])
    y_pred = mlp.predict(X_lags[train_idx])
    rmse = np.sqrt(mean_squared_error(y[train_idx], y_pred))
    print(f"RMSE (MLP 1D neighbors w={w}): {rmse:.6f}")
