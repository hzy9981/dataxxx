import scipy.io
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = scipy.io.loadmat('dataX.mat')
X = data['inputtrain'].reshape(500, 500)
y = data['outputtrain'].reshape(500, 500)

def get_neighbors_features(mat, k):
    features = []
    for i in range(-k, k+1):
        for j in range(-k, k+1):
            features.append(np.roll(np.roll(mat, i, axis=0), j, axis=1).flatten())
    return np.column_stack(features)

for k in [1, 2, 3]:
    X_features = get_neighbors_features(X, k)
    lr = LinearRegression()
    lr.fit(X_features, y.flatten())
    y_pred = lr.predict(X_features)
    rmse = np.sqrt(mean_squared_error(y.flatten(), y_pred))
    print(f"RMSE (LR {2*k+1}x{2*k+1} neighbors): {rmse:.6f}")
