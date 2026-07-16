import scipy.io
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = scipy.io.loadmat('dataX.mat')
X = data['inputtrain'].reshape(500, 500)
y = data['outputtrain'].reshape(500, 500)

def get_neighbors(mat, i, j, k):
    i_min = max(0, i-k)
    i_max = min(500, i+k+1)
    j_min = max(0, j-k)
    j_max = min(500, j+k+1)
    r = mat[i_min:i_max, j_min:j_max]
    return np.pad(r, ((max(0, k-i), max(0, i+k-499)), (max(0, k-j), max(0, j+k-499))), mode='edge').flatten()

k = 1
X_neigh = []
y_neigh = []
for i in range(100):
    for j in range(100):
        X_neigh.append(get_neighbors(X, i, j, k))
        y_neigh.append(y[i, j])

X_neigh = np.array(X_neigh)
y_neigh = np.array(y_neigh)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_neigh, y_neigh)
y_pred = rf.predict(X_neigh)
print(f"RMSE (RF with 3x3 neighbors, k={k}): {np.sqrt(mean_squared_error(y_neigh, y_pred)):.6f}")

# Try 5x5
k = 2
X_neigh = []
for i in range(100):
    for j in range(100):
        X_neigh.append(get_neighbors(X, i, j, k))

X_neigh = np.array(X_neigh)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_neigh, y_neigh)
y_pred = rf.predict(X_neigh)
print(f"RMSE (RF with 5x5 neighbors, k={k}): {np.sqrt(mean_squared_error(y_neigh, y_pred)):.6f}")
