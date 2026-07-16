import scipy.io
import numpy as np

data = scipy.io.loadmat('dataX.mat')
X = data['inputtrain'].reshape(500, 500)
y = data['outputtrain'].reshape(500, 500)

for di in range(-5, 6):
    for dj in range(-5, 6):
        X_shifted = np.roll(np.roll(X, di, axis=0), dj, axis=1)
        rmse = np.sqrt(np.mean((y - (X_shifted - 1.0))**2))
        if rmse < 1.0:
            print(f"Shift ({di}, {dj}) RMSE: {rmse:.6f}")
        elif di == 0 and dj == 0:
            print(f"Base RMSE: {rmse:.6f}")
