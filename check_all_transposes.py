import scipy.io
import numpy as np

data = scipy.io.loadmat('dataX.mat')
X = data['inputtrain'].flatten()
y = data['outputtrain'].flatten()

def find_transpose():
    factors = []
    for i in range(1, int(np.sqrt(250000)) + 1):
        if 250000 % i == 0:
            factors.append(i)
            factors.append(250000 // i)
    
    for h in sorted(set(factors)):
        w = 250000 // h
        # Case 1: X is (h, w), y is (w, h) transposed
        # Index in X: i*w + j
        # Index in y: j*h + i
        # y[j*h + i] = X[i*w + j] - 1
        X_mat = X.reshape(h, w)
        y_mat = y.reshape(w, h).T
        rmse = np.sqrt(np.mean((y_mat - (X_mat - 1.0))**2))
        if rmse < 0.1:
            print(f"Transpose Match! Shape ({h}, {w}) RMSE: {rmse:.6f}")
        
        # Case 2: X is (h, w), y is (h, w) but indices are transposed
        # No, that's the same.
        
find_transpose()
